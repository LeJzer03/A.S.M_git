import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    "mathtext.fontset": "cm",
    "figure.autolayout": True,
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
})

def read_ascii(path: Path) -> np.ndarray:
    try:
        data = np.genfromtxt(str(path), comments="#")
    except Exception:
        data = np.loadtxt(str(path))
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def load_master_time(folder: Path) -> np.ndarray:
    """Load time.ascii from folder and return the 1D time array (float)."""
    tfile = folder / "time.ascii"
    if not tfile.exists():
        raise FileNotFoundError(f"Required file not found: {tfile}")
    data = read_ascii(tfile)
    t = data[:, 0].astype(float)
    if not np.all(np.isfinite(t)):
        raise ValueError("time.ascii contains non-finite values.")
    return t


def infer_ylabel(name: str) -> str:
    unit_map = {
        "SigmaVM": "[MPa]",
        "Sigma_Yield": "[MPa]",
    }
    # Exact matches first
    if name in unit_map:
        return f"{name} {unit_map[name]}"

    # Prefix matches
    if name.startswith("Sigma_"):
        return f"{name} [MPa]"
    if name.startswith("A_"):           # backstress components
        return f"{name} [MPa]"
    if name.startswith("E_"):           # elastic strains
        return f"{name} [-]"
    if name == "EPL":                   # equivalent plastic strain
        return f"{name} [-]"
    return name


def analyze_simulation(folder: str):
    folder_path = Path(folder).resolve()
    if not (folder_path.exists() and folder_path.is_dir()):
        raise NotADirectoryError(f"Folder not found: {folder_path}")

    # 1) master time
    t_master = load_master_time(folder_path)
    nT = len(t_master)

    # 2) collect series
    ascii_files = sorted(folder_path.glob("*.ascii"))
    series = {}  # name -> np.ndarray aligned on t_master (float, len nT)

    for f in ascii_files:
        if f.name.lower() == "time.ascii":
            continue  # skip master time file

        name = f.stem
        data = read_ascii(f)

        if data.shape[1] >= 2:
            # Two columns: assume [time, value]
            t_local = data[:, 0].astype(float)
            y_local = data[:, 1].astype(float)
            # Interpolate onto master time grid
            # Handle monotonically increasing requirement
            # If not strictly monotonic, we can sort by time
            order = np.argsort(t_local)
            t_sorted = t_local[order]
            y_sorted = y_local[order]
            y_interp = np.interp(t_master, t_sorted, y_sorted, left=np.nan, right=np.nan)
            series[name] = y_interp
        else:
            # One column: value-only; align by length
            y_local = data[:, 0].astype(float)
            y = np.full(nT, np.nan, dtype=float)
            n = min(nT, len(y_local))
            y[:n] = y_local[:n]
            series[name] = y

    if not series:
        raise FileNotFoundError(f"No variable *.ascii files found in {folder_path} besides time.ascii")

    # 3) DataFrame indexed by time
    df = pd.DataFrame(series, index=t_master)
    df.index.name = "time [s]"

    # 4) Output directory and CSV
    out_dir = Path.cwd() / f"plots_{folder_path.name}"
    out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = out_dir / "aggregated_results.csv"
    df.to_csv(csv_path)

    # Optional: order columns for readability
    order_hint = [
        "Sigma_XX", "Sigma_YY", "Sigma_ZZ", "SigmaVM", "Sigma_Yield",
        "E_XX", "E_YY", "E_ZZ", "EPL",
        "A_XX", "A_YY", "A_ZZ",
    ]
    ordered = [c for c in order_hint if c in df.columns] + [c for c in df.columns if c not in order_hint]
    df = df[ordered]

    # 5) Plot each variable vs time (one figure per variable; no subplots, no color styles)
    x = df.index.values
    for col in df.columns:
        plt.figure()
        plt.plot(x, df[col].values)
        plt.xlabel(df.index.name if df.index.name else "time [s]")
        plt.ylabel(infer_ylabel(col))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        fig_path = out_dir / f"{col}.pdf"
        plt.savefig(fig_path)  
        plt.close()

    print(f"Saved aggregated CSV: {csv_path}")
    print(f"Saved plots (PDF) to: {out_dir}")
    return df, out_dir, csv_path


def main():
    # place your simulation folder path here
    sim_folder = r"C:\Ecole\MASTER - 1\Advanced Solid Mecanics\Project\project_file\test_4\workspace\CubeSurfaceTraction"
    analyze_simulation(sim_folder)


if __name__ == "__main__":
    main()