import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

# Palette 1
palette_1 = {
    "Med Blue": "#3594cc",
    "Light Blue": "#8cc5e3",
    "Med Orange": "#ea801c",
    "Light Orange": "#f0b077"
}
# Palette 2
palette_2 = {
    "Dull Purple": "#5e4c5f",
    "Med Gray":    "#999999",
    "Gold":        "#ffbb6f",
    "Slate Blue":  "#6c8ea0"
}

chosen_palette = palette_1  

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "figure.autolayout": True,
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "axes.prop_cycle": cycler(color=list(chosen_palette.values())),

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
    """
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
    """
    return df, out_dir, csv_path

def singlePlot(df, cols):
    cols = [c for c in cols if c in df.columns]  # sécurité si une colonne manque

    ax = df[cols].plot(linewidth=3)  # x = index (le temps)
    ax.set_xlabel(df.index.name or "time [s]")
    ax.set_ylabel(r"$\sigma$ [MPa]")
    ax.grid(True, which="both", ls="--")
    ax.legend(loc="upper center", ncol=2, fontsize="small")
    plt.tight_layout()
    plt.show()


# COMPUTATION OF THE RELEVANT VARIABLES
#-------------------------------------------------------------------------------------
def equivalentBackStress(df):
    Axx = df['A_XX']
    Ayy = df['A_YY']
    Azz = df['A_ZZ']
   
    s = (Axx**2 + Ayy**2 + Azz**2)
    return np.sqrt(1.5 * s)

def equivalentStress(df):
    sxx = df['Sigma_XX']
    syy = df['Sigma_YY']
    szz = df['Sigma_ZZ']
    
    s = sxx**2 + syy**2 + szz**2
    return np.sqrt(1.5 * s)

def vonMisesEquivalentstress(df):
    sxx = df['Sigma_XX']
    syy = df['Sigma_YY']
    szz = df['Sigma_ZZ']

    Axx = df['A_XX']
    Ayy = df['A_YY']
    Azz = df['A_ZZ']

    diff_xx = sxx - Axx
    diff_yy = syy - Ayy
    diff_zz = szz - Azz

    s_equiv = diff_xx**2 + diff_yy**2 + diff_zz**2
    return np.sqrt(1.5 * s_equiv)

#-------------------------------------------------------------------------------------


#PLOT FUNCITONS
#-------------------------------------------------------------------------------------
def multipleModelsPlot(index, xlabel, ylabel, sim_folders, labels, variable, f):
 
    plt.figure() 
    for folder, label in zip(sim_folders, labels):
        df, *_ = analyze_simulation(folder)

        #1. PLOT AN METAFOR RESULT (E_XX, SigmaVM, ...)
        if(index == 0):
            cols = [variable] # VARIABLE TO BE PLOTTED
            cols = [c for c in cols if c in df.columns]
            plt.plot(df.index, df[cols].values, label=label, linewidth=3)

        #2. PLOT A VALUE COMPUTED FROM THE METAFOR RESULTS
        if(index == 1):
            plt.plot(df.index, f(df), label=label, linewidth=3)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--")
    plt.legend(loc="upper center", ncol=2, fontsize="small")
    plt.tight_layout()
    plt.show()

def Plot(x, y, xlabel, ylabel):
    plt.plot(y, x, linewidth=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--")
    plt.legend(loc="upper center", ncol=2, fontsize="small")
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------


#USER ONLY NEEDS TO MODIFY THE FOLLOWING LINES
#-----------------------------------------------------------------------------------

def main():

    #1. PLOTTING ONE VARIABLE (as function of time) FOR MULTIPLE SCENARIOS PLACED IN DIFFERENT FOLDERS
    
    sim_folders = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\planestress\CubeperfectPlastic",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\planestress\Cubeiso",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\planestress\CubeKinH",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\planestress\CubeMixH"
    ]
    
    labels = ["Perfectly Plastic", "Isotropic", "Kinematic", "Mixed"]

    index = 1 # 0 or 1 DEPENDING IF YOU PLOT DIRECT VARIABLES (E_XX. SigmaVM,...) 
                        # OR ONE THAT MUST BE COMPUTED (equivalentBackStress,..)

    variable = "Sigma_XX" #ONLY USED WHEN index = 0 !! DO NO TRY TO PLOT MULTIPLE VARIABLES FOR DIFFERENT MODELS -> TOO MESSY
    function = equivalentBackStress  #ONLY USED WHEN index = 1

    xlabel = r"time [$\mathrm{s}$]"
    ylabel = r"$\bar\alpha$ [MPa]"
    multipleModelsPlot(index, xlabel, ylabel, sim_folders, labels, variable, function)
    

    #2. PLOTTING MULTIPLE VARIABLES (as function of time) FOR ONE PARTICULAR SCENARIO

    cols = ["Sigma_XX", "Sigma_YY", "Sigma_ZZ", "SigmaVM"]
    cols = ["E_XX", "E_YY", "E_ZZ"]
    df_iso, *_ = analyze_simulation(r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\planestress\Cubeiso")
    singlePlot(df_iso, cols)
    

    #3. PLOTTING TWO VARIABLES (one vs the other, not as a fct of time) FOR THE SAME SCENARIO

    df_iso, *_ = analyze_simulation(r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\planestress\Cubeiso")
    x = df_iso["E_XX"].values
    y = df_iso["Sigma_XX"].values

    xlabel = r"$\epsilon_{xx}$ [-]"
    ylabel = r"$\sigma_{xx}$ [MPa]"
    Plot(x, y, xlabel, ylabel)
#-----------------------------------------------------------------------------------
if __name__ == "__main__":
    main()


