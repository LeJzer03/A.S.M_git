import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

palette_okabe_ito = {
    "Orange": "#E69F00",
    "Green":  "#009E73",
    "Blue":   "#0072B2",
    "Light Blue": "#8cc5e3",
}

linestyles = ["-", "-", "-", "-"]
alphas = [1, 1, 1, 1]
chosen_palette = palette_okabe_ito

# plt.rcParams.update({
#     "mathtext.fontset": "cm",
#     "figure.autolayout": True,
#     "font.size": 16,
#     "axes.labelsize": 25,
#     "axes.titlesize": 25,
#     "legend.fontsize": 14,
#     "axes.prop_cycle": cycler(color=list(chosen_palette.values()))
#                         + cycler(linestyle=linestyles)
#                         + cycler(alpha=alphas),
# })
plt.rcParams.update({
    #"mathtext.fontset": "cm",
    "figure.autolayout": True,
    "font.size": 15,
    "axes.labelsize": 22,
    "axes.titlesize": 20,
    "legend.fontsize": 16,
    "axes.prop_cycle": cycler(color=list(chosen_palette.values()))
                        + cycler(linestyle=linestyles)
                        + cycler(alpha=alphas),

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

def equivalentStressPlaneStress(df):
    sxx = df['Sigma_XX']
    syy = df['Sigma_YY']
    szz = df['Sigma_ZZ']
    
    s = sxx**2
    return np.sqrt(s)


def sigmaYVisco(df, SigmaY0, hi):
    eVP = df['EPL']
    return SigmaY0 + hi * eVP

def eqBackStressVisco(df, hk):
    eVP = df['EPL']
    return hk*eVP

def VonMisesMinusSigmaY(df, SigmaY0, hi):
    sigmaVM = df['SigmaVM']
    sigmaY = sigmaYVisco(df, SigmaY0, hi)
    return sigmaVM - sigmaY


#-------------------------------------------------------------------------------------


#PLOT FUNCITONS
#-------------------------------------------------------------------------------------
def multipleModelsPlot(index, timeIndex, xlabel, ylabel, sim_folders, labels, variable, f, SigmaY0, hi, hk):
 
    # plt.figure(figsize=(7.5, 5.5)) 
    plt.figure(figsize=(11.5, 6.5)) 
    for folder, label in zip(sim_folders, labels):
        df, *_ = analyze_simulation(folder)

        #1. PLOT AN METAFOR RESULT (E_XX, SigmaVM, ...)
        if(index == 0):
            if variable in df.columns:
                y = df[variable].values.ravel()
                if(timeIndex == 0):
                    plt.plot(df.index, 100*y, label=label, linewidth=3)
                elif(timeIndex == 1):
                    plt.plot(100*y, f(df), label=label, linewidth=3)
            else:
                print(f"Variable {variable} not found in {folder}")

        #2. PLOT A VALUE COMPUTED FROM THE METAFOR RESULTS
        if(index == 1):
            plt.plot(df.index, f(df), label=label, linewidth=3)
            # plt.plot(df.index, f(df, SigmaY0, hi), label=label, linewidth=3) #SigmaYVisco
            # plt.plot(df.index, f(df, hk), label=label, linewidth=2.5) #eqBackStressVisco
            # plt.plot(df.index, f(df, SigmaY0, hi), label=label, linewidth=3) 

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--")
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, -3))
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(-3, -3))
    # forcer des graduations tous les 4 s
    from matplotlib.ticker import MultipleLocator
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.001 *100))
    # optionnel : agrandir la taille du "×10⁻³"
    plt.gca().yaxis.get_offset_text().set_fontsize(14)

    #plt.legend(loc="upper center", ncol=2, fontsize="small")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4)
    # plt.legend()
    # plt.xlim(0, 4)
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------


#USER ONLY NEEDS TO MODIFY THE FOLLOWING LINES
#-----------------------------------------------------------------------------------

def main():

    #-------------------------------------------------------------------------
    #Visco folders
    sim_folder_visco_no_hard_cte_load = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoNoHardening\eta10_2",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoNoHardening\eta10_3",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoNoHardening\eta10_4",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoNoHardening\eta10_5"
    ]

    sim_folder_visco_lin_iso_cte_load = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoLinIsoHard\const_load\eta10_2",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoLinIsoHard\const_load\eta10_3",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoLinIsoHard\const_load\eta10_4",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoLinIsoHard\const_load\eta10_5"
    ]
    sim_folder_visco_mix_cte_load = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoMixKinHard\const_load\eta10_2",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoMixKinHard\const_load\eta10_3",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoMixKinHard\const_load\eta10_4",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoMixKinHard\const_load\eta10_5"
    ]

    sim_folder_visco_lin_iso_triangular_load = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoLinIsoHard\triangular_load\eta10_2",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoLinIsoHard\triangular_load\eta10_3",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoLinIsoHard\triangular_load\eta10_4",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoLinIsoHard\triangular_load\eta10_5",
    ]
    sim_folder_visco_mix_triangular_load = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoMixKinHard\triangular_load\eta10_2",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoMixKinHard\triangular_load\eta10_3",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoMixKinHard\triangular_load\eta10_4",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoMixKinHard\triangular_load\eta10_5",
    ]

    sim_folder_visco_lin_iso_sawtooth_load = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoLinIsoHard\sawtooth_load\eta10_2",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoLinIsoHard\sawtooth_load\eta10_3",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoLinIsoHard\sawtooth_load\eta10_4",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoLinIsoHard\sawtooth_load\eta10_5"
    ]
    sim_folder_visco_mix_sawtooth_load = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoMixKinHard\sawtooth_load\eta10_2",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoMixKinHard\sawtooth_load\eta10_3",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoMixKinHard\sawtooth_load\eta10_4",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\viscoMixKinHard\sawtooth_load\eta10_5"
    ]

    sim_folder_iso_eta10_5 = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\stepVSsawtooth\linearIso\triangular",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\stepVSsawtooth\linearIso\sawtooth"
    ]
    sim_folder_mix_eta10_5 = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\stepVSsawtooth\linearMix\triangular",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\stepVSsawtooth\linearMix\sawtooth"
    ]   
    
    sim_folder_iso_loading_speed = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\loading_speed\isotropic\t_2s",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\loading_speed\isotropic\t_4s",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\loading_speed\isotropic\t_8s"
    ]
    sim_folder_mix_loading_speed = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\loading_speed\mixed\t_2s",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\loading_speed\mixed\t_4s",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part3\loading_speed\mixed\t_8s"
    ]

    #---------------------------------------------------------------------------------------------
    
    labels = ["$\eta = 10^2$ [MPa$\cdot$s]","$\eta = 10^3$ [MPa$\cdot$s]","$\eta = 10^4$ [MPa$\cdot$s]","$\eta = 10^5$ [MPa$\cdot$s]"]
    # labels = ["$T = 2\,[s]$","$T = 4\,[s]$","$T = 8\,[s]$"]
    # labels = ["Sawtooth", "Stepped"]

    index = 0 # 0-> variable || 1-> function
    timeIndex = 1 #0-> plot vs time || 1-> plot vs function
                    # if index = 1 -> always plot vs time

    hardening = "iso"
    SigmaY0 = 300.0 
    
    if(hardening=="iso"):
        hi = 40000.0 
        hk = 0
    elif(hardening=="mix"):
        theta = 0.75
        hi = theta * 40000.0  
        hk = (1 - theta) * 40000.0

    variable = "EPL" 
    function = equivalentStressPlaneStress
    # xlabel = r"time [s]"
    xlabel = r"$\bar \varepsilon^{\mathrm{vp}}\, \,$ [%]"
    # ylabel = r"$\bar \alpha\, \,$ [MPa]"
    # ylabel = r"$\bar \sigma\, \,$ [MPa]"
    # ylabel = r"$\bar \sigma^{VM}\, \,$ [MPa]"
    # ylabel = r"$ \sigma_y\, \,$ [MPa]"
    ylabel = r"$\sigma^{VM} - \sigma_y\, \,$ [MPa]"

    multipleModelsPlot(index, timeIndex, xlabel, ylabel, sim_folder_visco_lin_iso_triangular_load, labels, variable, function, SigmaY0, hi, hk)

#-----------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

