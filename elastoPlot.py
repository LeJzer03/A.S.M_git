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

linestyles = ["-", "-", "-", "--"]
alphas = [1, 1, 1, 1]
chosen_palette = palette_okabe_ito

plt.rcParams.update({
    #"mathtext.fontset": "cm",
    "figure.autolayout": True,
    "font.size": 15,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "legend.fontsize": 20,
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

# COMPUTATION OF THE RELEVANT VARIABLES
#-------------------------------------------------------------------------------------
def equivalentBackStress(df):
    Axx = df['A_XX']
    Ayy = df['A_YY']
    Azz = df['A_ZZ']
   
    s = (Axx**2 + Ayy**2 + Azz**2)
    return np.sqrt(1.5 * s)

def equivalentStressPlaneStrain(df):
    sxx = df['Sigma_XX']
    syy = df['Sigma_YY']
    szz = df['Sigma_ZZ']
    
    s = sxx**2 - sxx*szz + szz**2
    return np.sqrt(1.5 * s)

def equivalentStressPlaneStress(df):
    sxx = df['Sigma_XX']
    syy = df['Sigma_YY']
    szz = df['Sigma_ZZ']
    
    s = sxx**2
    return np.sqrt(s)

def vonMisesEquivalentPlaneStress(df):
    sxx = df['Sigma_XX']
    syy = df['Sigma_YY']
    szz = df['Sigma_ZZ']

    Axx = df['A_XX']
    Ayy = df['A_YY']
    Azz = df['A_ZZ']

    bxx = sxx - Axx
    byy = syy - Ayy
    bzz = szz - Azz

    vm2 = 0.5 * ((bxx - byy)**2 + (byy - bzz)**2 + (bzz - bxx)**2)
    return np.sqrt(vm2)

# def eqVM_PlaneStress(df): #identical results to the formula above
#     sigmaXX = df['Sigma_XX']
#     sXX = 2/3 * sigmaXX
#     alphaXX = df['A_XX']
#     vm_3 = 1.5 * (sXX - alphaXX)**2

#     return np.sqrt(vm_3)

def plasticDissipationRate(df):

    sigma_vm = df['SigmaVM']
    epl = df['EPL']
    
    time = df.index.values
    dt = np.diff(time)
    
    epl_rate = np.zeros_like(epl)
    epl_rate[:-1] = np.diff(epl) / dt
    epl_rate[-1] = epl_rate[-2]  
     
    dissipation_rate = sigma_vm * epl_rate
    return dissipation_rate

#-------------------------------------------------------------------------------------


#PLOT FUNCITONS
#-------------------------------------------------------------------------------------
def singlePlot(df, cols, labels=None, ncol_legend=4):
    """
    Plot selected columns vs time from dataframe `df` with a visual style
    similar to the provided figure.
    Usage:
        cols = ["Sigma_XX","Sigma_YY","Sigma_ZZ","SigmaVM"]
        singlePlot(df_perfPlastic, cols, labels=["Sigma_XX","Sigma_YY","Sigma_ZZ","SigmaVM"])
    """
    from matplotlib.ticker import MultipleLocator

    # keep only existing columns
    cols = [c for c in cols if c in df.columns]
    if not cols:
        print("Aucune colonne valide à tracer :", cols)
        return

    # prepare labels
    if labels is None:
        plot_labels = cols
    else:
        if len(labels) == len(cols):
            plot_labels = labels
        elif len(labels) == 1:
            plot_labels = labels * len(cols)
        else:
            print("Warning: labels length != cols length, using column names.")
            plot_labels = cols

    fig, ax = plt.subplots()

    # plot each series; use dashed line for SigmaVM to match example
    for col, lab in zip(cols, plot_labels):
        ax.plot(df.index, df[col].values, linewidth=3, label=lab)

    # axis labels and formatting
    #xlabel = r'$\mathrm{time} \, \, \mathrm{[s]}$'
    xlabel = xlabel = r"time [s]"
    ylabel = r"$\sigma$ [MPa]"
    #ylabel = r"$\sigma \text{[MPa]}$"



    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls="--", linewidth=0.7, color="0.7")
    # force x start at 0 (matches your figure)
    try:
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(left=0, right=xmax)
    except Exception:
        pass

    # nicer x ticks (optional)
    from matplotlib.ticker import MultipleLocator
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    # legend centered at top (inside figure) like the example
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol_legend, fontsize=12)
    leg.get_frame().set_alpha(1.0)

    fig.tight_layout()
    plt.show()



def multipleModelsPlot(index, xlabel, ylabel, sim_folders, labels, variable, f):
 
    plt.figure() 
    for folder, label in zip(sim_folders, labels):
        df, *_ = analyze_simulation(folder)

        #1. PLOT AN METAFOR RESULT (E_XX, SigmaVM, ...)
        if(index == 0):
            if variable in df.columns:
                y = df[variable].values.ravel()
                plt.plot(df.index, y, label=label, linewidth=4)
            else:
                print(f"Variable {variable} not found in {folder}")

        #2. PLOT A VALUE COMPUTED FROM THE METAFOR RESULTS
        if(index == 1):
            plt.plot(df.index, f(df), label=label, linewidth=4)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--")
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, -3))
    # forcer des graduations tous les 4 s
    from matplotlib.ticker import MultipleLocator
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    # optionnel : agrandir la taille du "×10⁻³"
    plt.gca().yaxis.get_offset_text().set_fontsize(14)

   #plt.legend(loc="upper center", ncol=2, fontsize="small")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=12)
    # plt.legend(fontsize=16)
    plt.tight_layout()
    plt.xlim(0, 4)
    plt.show()


def multipleModelsMultiplesTimes(index, xlabel, ylabel, sim_folders1, labels, variable, f):

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)

    # plot the same set of folders on both subplots:
    for ax in axes:
        for folder, label in zip(sim_folders1, labels):
            df, *_ = analyze_simulation(folder)
            if index == 0:
                if variable in df.columns:
                    ax.plot(df.index, df[variable], label=label, linewidth=3)
                else:
                    print(f"Variable {variable} not found in {folder}")
            elif index == 1:
                ax.plot(df.index, f(df), label=label, linewidth=3)

        ax.set_xlabel(xlabel)
        ax.grid(True, which="both", ls="--")

    # left: first 4 seconds, right: full time
    from matplotlib.ticker import MultipleLocator
    axes[0].set_xlim(0, 4)
    axes[0].set_ylabel(ylabel)
    axes[0].xaxis.set_major_locator(MultipleLocator(1))
    axes[1].xaxis.set_major_locator(MultipleLocator(4))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.ylim(-0.001,0.04)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize=14)
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
    
    #LOADING THE  PLANE STRAIN RESULTS:
    # #---------------------------------------------------------------------------------------------
    sim_folder_plane_stress = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part1\planestress\isotropic",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part1\planestress\kinematic",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part1\planestrain\mixed",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part1\planestress\noHardening"
    ]
    sim_folder_plane_strain = [
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part1\planestrain\isotropic",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part1\planestrain\kinematic",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part1\planestrain\mixed",
        r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part1\planestrain\noHardening"
    ]
    # #---------------------------------------------------------------------------------------------
    
    labels = ["Isotropic", "Kinematic", "Mixed", "Perfectly Plastic"]

    index = 0 # 0-> variable || 1-> function

    variable = "Sigma_Yield"
    function = plasticDissipationRate

    xlabel = r"time [s]"
    # ylabel = r"$\bar \varepsilon^{\mathrm{p}}\, \,$ [-]"
    # ylabel = r"$\bar \varepsilon^{\mathrm{p}}\, \,[$-$]$"
    # ylabel = r"$\bar \alpha\, \,$ [MPa]"
    # ylabel = r"$\bar \sigma\, \,$ [MPa]"
    # ylabel = r"$\bar \sigma^{VM}\, \,$ [MPa]"
    ylabel = r"$\bar \sigma_y\, \,$ [MPa]"
    # ylabel = r"$\mathbb{D}\, \,$ [W/$\text{m}^3$]"
    # ylabel = r"$\mathbb{D} \, \,[\mathrm{W\,/m^3}]$"
    
    # multipleModelsPlot(index, xlabel, ylabel, sim_folder_plane_strain,  labels, variable, function)
    multipleModelsMultiplesTimes(index, xlabel, ylabel, sim_folder_plane_strain,  labels, variable, function)
    

    #2. PLOTTING MULTIPLE VARIABLES (as function of time) FOR ONE PARTICULAR SCENARIO

    # cols = ["Sigma_XX", "Sigma_YY", "Sigma_ZZ", "SigmaVM"]
    # labels = [r"$\sigma_{xx}\, \,$ [MPa]",r"$\sigma_{yy}\, \,$[MPa]",r"$\sigma_{zz}\, \,$[MPa]", r"$\sigma^{VM}\, \,$[MPa]"]
    # df_perfPlastic, *_ = analyze_simulation(r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\part1\planestress\noHardening")
    # singlePlot(df_perfPlastic, cols, labels)
    

    #3. PLOTTING TWO VARIABLES (one vs the other, not as a fct of time) FOR THE SAME SCENARIO

    # df_iso, *_ = analyze_simulation(r"C:\Users\vinch\OneDrive - Universite de Liege\Documents\master1\q1\asm\project\workspace\planestress\IH")
    # x = df_iso["E_XX"].values
    # y = df_iso["Sigma_XX"].values

    # xlabel = r"$\epsilon_{xx}$ [-]"
    # ylabel = r"$\sigma_{xx}$ [MPa]"
    # Plot(x, y, xlabel, ylabel)
#-----------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
