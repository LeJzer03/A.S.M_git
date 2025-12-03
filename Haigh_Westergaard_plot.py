import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from matplotlib.patches import FancyArrowPatch
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

# ====================================================================
# 1. DATA LOADING 
# ====================================================================

def read_ascii(path: Path) -> np.ndarray:
    """Minimal helper: read space-separated ASCII file into a numpy array."""
    return np.loadtxt(path, dtype=float)


def load_master_time(folder_path: Path) -> np.ndarray:
    """Load the master time vector from time.ascii in the folder."""
    time_file = folder_path / "time.ascii"
    if not time_file.exists():
        raise FileNotFoundError(f"Master time file not found: {time_file}")
    data = read_ascii(time_file)
    if data.ndim == 1:
        return data.astype(float)
    return data[:, 0].astype(float)


def analyze_simulation(folder: str):
    """Your original data loading function."""
    folder_path = Path(folder).resolve()
    if not (folder_path.exists() and folder_path.is_dir()):
        raise NotADirectoryError(f"Folder not found: {folder_path}")

    # 1) master time
    t_master = load_master_time(folder_path)
    nT = len(t_master)

    # 2) collect series
    ascii_files = sorted(folder_path.glob("*.ascii"))
    series = {}

    for f in ascii_files:
        if f.name.lower() == "time.ascii":
            continue

        name = f.stem
        data = read_ascii(f)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if data.shape[1] >= 2:
            t_local = data[:, 0].astype(float)
            y_local = data[:, 1].astype(float)
            order = np.argsort(t_local)
            t_sorted = t_local[order]
            y_sorted = y_local[order]
            y_interp = np.interp(t_master, t_sorted, y_sorted,
                                 left=np.nan, right=np.nan)
            series[name] = y_interp
        else:
            y_local = data[:, 0].astype(float)
            y = np.full(nT, np.nan, dtype=float)
            n = min(nT, len(y_local))
            y[:n] = y_local[:n]
            series[name] = y

    if not series:
        raise FileNotFoundError(
            f"No variable *.ascii files found in {folder_path} besides time.ascii"
        )

    # 3) DataFrame indexed by time
    df = pd.DataFrame(series, index=t_master)
    df.index.name = "time [s]"

    return df


def get_row_at_time(df: pd.DataFrame, t_target: float):
    """Return the row closest to t_target and the actual time used."""
    times = df.index.to_numpy(dtype=float)
    idx = np.argmin(np.abs(times - t_target))
    t_used = times[idx]
    row = df.iloc[idx]
    return row, t_used


# ====================================================================
# 2. PROJECTION FUNCTIONS 
# ====================================================================

def project_to_deviatoric(s_xx, s_yy, s_zz):
    """Project stress components to deviatoric plane"""
    X = (1.0/np.sqrt(2.0)) * (s_zz - s_yy)
    Y = (1.0/np.sqrt(6.0)) * (2.0*s_xx - s_yy - s_zz)
    return X, Y


# ====================================================================
# 3. YOUR COLOR PALETTE
# ====================================================================

palette_1 = {
    "Med Blue": "#3594cc",
    "Med Orange": "#ea801c",
    "Light Blue": "#8cc5e3",
    "Light Orange": "#f0b077"
}

# Map colors to hardening types
COLORS_MAIN = {
    'Isotropic': palette_1["Med Orange"],
    'Kinematic': palette_1["Med Blue"],
    'Mixed': palette_1["Light Blue"],
    'PerfectlyPlastic': '#264653'
}

COLORS_LIGHT = {
    'Isotropic': palette_1["Light Orange"],
    'Kinematic': palette_1["Light Blue"],
    'Mixed': palette_1["Light Orange"],
    'PerfectlyPlastic': '#5E7A84'
}


# ====================================================================
# 4. IMPROVED PLOTTING FUNCTIONS
# ====================================================================

def plot_single_comparison(
    folder_list: list[str],
    labels: list[str],
    t_target: float,
    axis_limit: float = 250.0,
    show_trajectory: bool = True,
):
    """
    Create a single, clean comparison plot at one time instant.
    Shows yield surface evolution with optional stress trajectory.
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    # Modern style
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')

    # Load data
    database = {}
    for folder, label in zip(folder_list, labels):
        df = analyze_simulation(folder)
        database[label] = df

    # --- Draw principal stress axes as arrows ---
    angles = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
    axis_labels = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\sigma_{zz}$']

    for ang, lab in zip(angles, axis_labels):
        end_x = axis_limit * 0.8 * np.cos(ang)
        end_y = axis_limit * 0.8 * np.sin(ang)

        arrow = FancyArrowPatch(
            (0, 0), (end_x, end_y),
            arrowstyle='->', mutation_scale=25,
            linewidth=2, color="#000000", alpha=1.0,
        )
        ax.add_patch(arrow)

        label_x = axis_limit * 0.85 * np.cos(ang)
        label_y = axis_limit * 0.85 * np.sin(ang)
        ax.text(label_x, label_y, lab, ha='center', va='center',
                fontsize=18, fontweight='bold', color="#181818")

    # --- Reference circles = constant σ_VM ---
    for stress_val in [100, 200, 300, 400, 500]:
        if stress_val > axis_limit * 1.5:
            continue
        r_iso = np.sqrt(2.0/3.0) * stress_val
        circle = patches.Circle(
            (0, 0), r_iso,
            color='#CCCCCC', fill=False,
            linestyle='--', linewidth=1,
            alpha=0.4, zorder=1
        )
        ax.add_patch(circle)

        # Label for each circle
        label_ang = np.pi/6
        ax.text(r_iso * np.cos(label_ang), r_iso * np.sin(label_ang),
                f'{stress_val} MPa', fontsize=9, color='#888888',
                ha='left', va='bottom', style='italic')

    # --- Plot each hardening model ---
    for label in labels:
        df = database[label]
        row, t_used = get_row_at_time(df, t_target)

        color = COLORS_MAIN.get(label, '#666666')

        # Backstress
        alp_xx = float(row.get('A_XX', 0.0))
        alp_yy = float(row.get('A_YY', 0.0))
        alp_zz = float(row.get('A_ZZ', 0.0))

        # Yield stress
        if 'Kinematic' in label or 'Mixed' in label:
            current_yield = float(df['Sigma_Yield'].iloc[0])
        else:
            current_yield = float(row.get('Sigma_Yield', 100.0))

        R_current = np.sqrt(2.0/3.0) * current_yield

        # Center of yield surface
        Xc, Yc = project_to_deviatoric(alp_xx, alp_yy, alp_zz)

        # Current stress state
        sig_xx = float(row.get('Sigma_XX', 0.0))
        sig_yy = float(row.get('Sigma_YY', 0.0))
        sig_zz = float(row.get('Sigma_ZZ', 0.0))
        X_stress, Y_stress = project_to_deviatoric(sig_xx, sig_yy, sig_zz)

        # Yield surface circle (no fill)
        circ = patches.Circle(
            (Xc, Yc), R_current,
            linewidth=3, edgecolor=color,
            facecolor='none', linestyle='-',
            label=label, zorder=3
        )
        ax.add_patch(circ)

        # Center marker for kinematic / mixed
        if abs(Xc) > 1 or abs(Yc) > 1:
            ax.plot(Xc, Yc, 'x', color=color,
                    markersize=14, markeredgewidth=2.5, zorder=5)

            # Arrow from origin to center
            if np.hypot(Xc, Yc) > 10:
                arrow_center = FancyArrowPatch(
                    (0, 0), (Xc, Yc),
                    arrowstyle='->', mutation_scale=15,
                    linewidth=2.0, color=color,
                    alpha=1.0, linestyle=':', zorder=2
                )
                ax.add_patch(arrow_center)

        # Current stress point
        ax.plot(X_stress, Y_stress, 'o', color=color,
                markersize=12, markeredgewidth=2,
                markeredgecolor='white', zorder=6)

        # Optional trajectory of stress point
        # if show_trajectory:
        #     times = df.index.to_numpy()
        #     mask = times <= t_target

        #     X_traj, Y_traj = [],
        #     # subsample to at most ~50 points
        #     step = max(1, len(times[mask]) // 50)
        #     for t in times[mask][::step]:
        #         r, _ = get_row_at_time(df, t)
        #         sxx = float(r.get('Sigma_XX', 0.0))
        #         syy = float(r.get('Sigma_YY', 0.0))
        #         szz = float(r.get('Sigma_ZZ', 0.0))
        #         x, y = project_to_deviatoric(sxx, syy, szz)
        #         X_traj.append(x)
        #         Y_traj.append(y)

        #     if len(X_traj) > 1:
        #         ax.plot(X_traj, Y_traj, '-', color=color,
        #                 linewidth=1.5, alpha=0.4, zorder=2)

    # --- Formatting ---
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect('equal')

    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8, color='#CCCCCC')

    # Deviatoric-coordinate labels
    ax.set_xlabel(
        r'X – Deviatoric Stress Coordinate [MPa]',
        fontsize=14
    )
    ax.set_ylabel(
        r'Y – Deviatoric Stress Coordinate [MPa]',
        fontsize=14
    )


    # --- Legend construction ---
    handles, legend_labels = ax.get_legend_handles_labels()

    # Replace patch handles (circles) with line handles
    new_handles = []
    new_labels = []
    for h, lab in zip(handles, legend_labels):
        if isinstance(h, patches.Patch):
            line = Line2D([0], [0],
                          color=h.get_edgecolor(),
                          linewidth=3)
            new_handles.append(line)
            new_labels.append(lab)
        else:
            new_handles.append(h)
            new_labels.append(lab)

    handles = new_handles
    legend_labels = new_labels

    # Generic symbols for center and current stress
    cross_handle = Line2D(
        [0], [0], marker='x', color='black',
        linestyle='None', markersize=10,
        markeredgewidth=2, label='Yield Surface Center'
    )
    circle_handle = Line2D(
        [0], [0], marker='o', color='black',
        linestyle='None', markersize=8,
        markeredgewidth=2, markerfacecolor='white',
        label='Current Stress State'
    )

    # Legend entries explaining projection / circles
    projected_handle = Line2D(
        [0], [0],
        linestyle='--', color='gray', linewidth=1.5,
        label=r'Eq. stress circles: $R=\sqrt{2/3}\,\sigma_{\mathrm{VM}}$'
    )
    axes_handle = Line2D(
        [0], [0],
        linestyle='None', marker='.', color='white',
        label=r'$X,Y$: projected deviatoric coordinates'
    )

    all_handles = handles + [
        cross_handle, circle_handle,
        projected_handle, axes_handle
    ]
    all_labels = legend_labels + [
        'Yield Surface Center',
        'Current Stress State',
        r'Eq. stress circles: $R=\sqrt{2/3}\,\sigma_{\mathrm{VM}}$',
    ]

    ax.legend(
        all_handles, all_labels,
        loc='upper right', fontsize=9,
        framealpha=0.85, edgecolor='#CCCCCC',
        fancybox=False, shadow=False,
        handlelength=1.2, handletextpad=0.5, markerscale=0.8
    )

    # Make axes look neat
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#CCCCCC')

    ax.tick_params(labelsize=11, width=1.5, color='#CCCCCC')

    plt.tight_layout()
    return fig, ax




def plot_time_evolution_grid(
    folder_list: list[str],
    labels: list[str],
    times_to_plot: list[float],
    axis_limit: float = 250.0,
):
    """
    Create a grid showing evolution at multiple time steps.
    Each subplot shows all models at one time instant.
    Last subplot (bottom-right) is reserved for the legend.
    """
    
    n_times = len(times_to_plot)
    
    # Always create a 3x2 grid (6 subplots total: 5 for plots, 1 for legend)
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    axes = axes.flatten()
    
    fig.patch.set_facecolor('white')
    
    # Load all data once
    database = {}
    for folder, label in zip(folder_list, labels):
        df = analyze_simulation(folder)
        database[label] = df
    
    for idx, t_target in enumerate(times_to_plot):
        if idx >= 5:  # Only use first 5 subplots
            break
            
        ax = axes[idx]
        ax.set_facecolor('#FAFAFA')
        
        # --- Draw principal stress axes as arrows ---
        angles = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
        axis_labels = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\sigma_{zz}$']
        
        for ang, lab in zip(angles, axis_labels):
            end_x = axis_limit * 0.8 * np.cos(ang)
            end_y = axis_limit * 0.8 * np.sin(ang)
            
            arrow = FancyArrowPatch(
                (0, 0), (end_x, end_y),
                arrowstyle='->', mutation_scale=20,
                linewidth=1.5, color="#000000", alpha=1.0,
            )
            ax.add_patch(arrow)
            
            label_x = axis_limit * 0.85 * np.cos(ang)
            label_y = axis_limit * 0.85 * np.sin(ang)
            ax.text(label_x, label_y, lab, ha='center', va='center',
                   fontsize=14, fontweight='bold', color="#181818")
        
        # --- Reference circles ---
        for stress_val in [100, 200, 300, 400, 500]:
            if stress_val > axis_limit * 1.5:
                continue
            r_iso = np.sqrt(2.0/3.0) * stress_val
            circle = patches.Circle(
                (0, 0), r_iso,
                color='#CCCCCC', fill=False,
                linestyle='--', linewidth=0.8,
                alpha=0.4, zorder=1
            )
            ax.add_patch(circle)
            
            # Label
            label_ang = np.pi/6
            ax.text(r_iso * np.cos(label_ang), r_iso * np.sin(label_ang),
                   f'{stress_val} MPa', fontsize=7, color='#888888',
                   ha='left', va='bottom', style='italic')
        
        # Plot models
        for label in labels:
            df = database[label]
            row, t_used = get_row_at_time(df, t_target)
            
            color = COLORS_MAIN.get(label, '#666666')
            
            # Backstress
            alp_xx = float(row.get('A_XX', 0.0))
            alp_yy = float(row.get('A_YY', 0.0))
            alp_zz = float(row.get('A_ZZ', 0.0))
            
            # Yield stress
            if 'Kinematic' in label or 'Mixed' in label:
                current_yield = float(df['Sigma_Yield'].iloc[0])
            else:
                current_yield = float(row.get('Sigma_Yield', 100.0))
            
            R_current = np.sqrt(2.0/3.0) * current_yield
            Xc, Yc = project_to_deviatoric(alp_xx, alp_yy, alp_zz)
            
            # Current stress
            sig_xx = float(row.get('Sigma_XX', 0.0))
            sig_yy = float(row.get('Sigma_YY', 0.0))
            sig_zz = float(row.get('Sigma_ZZ', 0.0))
            X_stress, Y_stress = project_to_deviatoric(sig_xx, sig_yy, sig_zz)
            
            # Yield surface (NO FILL)
            circle = patches.Circle(
                (Xc, Yc), R_current,
                linewidth=2.5, edgecolor=color,
                facecolor='none',
                label=label if idx == 0 else "",
                zorder=3
            )
            ax.add_patch(circle)
            
            # Center marker
            if abs(Xc) > 1 or abs(Yc) > 1:
                ax.plot(Xc, Yc, 'x', color=color, markersize=12,
                       markeredgewidth=2, zorder=5)
                
                # Arrow from origin to center
                if np.hypot(Xc, Yc) > 10:
                    arrow_center = FancyArrowPatch(
                        (0, 0), (Xc, Yc),
                        arrowstyle='->', mutation_scale=12,
                        linewidth=1.5, color=color,
                        alpha=1.0, linestyle=':', zorder=2
                    )
                    ax.add_patch(arrow_center)
            
            # Current stress point
            ax.plot(X_stress, Y_stress, 'o', color=color,
                   markersize=10, markeredgewidth=1.5,
                   markeredgecolor='white', zorder=6)
        
        # Formatting
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8, color='#CCCCCC')
        
        # Title showing time
        ax.set_title(f't = {t_target:.1f} s', fontsize=16, fontweight='bold', pad=10)
        
        # Axis labels
        ax.set_xlabel(r'X – Deviatoric Stress Coordinate [MPa]', fontsize=15)
        ax.set_ylabel(r'Y – Deviatoric Stress Coordinate [MPa]', fontsize=15)
        
        # Clean spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#CCCCCC')
        
        ax.tick_params(labelsize=9, width=1.2, color='#CCCCCC')
    
    # --- LEGEND SUBPLOT (bottom-right corner, index 5) ---
    legend_ax = axes[5]
    legend_ax.axis('off')  # Turn off axis
    
    # Get handles and labels from first subplot
    handles_original, legend_labels = axes[0].get_legend_handles_labels()
    
    # Replace patch handles (circles) with line handles
    handles = []
    for h, lab in zip(handles_original, legend_labels):
        if isinstance(h, patches.Patch):
            line = Line2D([0], [0],
                         color=h.get_edgecolor(),
                         linewidth=3)
            handles.append(line)
        else:
            handles.append(h)
    
    # Add generic symbols
    cross_handle = Line2D(
        [0], [0], marker='x', color='black',
        linestyle='None', markersize=12,
        markeredgewidth=2.5, label='Yield Surface Center'
    )
    circle_handle = Line2D(
        [0], [0], marker='o', color='black',
        linestyle='None', markersize=10,
        markeredgewidth=2, markerfacecolor='white',
        label='Current Stress State'
    )
    
    # Legend entries for annotations
    projected_handle = Line2D(
        [0], [0],
        linestyle='--', color='gray', linewidth=1.5,
        label=r'Eq. stress circles: $R=\sqrt{2/3}\,\sigma_{\mathrm{VM}}$'
    )
    
    all_handles = handles + [
        cross_handle, circle_handle, projected_handle
    ]
    all_labels = legend_labels + [
        'Yield Surface Center',
        'Current Stress State',
        r'Eq. stress circles: $R=\sqrt{2/3}\,\sigma_{\mathrm{VM}}$',
    ]
    
    # Create legend in the dedicated subplot
    legend = legend_ax.legend(
        all_handles, all_labels,
        loc='center',
        fontsize=14,
        framealpha=0.95,
        edgecolor='#CCCCCC',
        fancybox=False,
        shadow=False,
        handlelength=2.0,
        handletextpad=0.8,
        labelspacing=1.2,
        borderpad=1.5
    )
    
    # Add a title to the legend subplot
    legend_ax.text(0.5, 0.95, 'Legend', ha='center', va='top',
                  fontsize=18, fontweight='bold',
                  transform=legend_ax.transAxes)
    
    plt.tight_layout()
    return fig, axes


def plot_single_simulation_evolution(
    folder: str,
    times_to_plot: list[float],
    axis_limit: float = 250.0,
    title: str = "Yield Surface Evolution"
):
    """
    Create a single row with 3 time plots + 1 legend subplot.
    No overall title, compact legend.
    """
    
    if len(times_to_plot) != 3:
        raise ValueError("This function is designed for exactly 3 time steps")
    
    # Create 1 row x 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    fig.patch.set_facecolor('white')
    
    # Load data
    df = analyze_simulation(folder)
    
    # DIAGNOSTIC: Check what data we have
    print(f"\n=== DIAGNOSTIC INFO ===")
    print(f"Columns in dataframe: {df.columns.tolist()}")
    if 'A_XX' in df.columns:
        print(f"A_XX range: [{df['A_XX'].min():.2f}, {df['A_XX'].max():.2f}]")
        print(f"A_XX at t=0: {df['A_XX'].iloc[0]:.2f}")
        print(f"A_XX at t=end: {df['A_XX'].iloc[-1]:.2f}")
    if 'Sigma_Yield' in df.columns:
        print(f"Sigma_Yield range: [{df['Sigma_Yield'].min():.2f}, {df['Sigma_Yield'].max():.2f}]")
        print(f"Sigma_Yield at t=0: {df['Sigma_Yield'].iloc[0]:.2f}")
        print(f"Sigma_Yield at t=end: {df['Sigma_Yield'].iloc[-1]:.2f}")
    
    # Determine hardening type based on backstress evolution
    has_backstress = False
    if 'A_XX' in df.columns:
        alpha_xx_values = df['A_XX'].values
        max_abs_alpha = np.max(np.abs(alpha_xx_values))
        has_backstress = max_abs_alpha > 1.0
        print(f"Max |A_XX|: {max_abs_alpha:.2f}")
        print(f"Has backstress detected: {has_backstress}")
    
    # For kinematic/mixed: use initial yield stress (constant radius)
    if has_backstress:
        initial_yield = float(df['Sigma_Yield'].iloc[0])
        print(f"Using CONSTANT yield stress (kinematic/mixed): {initial_yield:.2f} MPa")
    else:
        print(f"Using EVOLVING yield stress (isotropic)")
    
    print(f"======================\n")
    
    # Color scheme for time progression
    colors_time = plt.cm.viridis(np.linspace(0.2, 0.9, 3))
    
    # Plot first 3 subplots
    for idx, t_target in enumerate(times_to_plot):
        ax = axes[idx]
        ax.set_facecolor('#FAFAFA')
        
        # --- Draw principal stress axes ---
        angles = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
        axis_labels = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\sigma_{zz}$']
        
        for ang, lab in zip(angles, axis_labels):
            end_x = axis_limit * 0.8 * np.cos(ang)
            end_y = axis_limit * 0.8 * np.sin(ang)
            
            arrow = FancyArrowPatch(
                (0, 0), (end_x, end_y),
                arrowstyle='->', mutation_scale=20,
                linewidth=1.5, color="#000000", alpha=1.0,
            )
            ax.add_patch(arrow)
            
            label_x = axis_limit * 0.85 * np.cos(ang)
            label_y = axis_limit * 0.85 * np.sin(ang)
            ax.text(label_x, label_y, lab, ha='center', va='center',
                   fontsize=14, fontweight='bold', color="#181818")
        
        # --- Reference circles with labels ---
        for stress_val in [100, 200, 300, 400, 500]:
            if stress_val > axis_limit * 1.5:
                continue
            r_iso = np.sqrt(2.0/3.0) * stress_val
            circle = patches.Circle(
                (0, 0), r_iso,
                color='#CCCCCC', fill=False,
                linestyle='--', linewidth=0.8,
                alpha=0.4, zorder=1
            )
            ax.add_patch(circle)
            
            # Add stress value label to circles
            label_ang = np.pi/6
            ax.text(r_iso * np.cos(label_ang), r_iso * np.sin(label_ang),
                   f'{stress_val} MPa', fontsize=7, color='#888888',
                   ha='left', va='bottom', style='italic')
        
        # Get data at this time
        row, t_used = get_row_at_time(df, t_target)
        color = colors_time[idx]
        
        # Backstress
        alp_xx = float(row.get('A_XX', 0.0))
        alp_yy = float(row.get('A_YY', 0.0))
        alp_zz = float(row.get('A_ZZ', 0.0))
        
        # Yield stress - CORRECTED LOGIC
        if has_backstress:
            # Kinematic/Mixed: constant radius, moving center
            current_yield = initial_yield
        else:
            # Isotropic: growing radius, fixed center
            current_yield = float(row.get('Sigma_Yield', 100.0))
        
        # DIAGNOSTIC for this timestep
        print(f"Time {t_used:.2f}s: α_xx={alp_xx:.2f}, σ_y={current_yield:.2f}, R={np.sqrt(2.0/3.0)*current_yield:.2f}")
        
        R_current = np.sqrt(2.0/3.0) * current_yield
        Xc, Yc = project_to_deviatoric(alp_xx, alp_yy, alp_zz)
        
        print(f"  Center: ({Xc:.2f}, {Yc:.2f})")
        
        # Current stress
        sig_xx = float(row.get('Sigma_XX', 0.0))
        sig_yy = float(row.get('Sigma_YY', 0.0))
        sig_zz = float(row.get('Sigma_ZZ', 0.0))
        X_stress, Y_stress = project_to_deviatoric(sig_xx, sig_yy, sig_zz)
        
        print(f"  Stress point: ({X_stress:.2f}, {Y_stress:.2f})")
        
        # Yield surface (NO FILL)
        circle = patches.Circle(
            (Xc, Yc), R_current,
            linewidth=2.5, edgecolor=color,
            facecolor='none',
            zorder=3
        )
        ax.add_patch(circle)
        
        # Center marker
        if abs(Xc) > 1 or abs(Yc) > 1:
            ax.plot(Xc, Yc, 'x', color=color, markersize=12,
                   markeredgewidth=2, zorder=5)
            
            # Arrow from origin to center
            if np.hypot(Xc, Yc) > 10:
                arrow_center = FancyArrowPatch(
                    (0, 0), (Xc, Yc),
                    arrowstyle='->', mutation_scale=12,
                    linewidth=1.5, color=color,
                    alpha=1.0, linestyle=':', zorder=2
                )
                ax.add_patch(arrow_center)
        
        # Current stress point
        ax.plot(X_stress, Y_stress, 'o', color=color,
               markersize=10, markeredgewidth=1.5,
               markeredgecolor='white', zorder=6)
        
        # Formatting
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8, color='#CCCCCC')
        
        # Title showing time
        ax.set_title(f't = {t_used:.2f} s', fontsize=16, fontweight='bold', pad=10)
        
        # Axis labels
        ax.set_xlabel(r'X – Deviatoric Stress Coordinate [MPa]', fontsize=15)
        ax.set_ylabel(r'Y – Deviatoric Stress Coordinate [MPa]', fontsize=15)
        
        # Clean spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#CCCCCC')
        
        ax.tick_params(labelsize=9, width=1.2, color='#CCCCCC')
    
    # --- LEGEND SUBPLOT (4th subplot) ---
    legend_ax = axes[3]
    legend_ax.axis('off')
    
    # Create time-based legend handles
    time_handles = []
    time_labels = []
    for idx, t_target in enumerate(times_to_plot):
        row, t_used = get_row_at_time(df, t_target)
        color = colors_time[idx]
        line = Line2D([0], [0], color=color, linewidth=3)
        time_handles.append(line)
        time_labels.append(f't = {t_used:.2f} [s]')
    
    # Add generic symbols
    cross_handle = Line2D(
        [0], [0], marker='x', color='black',
        linestyle='None', markersize=12,
        markeredgewidth=2.5, label='Yield Surface Center'
    )
    circle_handle = Line2D(
        [0], [0], marker='o', color='black',
        linestyle='None', markersize=10,
        markeredgewidth=2, markerfacecolor='white',
        label='Current Stress State'
    )
    
    # Legend entry for reference circles
    projected_handle = Line2D(
        [0], [0],
        linestyle='--', color='gray', linewidth=2,
        label=r'Eq. stress circles'
    )
    
    all_handles = time_handles + [
        cross_handle, circle_handle, projected_handle
    ]
    all_labels = time_labels + [
        'Yield Surface Center',
        'Current Stress State',
        r'Eq. stress circles',
    ]
    
    # Create compact legend with larger text
    legend = legend_ax.legend(
        all_handles, all_labels,
        loc='center left',
        fontsize=24,
        framealpha=0.95,
        edgecolor='#CCCCCC',
        fancybox=False,
        shadow=False,
        handlelength=1.5,
        handletextpad=0.5,
        labelspacing=0.6,
        borderpad=0.8,
        bbox_to_anchor=(0.1, 0.5)
    )
    
    plt.tight_layout()
    return fig, axes


# ====================================================================
# 5. EXAMPLE USAGE
# ====================================================================

if __name__ == "__main__":
    # sim_folder_stress = [
    #     r"C:\Ecole\MASTER - 1\Advanced Solid Mecanics\Project\project_file\Results\Plain Stress (1 normal cycle)\Linear_Isotropic_hardening",
    #     r"C:\Ecole\MASTER - 1\Advanced Solid Mecanics\Project\project_file\Results\Plain Stress (1 normal cycle)\Linear_Kinematic_hardening",
    #     r"C:\Ecole\MASTER - 1\Advanced Solid Mecanics\Project\project_file\Results\Plain Stress (1 normal cycle)\Linear_Mixed_hardening",
    # ]

    # sim_folder_strain = [
    #     r"C:\Ecole\MASTER - 1\Advanced Solid Mecanics\Project\project_file\Results\Plain Strain (1 normal cycle)\Linear_Isotropic_hardening",
    #     r"C:\Ecole\MASTER - 1\Advanced Solid Mecanics\Project\project_file\Results\Plain Strain (1 normal cycle)\Linear_Kinematic_hardening",
    #     r"C:\Ecole\MASTER - 1\Advanced Solid Mecanics\Project\project_file\Results\Plain Strain (1 normal cycle)\Linear_Mixed_hardening"
    # ]
    
    # labels = ["Isotropic", "Kinematic", "Mixed"]
    
    # # Output directory
    output_dir = Path(r"C:\Ecole\MASTER - 1\Advanced Solid Mecanics\Project\Illustration Report\HW plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # # Time steps to plot
    # times = [0.0, 1.0, 2.0, 3.0, 4.0]

    # # Plot for PLANE STRESS
    # print("Creating plots for Plane Stress...")
    # for t in times:
    #     print(f"  Time: {t:.1f} s")
        
    #     fig1, ax1 = plot_single_comparison(
    #         sim_folder_stress,
    #         labels,
    #         t_target=t,
    #         axis_limit=550.0,
    #         show_trajectory=True
    #     )
        
    #     # Save with naming convention: HW_t{time}_stress.pdf
    #     filename = output_dir / f"HW_t{int(t)}_stress.pdf"
    #     plt.savefig(filename, dpi=300, bbox_inches='tight')
    #     print(f"    Saved: {filename}")
    #     plt.close()
    
    # # Plot for PLANE STRAIN
    # print("\nCreating plots for Plane Strain...")
    # for t in times:
    #     print(f"  Time: {t:.1f} s")
        
    #     fig2, ax2 = plot_single_comparison(
    #         sim_folder_strain,
    #         labels,
    #         t_target=t,
    #         axis_limit=550.0,
    #         show_trajectory=True
    #     )
        
    #     # Save with naming convention: HW_t{time}_strain.pdf
    #     filename = output_dir / f"HW_t{int(t)}_strain.pdf"
    #     plt.savefig(filename, dpi=300, bbox_inches='tight')
    #     print(f"    Saved: {filename}")
    #     plt.close()
    
    # # Optional: Grid evolution plots
    # print("\nCreating evolution grids...")
    
    # # Grid for stress
    # fig_grid_stress, _ = plot_time_evolution_grid(
    #     sim_folder_stress,
    #     labels,
    #     times,
    #     axis_limit=550.0
    # )
    # filename_grid_stress = output_dir / "HW_evolution_stress.pdf"
    # plt.savefig(filename_grid_stress, dpi=300, bbox_inches='tight')
    # print(f"  Saved: {filename_grid_stress}")
    # plt.close()
    
    # # Grid for strain
    # fig_grid_strain, _ = plot_time_evolution_grid(
    #     sim_folder_strain,
    #     labels,
    #     times,
    #     axis_limit=550.0
    # )
    # filename_grid_strain = output_dir / "HW_evolution_strain.pdf"
    # plt.savefig(filename_grid_strain, dpi=300, bbox_inches='tight')
    # print(f"  Saved: {filename_grid_strain}")
    # plt.close()
    
    
    # NEW: Plot evolution for stepped loading
    print("\nCreating evolution plot for stepped loading...")
    
    stepped_folder = r"C:\Ecole\MASTER - 1\Advanced Solid Mecanics\Project\project_file\Results\triangular_loading\Linear_Mixed_hardening\eta10_2"
    
    # Define your three time steps here
    times_stepped = [0.65, 1.0, 1.35]  # Adjust these values as needed
    
    fig_stepped, _ = plot_single_simulation_evolution(
        stepped_folder,
        times_stepped,
        axis_limit=550.0,
        title="Yield Surface Evolution - Stepped Loading (Mixed Hardening, η=10⁻⁴)"
    )
    
    # Save BEFORE showing
    filename_stepped = output_dir / "HW_evolution_stepped_loading.pdf"
    plt.savefig(filename_stepped, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename_stepped}")
    
    # Show AFTER saving
    plt.show()
    plt.close()
    
    print("\n--- Done. All plots saved! ---")