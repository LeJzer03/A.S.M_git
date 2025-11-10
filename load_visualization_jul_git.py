import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from pathlib import Path

# Plot style (same layout as plot_vinc.py)
palette_1 = {
    "Med Blue": "#3594cc",
    "Light Blue": "#8cc5e3",
    "Med Orange": "#ea801c",
    "Light Orange": "#f0b077"
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

# Classe simplifiée pour simuler PieceWiseLinearFunction
class PieceWiseLinearFunction:
    def __init__(self):
        self.times = []
        self.values = []
    
    def setData(self, time, value):
        self.times.append(time)
        self.values.append(value)
    
    def getValue(self, t):
        return np.interp(t, self.times, self.values)

# LOAD PARAMETERS
Trac = -100.0       # Traction (MPa)
Ncycle = 1         # Number of cycles
Tcycle = 4.0       # Duration of one cycle

# Define fct (cyclic loading)
fct = PieceWiseLinearFunction()
for i in range(0, Ncycle):
    fct.setData(i*Tcycle + 0., 0.)
    fct.setData(i*Tcycle + Tcycle/4.0, 1.0)
    fct.setData(i*Tcycle + Tcycle*2.0/4.0, 0.0)
    fct.setData(i*Tcycle + Tcycle*3.0/4.0, -1.0)
    fct.setData(i*Tcycle + Tcycle*4.0/4.0, 0.0)

# Define fct2 (cycle + hold)
fct2 = PieceWiseLinearFunction()
for i in range(0, Ncycle):
    fct2.setData(i*Tcycle + 0., 0.)
    fct2.setData(i*Tcycle + Tcycle/5.0, 1.0)
    fct2.setData(i*Tcycle + Tcycle*2/5.0, 0.0)
    fct2.setData(i*Tcycle + Tcycle*3.0/5.0, -1.0)
    fct2.setData(i*Tcycle + Tcycle*4.0/5.0, 0.0)
    fct2.setData(i*Tcycle + Tcycle*5.0/5.0, 0.0)

# Define fct3 (cycle with zero holds)
fct3 = PieceWiseLinearFunction()
for i in range(0, Ncycle):
    fct3.setData(i*Tcycle + 0., 0.)
    fct3.setData(i*Tcycle + Tcycle/5.0, 1.0)
    fct3.setData(i*Tcycle + Tcycle*2/5.0, 0.0)
    fct3.setData(i*Tcycle + Tcycle*3/5.0, 0.0)
    fct3.setData(i*Tcycle + Tcycle*4/5.0, 0.0)
    fct3.setData(i*Tcycle + Tcycle*5/5.0, 0.0)

# Generate time array
time_array = np.linspace(0, Ncycle*Tcycle, 1000)

# Compute pressure values (P = -Trac * fct)
pressure_fct = np.array([fct.getValue(t) * (-Trac) for t in time_array])
pressure_fct2 = np.array([fct2.getValue(t) * (-Trac) for t in time_array])
pressure_fct3 = np.array([fct3.getValue(t) * (-Trac) for t in time_array])

# Plot: three-panel view (same layout as plot_vinc)
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(time_array, pressure_fct, linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Pressure (MPa)')
plt.title('fct: Cyclic Loading')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.subplot(1, 3, 2)
plt.plot(time_array, pressure_fct2, linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Pressure (MPa)')
plt.title('fct2: Single Cycle + Hold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.subplot(1, 3, 3)
plt.plot(time_array, pressure_fct3, linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Pressure (MPa)')
plt.title('fct3: Cycle with Zero Holds')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()



t_max = Trac  # amplitude used for y limits (MPa)
t_end = Ncycle * Tcycle
x_ticks = np.arange(0, int(np.ceil(t_end)) + 1, 1)

plt.figure(figsize=(4, 4))
ax = plt.gca()
ax.plot(time_array, pressure_fct, linewidth=3)
ax.set_xlabel(r"time [$\mathrm{s}$]")
ax.set_ylabel(r"t [MPa]")
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linestyle='--', linewidth=0.6)
ax.set_xticks(x_ticks)
#ax.set_ylim(-t_max, t_max)
ax.set_yticks([-t_max, 0.0, t_max])
ax.set_yticklabels([r"$t_{\mathrm{max}}$", r"$0$", r"$-t_{\mathrm{max}}$"])

plt.tight_layout()


out_dir = Path(r"C:\Ecole\MASTER - 1\Advanced Solid Mecanics\Project\Illustration Report")
out_dir.mkdir(parents=True, exist_ok=True)
fig_path = out_dir / "load_visualization.pdf"
plt.savefig(str(fig_path))
print(f"Figure saved to: {fig_path}")

plt.show()




