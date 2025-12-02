import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


palette_okabe_ito = {
    "Green":  "#009E73",
    "Orange": "#E69F00",
    "Blue":   "#0072B2",
    "Light Blue": "#8cc5e3",
}

linestyles = ["-", "--", "-", "--"]
alphas = [1, 1, 1, 1]
chosen_palette = palette_okabe_ito

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "figure.autolayout": True,
    "font.size": 18,
    "axes.labelsize": 21,
    "axes.titlesize": 22,
    "legend.fontsize": 14,
    "axes.prop_cycle": cycler(color=list(chosen_palette.values()))
                        + cycler(linestyle=linestyles)
                        + cycler(alpha=alphas),

})

# --- Paramètres ---
Tcycle = 4.0   # période d'un cycle
Ncycle = 1    # nombre de cycles

# --- Nœuds (exactement comme dans ton code) ---
tau = (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]) / 8.0) * Tcycle
vals = np.array([0.0, 440.0, 440.0, 0.0, 0.0, -440.0, -440.0, 0.0, 0.0])

# --- Nœuds pour le triangular sawtooth ---
# tau = (np.array([0, 2, 4, 6, 8]) / 8.0) * Tcycle
# vals = np.array([0.0, 440.0, 0.0, -440.0, 0.0])


# tau  = np.array([0.0, 1.0, 16])
# vals = np.array([0.0, 440, 440])



# --- Construction sur N cycles (en évitant le doublon au raccord) ---
t_all, y_all = [], []
for i in range(Ncycle):
    t0 = i * Tcycle
    if i == 0:
        t_all.append(t0 + tau)
        y_all.append(vals)
    else:
        t_all.append(t0 + tau[1:])   # évite de répéter le point t=t0
        y_all.append(vals[1:])

t = np.concatenate(t_all)
y = np.concatenate(y_all)

tmax = np.max(np.abs(y))


# --- Plot ---
plt.figure(figsize=(5, 6))
plt.plot(t, y, linewidth=3)
# tmax = np.max(np.abs(y))

plt.ylim(-1.05*tmax, 1.05*tmax)  # optionnel mais rend le rendu propre
plt.yticks(
    [-tmax, 0, tmax],
    [r"$-t_{\max}$", r"$0$", r"$t_{\max}$"], 
    fontsize=19
)
xlabel = r"$\mathrm{time} \,[\mathrm{s}]$"
ylabel = r"$\mathrm{t}\, \,[\mathrm{MPa}]$"
plt.xlabel(xlabel)
plt.ylabel(ylabel, labelpad=-20)
# plt.ylim(16, 450)
plt.grid(True, ls="--")
from matplotlib.ticker import MultipleLocator
plt.gca().xaxis.set_major_locator(MultipleLocator(4))
plt.tight_layout()
plt.show()


