"""Figure for the Model 3 repair: (a) AUC after each fix on the row split and on new events,
(b) calibration on new events of the committed network and the repaired one."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import sys, json
from pathlib import Path

sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import plotstyle  # noqa: E402
from plotstyle import (BLUE, DEEMPH, INK, INK2, MUTED, ORANGE, TEXTWIDTH, dot, legend_line,  # noqa: E402
                       panel_title, setup_style)

O = Path(str(OUT))
plotstyle.FIGS = O
setup_style()
N = json.load(open(O / "numbers.json"))

rows = [("original", "As committed"), ("no_leak", "1  Drop sincere_o"),
        ("recode", "2  Recode, correct MI"), ("pc_parents", "3  PC picks match's family"),
        ("bdeu", "4  BDeu smoothing"), ("nb_k10_e5", "5  match → every variable")]
fig = plt.figure(figsize=(TEXTWIDTH, 1.95), layout="constrained")
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.75])

ax = fig.add_subplot(gs[0])
yy = np.arange(len(rows))[::-1]
for y, (k, _) in zip(yy, rows):
    a, b = N[f"{k}_their"]["auc"], N[f"{k}_wave"]["auc"]
    ax.plot([a, b], [y, y], color=DEEMPH, lw=1.0, zorder=2)
    dot(ax, a, y, MUTED, size=4.4)
    dot(ax, b, y, BLUE, size=4.4)
    ax.text(max(a, b) + 0.008, y, f"{b:.2f}", va="center", fontsize=6.2, color=INK)
lo, hi = N["ci"]["nb_wave"]
ax.hlines(yy[-1], lo, hi, color=BLUE, lw=0.9, zorder=3)
lr = N["lr_all_wave"]["auc"]
ax.axvline(lr, color=BLUE, lw=0.7, ls=(0, (2, 2)), zorder=1)
ax.text(lr - 0.003, yy[0] + 0.55, "Logistic regression,\nall pre-date variables", fontsize=5.6, color=INK2,
        ha="right", va="center")
ax.axvline(0.5, color=INK2, lw=0.7, zorder=1)
ax.set_yticks(yy)
ax.set_yticklabels([lbl for _, lbl in rows])
ax.set_ylim(-1.45, len(rows) - 0.1)
ax.set_xlim(0.48, 0.70)
ax.set_xlabel("AUC for match")
ax.grid(axis="y", visible=False)
panel_title(ax, "a", "Each fix, row split vs. new events")
ax.legend(handles=[legend_line(MUTED, "Row split (85/15)", lw=0),
                   legend_line(BLUE, "New events (by wave)", lw=0)],
          loc="lower right", ncol=2, fontsize=5.8, handletextpad=0.2, columnspacing=1.0,
          borderaxespad=0.1, frameon=True, framealpha=1, edgecolor="none")

ax = fig.add_subplot(gs[1])
edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.55, 0.999, 1.001])
for path, color, label in [(O / "ablation" / "original_wave_p.npy", ORANGE, "As committed"),
                           (O / "variants" / "nb_k10_e5_wave_p.npy", BLUE, "Repaired")]:
    p, y = np.load(path)
    b = np.digitize(p, edges[1:-1])
    xs, ys, ns = [], [], []
    for k in np.unique(b):
        m = b == k
        if m.sum() >= 30:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    ax.plot(xs, ys, color=color, lw=1.2, zorder=3)
    dot(ax, xs, ys, color, size=3.6)
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, zorder=1)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 0.62)
ax.set_xlabel("Predicted P(match)")
ax.set_ylabel("Observed match rate")
panel_title(ax, "b", "Calibration, new events")
ax.legend(handles=[legend_line(ORANGE, "As committed"), legend_line(BLUE, "Repaired")],
          loc="upper left", fontsize=5.8, borderaxespad=0.1)
plotstyle.save(fig, "fig_model3")
print("ok")
