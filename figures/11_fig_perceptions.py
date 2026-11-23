"""Compact 3-panel figure for the 'closer look at perceptions' section: stated vs revealed weights
(women, men) and sampling error under rare evidence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from plotstyle import (DEEMPH, GENDER_COLOR, INK, INK2, METHOD_COLOR, MUTED, TEXTWIDTH, dot, legend_line,  # noqa: E402
                       legend_patch, panel_title, rbar, save, setup_style)
from prep import RESULTS  # noqa: E402

setup_style()
LBL = {"Attr": "Attractive", "Sinc": "Sincere", "Intel": "Intelligent", "Fun": "Fun", "Amb": "Ambitious",
       "Shar": "Shared interests"}
q2 = pd.read_csv(RESULTS / "q2_stated_revealed.csv")
se = pd.read_csv(RESULTS / "sampling_error.csv")
fig = plt.figure(figsize=(TEXTWIDTH, 1.8), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 0.88, 1.0])
for k, g in enumerate(["F", "M"]):
    ax = fig.add_subplot(gs[k])
    s = q2[q2.gender == g].set_index("trait").loc[["Amb", "Intel", "Sinc", "Shar", "Fun", "Attr"]]
    yy = np.arange(len(s))
    rbar(ax, yy + 0.17, s.stated_share, 0.3, DEEMPH, orient="h")
    rbar(ax, yy - 0.17, s["share_A-halo"], 0.3, GENDER_COLOR[g], orient="h")
    ax.hlines(yy - 0.17, s["share_A-halo_lo95"], s["share_A-halo_hi95"], color=INK2, lw=0.8, zorder=4)
    a = s.loc["Attr"]
    ax.text(a["share_A-halo"] + 0.07, 5 - 0.17, f"{a['share_A-halo']:.0%}", va="center", fontsize=6.4, color=INK)
    ax.text(a.stated_share + 0.015, 5 + 0.17, f"{a.stated_share:.0%}", va="center", fontsize=6.4, color=INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels([LBL[v] for v in s.index] if k == 0 else [])
    ax.set_xlim(0, 0.8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_xlabel("Share of weight")
    ax.grid(axis="y", visible=False)
    panel_title(ax, "ab"[k], "Women" if g == "F" else "Men")
    ax.legend(handles=[legend_patch(DEEMPH, "Stated"), legend_patch(GENDER_COLOR[g], "Revealed")],
              loc="lower right", fontsize=6.0, handlelength=0.9, borderaxespad=0.1)
ax = fig.add_subplot(gs[2])
for m in ["rejection", "lw", "gibbs"]:
    r = se[(se["query"] == "rare") & (se.method == m)]
    ax.plot(r.N, r.mae, color=METHOD_COLOR[m], lw=1.4)
    dot(ax, r.N.iloc[[0, 6, 12]], r.mae.iloc[[0, 6, 12]], METHOD_COLOR[m], size=3.4)
nref = np.array([100, 100000])
ax.plot(nref, 0.11 * (nref / 100) ** -0.5, color=MUTED, lw=0.7)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(8e-5, 0.6)
ax.set_xlabel("Number of samples N")
ax.set_ylabel("Mean absolute error")
panel_title(ax, "c", "Sampling, rare evidence")
ax.legend(handles=[legend_line(METHOD_COLOR[m], {"rejection": "Rejection", "lw": "Lik. weighting", "gibbs": "Gibbs"}[m],
                               marker=None) for m in ["rejection", "lw", "gibbs"]], loc="lower left", fontsize=5.9)
save(fig, "fig_perceptions")
print("ok")
