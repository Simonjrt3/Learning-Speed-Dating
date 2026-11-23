"""Plot style shared by every figure (same palette, font and mark specs as the data report).

Figures are drawn at their final printed size (NeurIPS text width = 5.5 in), so fonts are never
rescaled by LaTeX. Bars have 4 pt-ish rounded data ends (square at the baseline), lines are thin,
markers carry a surface-coloured ring, gridlines are solid hairlines.
"""
from __future__ import annotations

import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

FIGS = None  # set by the caller (fig_model3.py sets plotstyle.FIGS = OUT)

TEXTWIDTH = 5.5          # inches (NeurIPS)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SLOTS
GENDER_COLOR = {"F": BLUE, "M": ORANGE}
GENDER_LABEL = {"F": "Women", "M": "Men"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
DEEMPH = "#c3c2b7"
SURFACE = "#ffffff"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd"] + BLUE_RAMP[::2])
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f2a6a4", "#e34948", "#a3292b"])
METHOD_COLOR = {"rejection": ORANGE, "lw": BLUE, "gibbs": AQUA}
METHOD_LABEL = {"rejection": "Rejection sampling", "lw": "Likelihood weighting", "gibbs": "Gibbs sampling"}


def setup_style():
    for f in glob.glob("/usr/share/fonts/truetype/google-fonts/ofl/firasans/FiraSans-*.ttf"):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
    have = {f.name for f in fm.fontManager.ttflist if f.fname.lower().endswith(".ttf")}
    family = next((f for f in ["Fira Sans", "Arial", "Helvetica", "Liberation Sans"] if f in have), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": family, "font.size": 7.6,
        "axes.titlesize": 7.8, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "axes.titlepad": 6,
        "axes.labelsize": 7.2, "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.minor.size": 0, "ytick.minor.size": 0,
        "xtick.major.pad": 3, "ytick.major.pad": 3,
        "legend.frameon": False, "legend.fontsize": 6.6, "legend.handlelength": 1.2, "legend.labelcolor": INK2,
        "lines.linewidth": 1.4, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 4.0,
        "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.prop_cycle": cycler(color=SLOTS),
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
    })


# ------------------------------------------------------------------ rounded bars (drawn at save time)
def rbar(ax, pos, length, width, color, base=0.0, orient="v", radius_pt=1.6, zorder=2):
    pos = np.atleast_1d(np.asarray(pos, dtype=float))
    length = np.atleast_1d(np.asarray(length, dtype=float))
    base = np.broadcast_to(np.asarray(base, dtype=float), pos.shape)
    colors = [color] * len(pos) if isinstance(color, str) else list(color)
    if orient == "v":
        ax.bar(pos, length, width=width, bottom=base, alpha=0)
    else:
        ax.barh(pos, length, height=width, left=base, alpha=0)
    reg = ax.figure.__dict__.setdefault("_rbars", [])
    for p, l, b, c in zip(pos, length, base, colors):
        if np.isfinite(l) and l != 0:
            reg.append((ax, float(p), float(l), float(width), c, float(b), orient, radius_pt, zorder))


def _draw_rounded(fig):
    reg = fig.__dict__.get("_rbars", [])
    if not reg:
        return
    fig.canvas.draw()
    if fig.get_layout_engine() is not None:
        fig.set_layout_engine("none")
    for ax, p, l, w, c, b, orient, r_pt, z in reg:
        bb = ax.get_window_extent()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        upp_x = abs(x1 - x0) / bb.width * fig.dpi / 72.0
        upp_y = abs(y1 - y0) / bb.height * fig.dpi / 72.0
        if ax.get_xscale() != "linear" or ax.get_yscale() != "linear":
            continue
        rx, ry = r_pt * upp_x, r_pt * upp_y
        sgn = 1.0 if l > 0 else -1.0
        if orient == "v":
            xa, xb, ya, yb = p - w / 2, p + w / 2, b, b + l
            rx, ry = min(rx, w / 2), min(ry, abs(l) / 2)
            verts = [(xa, ya), (xa, yb - sgn * ry), (xa, yb), (xa + rx, yb), (xb - rx, yb), (xb, yb),
                     (xb, yb - sgn * ry), (xb, ya), (xa, ya)]
        else:
            ya, yb, xa, xb = p - w / 2, p + w / 2, b, b + l
            ry, rx = min(ry, w / 2), min(rx, abs(l) / 2)
            verts = [(xa, ya), (xb - sgn * rx, ya), (xb, ya), (xb, ya + ry), (xb, yb - ry), (xb, yb),
                     (xb - sgn * rx, yb), (xa, yb), (xa, ya)]
        codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3,
                 MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=c, edgecolor="none", zorder=z))
    fig.__dict__["_rbars"] = []


def save(fig, name):
    _draw_rounded(fig)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def dot(ax, x, y, color, size=4.2, zorder=4, **kw):
    return ax.plot(x, y, "o", color=color, markersize=size, markeredgecolor=SURFACE, markeredgewidth=0.9,
                   zorder=zorder, linestyle="none", **kw)


def legend_line(color, label, marker="o", lw=1.4):
    from matplotlib.lines import Line2D
    return Line2D([0], [0], color=color, lw=lw, marker=marker, markersize=4.2, markeredgecolor=SURFACE,
                  markeredgewidth=0.9, label=label)


def legend_patch(color, label):
    from matplotlib.patches import Patch
    return Patch(facecolor=color, edgecolor="none", label=label)


def panel_title(ax, letter, text):
    ax.set_title(f"{letter}   {text}")
