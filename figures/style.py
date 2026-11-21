"""
================================================================================
 FIGURE STYLE  --  publication-quality defaults
================================================================================

WHY A SHARED STYLE FILE
-----------------------
Every figure in the paper must look like it belongs to the same document. Fonts,
line weights, colours, and grid style are defined ONCE here so that changing the
look of the paper is a one-file edit rather than a hunt through every script.

DESIGN CHOICES AND WHY
----------------------
  * Serif fonts        -- match the body text of IEEE / Elsevier templates.
  * Colour-blind safe  -- ~8% of men have red-green colour deficiency. Reviewers
                          are people. The palette below is Okabe-Ito, which is
                          distinguishable under all common forms of CVD.
  * Redundant encoding -- colour is NEVER the only signal. Every series also has
                          a distinct marker, so figures survive greyscale
                          printing (still common in review copies).
  * Vector output      -- PDF for the paper (infinitely scalable), PNG for quick
                          viewing.
  * 300 DPI            -- the minimum most journals accept for raster figures.

SIZING
------
Widths are chosen for a two-column journal page:
    SINGLE_COL = 3.5 in   fits one column
    DOUBLE_COL = 7.16 in  spans both columns
Figures drawn at final size keep their fonts legible; scaling a figure down in
LaTeX shrinks the text with it, which is the most common reason figures look
bad in print.
================================================================================
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")            # non-interactive: works headless
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1 : PAGE GEOMETRY
# ==============================================================================

SINGLE_COL = 3.5      # inches -- one column of a two-column paper
DOUBLE_COL = 7.16     # inches -- full page width
GOLDEN = 0.618        # height/width ratio that generally looks right


# ==============================================================================
# SECTION 2 : COLOUR
# ==============================================================================
#
# Okabe-Ito palette: designed to stay distinguishable under protanopia,
# deuteranopia and tritanopia. Using it is a small courtesy that costs nothing
# and occasionally matters a great deal to a reader.
# ==============================================================================

COLORS = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "cyan":   "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
    "grey":   "#999999",
}

# Order used when a plot needs N distinct series.
CYCLE = [COLORS[c] for c in ["blue", "orange", "green", "red", "purple", "cyan"]]

# Markers pair with colours so series remain distinguishable in greyscale.
MARKERS = ["o", "s", "^", "D", "v", "P"]

# Fixed colour per observability rung, so `k1` is the same colour in EVERY
# figure of the paper. Consistency across figures is what lets a reader build
# intuition instead of re-reading each legend.
RUNG_COLORS = {
    "k0":   COLORS["grey"],       # decentralized endpoint
    "k1":   COLORS["blue"],
    "k2":   COLORS["green"],
    "k3":   COLORS["orange"],
    "k3+P": COLORS["red"],        # maximal centralization endpoint
}

RUNG_MARKERS = {
    "k0": "o", "k1": "s", "k2": "^", "k3": "D", "k3+P": "v",
}


# ==============================================================================
# SECTION 3 : RCPARAMS
# ==============================================================================

def apply_style() -> None:
    """Install the publication style. Call once at the top of every script."""
    plt.rcParams.update({
        # ---- Fonts ----------------------------------------------------------
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,

        # ---- Lines and markers ----------------------------------------------
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.8,

        # ---- Axes -----------------------------------------------------------
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,          # grid behind data, never on top of it
        "axes.spines.top": False,        # removing the box reduces visual noise
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=CYCLE),

        # ---- Grid: present but quiet ----------------------------------------
        "grid.color": "#CCCCCC",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,

        # ---- Legend ---------------------------------------------------------
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.borderpad": 0.4,

        # ---- Output ---------------------------------------------------------
        "figure.dpi": 150,               # on-screen
        "savefig.dpi": 300,              # journal minimum
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,              # embed TrueType so text stays editable
        "ps.fonttype": 42,
    })


# ==============================================================================
# SECTION 4 : HELPERS
# ==============================================================================

def save(fig, out_dir, name: str, formats=("pdf", "png")) -> list:
    """Save a figure in several formats and report the paths.

    PDF is what goes in the paper (vector, scales perfectly). PNG is for
    quick viewing and for pasting into slides or emails.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for fmt in formats:
        p = out_dir / f"{name}.{fmt}"
        fig.savefig(p, format=fmt)
        paths.append(p)

    return paths


def annotate_pilot_warning(fig, text: str = None) -> None:
    """Stamp a figure as PILOT DATA.

    Charter §3: the pilot is underpowered by design and must never be reported
    as a finding. Putting that on the figure itself means the warning travels
    with the image -- into slides, emails, and drafts -- rather than living only
    in a caption that gets separated from it.
    """
    text = text or (
        "PILOT DATA (5 seeds) — decides whether the full sweep is worth "
        "running; not a finding"
    )
    fig.text(
        0.5, -0.02, text,
        ha="center", va="top",
        fontsize=7, style="italic", color=COLORS["red"],
    )


def significance_stars(p: float) -> str:
    """Conventional significance markers."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."
