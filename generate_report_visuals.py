"""Draw reproducible pipeline and model-architecture PNGs for both reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "final"
NAVY = "#15314B"
TEAL = "#087D87"
BLUE = "#2C6793"
PALE = "#E9F2F4"
LIGHT = "#F1F5F8"
MUTED = "#546675"


def box(ax, x, y, w, h, title, lines, *, fill=LIGHT, edge=BLUE):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.45,rounding_size=1.5",
                           linewidth=1.7, edgecolor=edge, facecolor=fill)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center",
            fontsize=15, fontweight="bold", color=NAVY)
    ax.text(x + w / 2, y + h / 2 - 2.0, lines, ha="center", va="center",
            fontsize=12.3, color=MUTED, linespacing=1.22)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=19, linewidth=2, color=TEAL,
                                 connectionstyle="arc3,rad=0"))


def canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(12.5, 8.8), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 104)
    ax.axis("off")
    ax.text(3, 101, title, va="top", ha="left", fontsize=20,
            color=NAVY, fontweight="bold")
    ax.text(3, 96, subtitle, va="top", ha="left", fontsize=12.5, color=MUTED)
    return fig, ax


def pipeline() -> Path:
    fig, ax = canvas("Data and evaluation pipeline",
                     "Each join uses the same Phase 1 row index; future flare annotations provide labels only.")
    box(ax, 3, 77, 44, 13, "Pre-sliced SWAN-SF windows",
        "Five partitions • 331,185 12-hour windows\nFrom the partition archives", fill=PALE)
    box(ax, 53, 77, 44, 13, "Full active-region series",
        "4,098 series • MFLARE/XFLARE annotations\n12-minute cadence", fill=PALE)
    box(ax, 3, 57, 44, 13, "Phase 1 shared features",
        "Canonical index and split assignments\n24 baseline + 120 temporal features")
    box(ax, 53, 57, 44, 13, "Lead-time targets",
        "Peak in (issue + H, issue + H + 24 h]\nH = 0, 6, 12, 24 or 48 h")
    box(ax, 21, 37, 58, 14, "Join and filter eligible rows",
        "Require complete 24-hour target interval after H\nAt least 114/120 expected 12-minute samples\nPhase 3 common cohort: 210,341 windows", fill="#F4FAF8", edge=TEAL)
    box(ax, 3, 18, 44, 13, "Split before model fitting",
        "Region and chronological generalization tests\nRandom split: overlap diagnostic only")
    box(ax, 53, 18, 44, 13, "Training-only preprocessing",
        "Saved median imputation and z-score statistics\nApplied unchanged to validation and test")
    box(ax, 3, 0, 44, 11, "Sequential model training",
        "Validation early stopping and F1 threshold", fill=PALE)
    box(ax, 53, 0, 44, 11, "Held-out verification and reports",
        "Reload models • check hashes and row IDs\n33/33 saved runs verified", fill=PALE)
    for a in [(25,77,25,70),(75,77,75,70),(25,57,43,51),(75,57,57,51),
              (50,37,25,31),(75,18,25,11),(47,24,53,24),
              (47,5.5,53,5.5)]:
        arrow(ax,*a)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pipeline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def architecture() -> Path:
    fig, ax = canvas("Forecast-model architecture",
                     "One feature contract, separate target labels, and a validation-only decision threshold.")
    box(ax, 3, 78, 44, 13, "Baseline input",
        "24 magnetic parameters at window end\nNo flare, X-ray or future columns", fill=PALE)
    box(ax, 53, 78, 44, 13, "Optional temporal input",
        "120 lags, rates and rolling summaries\nAll from the 12-hour observation period", fill=PALE)
    box(ax, 24, 58, 52, 13, "Model-ready feature matrix",
        "24 baseline or 144 baseline + temporal features\nSplit-specific training-only normalization", fill="#F4FAF8", edge=TEAL)
    box(ax, 3, 28, 44, 23, "Phase 2 • H = 0 h",
        "Logistic Regression • Random Forest (80 trees)\nXGBoost (≤180 trees, depth 3, CUDA)\nANN (32 → 16 → 1 units, CPU)\n2 splits × 2 feature sets × 4 models = 16 runs")
    box(ax, 53, 28, 44, 23, "Phase 3 • H = 6 / 12 / 24 / 48 h",
        "Separate M/X peak target for each H\nXGBoost with bounded CUDA training\nRegion and chronological splits\n4 leads × 2 splits × 2 feature sets = 16 runs")
    box(ax, 3, 5, 44, 16, "Validation partition",
        "PR-AUC early stopping where used\nDecision threshold maximizes validation F1", fill=PALE)
    box(ax, 53, 5, 44, 16, "Held-out test partition",
        "AP, ROC-AUC, P/R/F1, TSS, HSS\nRegion bootstrap and model interpretation", fill=PALE)
    for a in [(25,78,44,71),(75,78,56,71),(43,58,25,51),(57,58,75,51),
              (25,28,25,21),(75,28,75,21),(47,13,53,13)]:
        arrow(ax,*a)
    ax.text(50, 24.5, "+1 random-split XGBoost diagnostic = 33 verified runs total",
            ha="center", va="center", fontsize=10, color=TEAL,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CFDEE2"))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "architecture.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def build_visuals() -> tuple[Path, Path]:
    return pipeline(), architecture()


if __name__ == "__main__":
    for item in build_visuals():
        print(item)
