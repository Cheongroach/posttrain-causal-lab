"""Render the conceptual intervention diagram; no experimental data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main() -> None:
    output = Path(__file__).resolve().parent
    font = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"}
    )
    fig, ax = plt.subplots(figsize=(14, 7.4), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 1400), ylim=(0, 740))
    ax.axis("off")
    ink, muted, border, green = "#1b2631", "#566573", "#ccd1d1", "#19734b"

    def node(x, y, width, height, title, subtitle="", focus=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=10",
                linewidth=1.5 if focus else 1.0,
                edgecolor=green if focus else border,
                facecolor="#f8f9f9",
            )
        )
        ax.text(
            x + width / 2,
            y + height / 2 + (13 if subtitle else 0),
            title,
            ha="center",
            va="center",
            color=ink,
            fontsize=11.4,
        )
        if subtitle:
            ax.text(
                x + width / 2,
                y + height / 2 - 15,
                subtitle,
                ha="center",
                va="center",
                color=muted,
                fontsize=9.6,
            )

    def arrow(start, end):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.2,
                color=muted,
                shrinkA=2,
                shrinkB=2,
            )
        )

    ax.text(60, 700, "Post-training causal tracing", fontsize=17, color=ink)
    ax.text(
        60,
        672,
        "Proposed interventions; no measured effects",
        fontsize=10.5,
        color=muted,
    )
    positions = [60, 405, 750, 1095]
    labels = [
        ("Training signal", "Data / teacher / human feedback"),
        ("LoRA update", "Frozen backbone + delta weights"),
        ("Internal computation", "Matched prefix / hook / position"),
        ("Behavior", "Correctness / preference / format"),
    ]
    for x, (title, subtitle) in zip(positions, labels):
        node(x, 537, 245, 92, title, subtitle)
    for x, next_x in zip(positions, positions[1:]):
        arrow((x + 245, 583), (next_x, 583))

    for x, width in [(60, 625), (725, 615)]:
        ax.add_patch(
            FancyBboxPatch(
                (x, 60),
                width,
                425,
                boxstyle="round,pad=0,rounding_size=12",
                linewidth=1,
                edgecolor=border,
                facecolor="white",
            )
        )

    ax.text(85, 452, "A. Intervene during training", fontsize=13, color=ink)
    ax.text(
        85,
        427,
        "Tests signal changes from the same starting state",
        fontsize=10.5,
        color=muted,
    )
    node(215, 320, 310, 72, "Common checkpoint", "Weights + optimizer + RNG + cursor")
    node(85, 178, 250, 75, "Original signal", "Unmodified continuation")
    node(
        410,
        178,
        250,
        75,
        "Changed signal",
        "Targeted / random / heuristic",
        True,
    )
    arrow((307, 320), (210, 253))
    arrow((433, 320), (535, 253))
    ax.text(370, 276, "branch", fontsize=9.8, color=muted, ha="center")
    node(215, 82, 310, 64, "Paired evaluation", "Quality + regressions + total cost")
    arrow((210, 178), (305, 146))
    arrow((535, 178), (435, 146))

    ax.text(
        750,
        452,
        "B. Intervene during model execution",
        fontsize=13,
        color=ink,
    )
    ax.text(
        750,
        427,
        "Tests update / activation effects on behavior",
        fontsize=10.5,
        color=muted,
    )
    node(
        750,
        315,
        255,
        77,
        "Recipient checkpoint",
        "Same prompt + same token prefix",
    )
    node(
        1065,
        315,
        250,
        77,
        "Intervention source",
        "Donor checkpoint / update subset",
    )
    node(
        750,
        178,
        565,
        76,
        "Replace activation or retain / remove LoRA update",
        "Forward restore + reverse removal + identity / random controls",
        True,
    )
    arrow((877, 315), (955, 254))
    arrow((1190, 315), (1110, 254))
    node(
        750,
        82,
        565,
        64,
        "Recompute behavior under the intervention",
        "Inference effects do not identify training-history causes",
    )
    arrow((1032, 178), (1032, 146))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output / "intervention-design.svg", bbox_inches=None)
    fig.savefig(output / "intervention-design.png", bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    main()
