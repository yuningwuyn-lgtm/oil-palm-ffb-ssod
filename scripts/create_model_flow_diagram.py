from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript_jae" / "figures" / "fig_model_flow.png"


COLORS = {
    "data": "#E8F1F8",
    "model": "#E7F3E7",
    "quality": "#FFF1CC",
    "eval": "#F0E7F3",
    "risk": "#F7E0DC",
    "text": "#222222",
    "line": "#333333",
}


def box(ax, x, y, w, h, text, color, fontsize=10.5, lw=1.3):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=lw,
        edgecolor=COLORS["line"],
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        linespacing=1.12,
    )
    return patch


def arrow(ax, x1, y1, x2, y2, rad=0.0, lw=1.3, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=12,
            linewidth=lw,
            color=COLORS["line"],
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def lane_label(ax, y, text):
    ax.text(
        0.035,
        y,
        text,
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.plot([0.03, 0.97], [y - 0.035, y - 0.035], color="#D0D0D0", lw=0.9)


def main():
    fig, ax = plt.subplots(figsize=(15, 8.6), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.035,
        0.955,
        "Proposed model flow: quality-controlled SSOD for external-domain FFB detection",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        0.035,
        0.915,
        "The source-only model is used to expose domain shift; the fair comparison is between target-domain adaptation and strict SSOD.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#5A5A5A",
    )

    lane_label(ax, 0.835, "1. Supervised source model and target-domain adaptation")
    b1 = box(ax, 0.055, 0.705, 0.16, 0.085, "Source video-frame\ntraining split", COLORS["data"])
    b2 = box(ax, 0.27, 0.705, 0.16, 0.085, "Model 0\nsource-only YOLOv8n", COLORS["model"])
    b3 = box(ax, 0.49, 0.705, 0.17, 0.085, "External train/valid\n4-class masked domain", COLORS["data"])
    b4 = box(ax, 0.735, 0.705, 0.17, 0.085, "Model 2 Balanced\nfair adaptation baseline", COLORS["model"])
    arrow(ax, 0.215, 0.748, 0.27, 0.748)
    arrow(ax, 0.66, 0.748, 0.735, 0.748)
    arrow(ax, 0.43, 0.748, 0.49, 0.748)
    ax.text(0.455, 0.803, "domain-shift motivation", ha="center", fontsize=8.8, color="#666666")

    lane_label(ax, 0.625, "2. Multi-view pseudo-label quality control")
    c1 = box(ax, 0.055, 0.485, 0.145, 0.08, "Weak or\nunlabeled images", COLORS["data"])
    c2 = box(ax, 0.245, 0.485, 0.145, 0.08, "Teacher\nModel 2 Balanced", COLORS["model"])
    c3 = box(ax, 0.435, 0.485, 0.145, 0.08, "Original +\naugmented views", COLORS["quality"])
    c4 = box(ax, 0.625, 0.485, 0.145, 0.08, "Class and box\nconsistency", COLORS["quality"])
    c5 = box(ax, 0.815, 0.485, 0.145, 0.08, "Accepted\npseudo-labels", COLORS["model"])
    arrow(ax, 0.2, 0.525, 0.245, 0.525)
    arrow(ax, 0.39, 0.525, 0.435, 0.525)
    arrow(ax, 0.58, 0.525, 0.625, 0.525)
    arrow(ax, 0.77, 0.525, 0.815, 0.525)

    q1 = box(ax, 0.245, 0.385, 0.145, 0.055, "confidence", "#FFF7E2", fontsize=9.5, lw=1.0)
    q2 = box(ax, 0.435, 0.385, 0.145, 0.055, "box geometry", "#FFF7E2", fontsize=9.5, lw=1.0)
    q3 = box(ax, 0.625, 0.385, 0.145, 0.055, "edge penalty", "#FFF7E2", fontsize=9.5, lw=1.0)
    q4 = box(ax, 0.815, 0.385, 0.145, 0.055, "weak-folder prior", "#FFF7E2", fontsize=9.5, lw=1.0)
    ax.text(0.055, 0.412, "Quality score Q combines:", ha="left", va="center", fontsize=10.5, color="#555555")
    arrow(ax, 0.318, 0.485, 0.318, 0.44, style="->")
    arrow(ax, 0.507, 0.485, 0.507, 0.44, style="->")
    arrow(ax, 0.697, 0.485, 0.697, 0.44, style="->")
    arrow(ax, 0.887, 0.485, 0.887, 0.44, style="->")

    lane_label(ax, 0.315, "3. Precision-first self-training and locked evaluation")
    d0 = box(ax, 0.055, 0.185, 0.15, 0.085, "Accepted labels\nfrom QC", COLORS["model"])
    d1 = box(ax, 0.255, 0.185, 0.15, 0.085, "Class-wise dynamic\nthresholds", COLORS["quality"])
    d2 = box(ax, 0.455, 0.185, 0.15, 0.085, "Balanced quota\nper class", COLORS["quality"])
    d3 = box(ax, 0.655, 0.185, 0.15, 0.085, "Strict SSOD\nstudent model", COLORS["model"])
    d4 = box(ax, 0.835, 0.185, 0.13, 0.085, "Locked external\n+ scene test", COLORS["eval"], fontsize=10)
    arrow(ax, 0.205, 0.228, 0.255, 0.228)
    arrow(ax, 0.405, 0.228, 0.455, 0.228)
    arrow(ax, 0.605, 0.228, 0.655, 0.228)
    arrow(ax, 0.805, 0.228, 0.835, 0.228)

    callout = FancyBboxPatch(
        (0.055, 0.055),
        0.86,
        0.065,
        boxstyle="round,pad=0.014,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#666666",
        facecolor="#F7F7F7",
    )
    ax.add_patch(callout)
    ax.text(
        0.485,
        0.087,
        "Interpretation: Model 0 = domain-shift evidence; Model 2 Balanced = fair baseline; Strict SSOD = high-precision proposed setting.",
        ha="center",
        va="center",
        fontsize=9.6,
        color="#333333",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
