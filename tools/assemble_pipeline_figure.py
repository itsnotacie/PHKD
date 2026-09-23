"""Assemble a concrete pipeline visualization from the previously-generated
example images.

Layout (2 rows x 4 cols = 8 panels showing the algorithm flow):

  [ Ground panorama ] --> [ Teacher p^T ] --> [ Peak-reweight q ] --> [ L_peak ]
                                                                          |
                                                                          v
  [ Satellite tile  ] --> [ Student p^S ] --> [ Hard negatives   ] --> [ L_hard ]

Also produces a compact 6-panel and a wider single-row version.
Outputs saved under figs/pipeline_examples/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch


IN = Path("/home/yiru_fang/ICASSP_27_paper/figs/pipeline_examples")


def load(name):
    return mpimg.imread(str(IN / name))


def big_pipeline_2row():
    """8-panel 2-row pipeline with real satellite / heatmap images."""
    fig = plt.figure(figsize=(15, 7))
    # Grid: 2 rows x 4 cols
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.20)

    panels = [
        # (row, col, image_name, title, subtitle)
        (0, 0, "01_ground_panorama.png",
         "Input: ground panorama",
         r"$I_{\mathrm{grd}}$"),
        (0, 1, "h05_teacher_overlay.png",
         "Teacher spatial score",
         r"$p^{T} = \mathrm{softmax}(z^{T}/\tau)$"),
        (0, 2, "h04_reweighted_log.png",
         "Peak-reweighted target",
         r"$q_i = w_i p_i^{T} / Z$, $\lambda=16$"),
        (0, 3, "h07_teacher_overlay_gt.png",
         "Teacher peak region (yellow box)",
         r"$\mathcal{P} = \{i : \|u_i - u_{t^*}\|_\infty \leq r\}$"),

        (1, 0, "03_satellite.png",
         "Input: satellite tile",
         r"$I_{\mathrm{sat}}$"),
        (1, 1, "h06_student_overlay.png",
         "Student spatial score",
         r"$p^{S} = \mathrm{softmax}(z^{S}/\tau)$"),
        (1, 2, "08_hard_negatives.png",
         "Hard negatives outside peak",
         r"$\mathcal{H} = \mathrm{TopK}_{i \notin \mathcal{P}}(p^{S}_i)$"),
        (1, 3, "h08_student_overlay_gt_pred.png",
         "Student prediction (red cross)",
         r"$\hat{i} = \arg\max_i p^{S}_i$"),
    ]

    for r, c, name, title, sub in panels:
        ax = fig.add_subplot(gs[r, c])
        try:
            img = load(name)
            ax.imshow(img)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"missing:\n{name}",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\n{sub}", fontsize=9)

    # Row labels (left of first column)
    fig.text(0.02, 0.73, "Teacher\n(frozen)",
             fontsize=12, ha="center", va="center",
             fontweight="bold", color="#8B0000",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFE4E1", edgecolor="#8B0000"))
    fig.text(0.02, 0.28, "Student\n(trained)",
             fontsize=12, ha="center", va="center",
             fontweight="bold", color="#003F7F",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#E1EAFF", edgecolor="#003F7F"))

    # Bottom legend for the two losses
    fig.text(0.5, 0.03,
             r"$\mathcal{L}_{\mathrm{peak}} = -\sum_i q_i \log p^{S}_i$"
             r"    $+$    "
             r"$\mathcal{L}_{\mathrm{hard}} = \sum_{i \in \mathcal{H}} p^{S}_i$",
             fontsize=11, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5DC", edgecolor="black"))

    out = IN / "pipeline_concrete_2row.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out}")
    # PDF version
    fig2 = plt.figure(figsize=(15, 7))
    gs = fig2.add_gridspec(2, 4, hspace=0.35, wspace=0.20)
    for r, c, name, title, sub in panels:
        ax = fig2.add_subplot(gs[r, c])
        try:
            img = load(name)
            ax.imshow(img)
        except FileNotFoundError:
            pass
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\n{sub}", fontsize=9)
    fig2.text(0.02, 0.73, "Teacher\n(frozen)", fontsize=12, ha="center",
              va="center", fontweight="bold", color="#8B0000",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFE4E1", edgecolor="#8B0000"))
    fig2.text(0.02, 0.28, "Student\n(trained)", fontsize=12, ha="center",
              va="center", fontweight="bold", color="#003F7F",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#E1EAFF", edgecolor="#003F7F"))
    fig2.text(0.5, 0.03,
              r"$\mathcal{L}_{\mathrm{peak}} = -\sum_i q_i \log p^{S}_i$"
              r"    $+$    "
              r"$\mathcal{L}_{\mathrm{hard}} = \sum_{i \in \mathcal{H}} p^{S}_i$",
              fontsize=11, ha="center", va="center",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5DC", edgecolor="black"))
    out_pdf = IN / "pipeline_concrete_2row.pdf"
    fig2.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig2)
    print(f"wrote {out_pdf}")


def compact_6panel():
    """Compact 2x3 grid: pano, sat, teacher, q, student, hard-neg."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    labels = [
        ("01_ground_panorama.png", "Ground panorama"),
        ("03_satellite.png", "Satellite tile"),
        ("h07_teacher_overlay_gt.png", r"Teacher $p^T$ + peak region"),
        ("h04_reweighted_log.png", r"Reweighted target $q$ ($\lambda=16$)"),
        ("h06_student_overlay.png", r"Student $p^S$"),
        ("08_hard_negatives.png", r"Hard negatives $\mathcal{H}$ (top-$K$ outside $\mathcal{P}$)"),
    ]
    for ax, (name, title) in zip(axes.flat, labels):
        try:
            img = load(name)
            ax.imshow(img)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"missing\n{name}", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    out = IN / "pipeline_concrete_6panel.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {out}")


def wide_horizontal():
    """Single-row horizontal flow with arrows."""
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    steps = [
        ("03_satellite.png", "Sat + panorama"),
        ("h05_teacher_overlay.png", r"Teacher $p^T$"),
        ("h04_reweighted_log.png", r"Reweight $\to q$"),
        ("h06_student_overlay.png", r"Student $p^S$"),
        ("08_hard_negatives.png", r"Hard negatives $\mathcal{H}$"),
    ]
    for ax, (name, title) in zip(axes, steps):
        try:
            img = load(name)
            ax.imshow(img)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"missing\n{name}", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    out = IN / "pipeline_concrete_horizontal.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    big_pipeline_2row()
    compact_6panel()
    wide_horizontal()
