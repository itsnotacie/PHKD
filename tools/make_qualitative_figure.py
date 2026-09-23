"""Generate the qualitative comparison figure for PHKD.

Loads five VIGOR models (teacher, KD, Peak-only, Hard-only, PHKD),
runs them on a small set of fixed VIGOR cross-area test samples,
and renders a grid figure with heatmap overlays, GT / prediction
markers, and per-cell localization errors.

Output: figs/qualitative_comparison.pdf (in the paper repo).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace

REPO = Path("/home/yiru_fang/geokd-main")
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

from train_geokd import (
    build_localization_model,
    load_checkpoint_state,
    load_teacher,
    make_backbone,
    forward_vigor,
    unwrap_model,
)
from dataset.VIGOR import VIGOR


# Models to render (order = columns in the figure).
MODEL_SPECS = [
    ("Teacher", "teacher", None,
        "/home/yiru_fang/geokd-main/ckpt/translation/vigor/cross-geodistill-dino.pth"),
    ("KD", "student",
        REPO / "checkpoints/vigor/geokd/student/vigor-cross-hinton-t1_best.pth", None),
    ("DIST", "student",
        REPO / "checkpoints/vigor/geokd/student/vigor-cross-dist_best.pth", None),
    ("DKD", "student",
        REPO / "checkpoints/vigor/geokd/student/vigor-cross-dkd_best.pth", None),
    ("AB", "student",
        REPO / "checkpoints/vigor/geokd/student/vigor-cross-boundary_best.pth", None),
    ("DS$^2$D$^2$", "student",
        REPO / "checkpoints/vigor/geokd/student/vigor-cross-wavelet_best.pth", None),
    ("PHKD (Ours)", "student",
        REPO / "checkpoints/vigor/geokd/student/vigor-cross-peak-hard-peak-w16_best.pth", None),
]

TEACHER_SPECS = MODEL_SPECS[0]
STUDENT_TEACHER_CKPT = (
    "/home/yiru_fang/geokd-main/ckpt/translation/vigor/cross-geodistill-dino.pth"
)

VIGOR_ROOT = "/home/yiru_fang/VIGOR"

# Fixed test indices for reproducibility.
SAMPLE_INDICES = [0, 200]


def make_args(dataset_root: str = VIGOR_ROOT) -> SimpleNamespace:
    """Minimal args namespace mimicking a config YAML for eval-only paths."""
    return SimpleNamespace(
        dataset="vigor",
        dataset_root=dataset_root,
        cross_area=True,
        image_size=512,
        sat_size=640,
        zoom=20,
        bev_size=512,
        grid_size=8,
        ori_noise=45,
        teacher_dino_model="vitb14",
        student_dino_model="vits14",
        student_width=0.5,
        init_from_teacher=False,
        levels=[0, 2],
        channels=[64, 16, 4],
        student_temp=0.06,
        teacher_temp=0.06,
        distill_mode="peak_hard",
        train=False,
        batch_size_per_gpu=1,
        seed=2023,
        clip=1.0,
        eval_target="student",
        lr=1e-4,
        min_lr=1e-5,
        weight_decay=1e-5,
        distributed=False,
        gpuid=[0],
        num_workers=0,
        save_path="/tmp/qualfig",
        wandb=False,
        wandb_log_interval=20,
        visualize=False,
        save_visualization=False,
        vis_freq=100,
        # required by loss/setup code paths even in eval-only
        supervision="distill",
        supervised_loss="contrastive",
        localization_sigma=2.0,
        localization_temperature=10.0,
        contrastive_weight=0.1,
        reset_scheduler=False,
        reset_optimizer=False,
        reset_epoch=False,
        peak_radius=2,
        peak_topk=1,
        peak_region_weight=4.0,
        hard_negative_topk=32,
        hard_negative_weight=1.0,
        boundary_weight=0.0,
        boundary_margin=1.0,
        uncertainty_weighting=False,
        uncertainty_weight_min=0.25,
        uncertainty_weight_power=1.0,
        freq_cutoff=0.25,
        freq_low_weight=1.0,
        freq_high_weight=1.0,
        freq_ce_weight=1.0,
        teacher_ckpts=None,
        teacher_dino_models=None,
        teacher_weights=None,
        student_ckpt=None,
        teacher_ckpt=STUDENT_TEACHER_CKPT,
        dino_model="vitb14",
        batch_size=1,
        name="qualfig",
        model_name="qualfig",
    )


def load_student_model(ckpt_path: Path, device: torch.device):
    """Instantiate the ViT-S/14 SlimDPT student and load weights."""
    args = make_args()
    args.eval_target = "student"
    args.student_ckpt = str(ckpt_path)

    model = build_localization_model(
        args,
        dataset="vigor",
        dino_model=args.student_dino_model,
        student_width=args.student_width,
    ).to(device)
    state, _ = load_checkpoint_state(str(ckpt_path), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    dino = make_backbone(args.student_dino_model).to(device).eval()
    for p in dino.parameters():
        p.requires_grad = False

    return model, dino, args


def load_teacher_model(ckpt_path: str, device: torch.device):
    args = make_args()
    args.eval_target = "teacher"
    args.teacher_ckpt = ckpt_path

    model, _ = load_teacher(args, device)
    model.eval()
    dino = make_backbone(args.teacher_dino_model).to(device).eval()
    for p in dino.parameters():
        p.requires_grad = False
    return model, dino, args


def infer_one(model, dino, args, batch, device):
    """Return (corr_hw, pred_pixel_offset, gt_pixel_offset, meter_per_pixel)."""
    with torch.no_grad():
        out = forward_vigor(args, dino, model, batch, device)
        model_ref = unwrap_model(model)
        corr = model_ref.calc_corr_for_val(
            out["sat_feat"],
            out["sat_conf"],
            out["g2s_feat"],
            out["g2s_conf"],
        )
        # corr: (B=1, H, W). Take level-0 (finest) via the args.levels[-1] used
        # by validate_vigor. Rescale offsets to satellite pixels (512x512).
        b, corr_h, corr_w = corr.shape
        max_index = torch.argmax(corr.reshape(b, -1), dim=1)
        pred_col = (max_index % corr_w).float() - corr_w / 2
        pred_row = (max_index // corr_w).float() - corr_h / 2

        _, _, feat_h, _ = out["sat_feat"][args.levels[-1]].shape
        # Scale from feature-grid offset to sat-pixel offset.
        px_per_feat = 512 / feat_h
        pred_px = (float(pred_col) * px_per_feat, float(pred_row) * px_per_feat)
        mpp = float(out["meter_per_pixel"])

    # GT offset (batch coords → pixel coords on 512-sat).
    sat_delta = out["sat_delta"].cpu().numpy()[0]  # [gt_shift_x, gt_shift_y]
    gt_px = (sat_delta[0] * 128.0, sat_delta[1] * 128.0)  # sat_delta is in units of L/4=128 pixels

    # Corr map upsampled to 512x512 for overlay.
    corr_soft = F.softmax(corr.reshape(b, -1) / args.student_temp, dim=1)
    corr_soft = corr_soft.reshape(b, corr_h, corr_w)
    corr_full = F.interpolate(
        corr_soft.unsqueeze(1), size=(512, 512), mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()

    return corr_full, pred_px, gt_px, mpp, (corr_h, corr_w), int(max_index.item())


def teacher_peak_box(pred_px, feat_dims, radius=2):
    """Return (x0, y0, w, h) axis-aligned rectangle for teacher's peak region."""
    feat_h, feat_w = feat_dims
    px_per_feat = 512 / feat_h
    # radius in feature cells → converted to sat pixels
    size = (2 * radius + 1) * px_per_feat
    x0 = 256 + pred_px[0] - size / 2
    y0 = 256 + pred_px[1] - size / 2
    return (x0, y0, size, size)


def render(all_results, sample_meta, out_path):
    """Compose the grid figure."""
    n_rows = len(all_results)
    n_cols = len(MODEL_SPECS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.1 * n_cols, 2.25 * n_rows),
        squeeze=False,
    )

    # Build a translucent hot colormap for the heatmap overlay.
    hot_cmap = plt.cm.hot
    for i, (sat_img, results) in enumerate(all_results):
        # Teacher (col 0) gives us the peak box shared across the row.
        teacher_pred_px = results[0]["pred_px"]
        teacher_feat_dims = results[0]["feat_dims"]
        peak_rect = teacher_peak_box(teacher_pred_px, teacher_feat_dims)

        row_max = max(r["corr"].max() for r in results)
        row_min = min(r["corr"].min() for r in results)

        for j, r in enumerate(results):
            ax = axes[i][j]
            ax.imshow(sat_img)
            # normalized heatmap
            norm = (r["corr"] - row_min) / (row_max - row_min + 1e-8)
            ax.imshow(norm, cmap=hot_cmap, alpha=0.45)
            # peak-region dashed box (from teacher)
            ax.add_patch(Rectangle(
                (peak_rect[0], peak_rect[1]), peak_rect[2], peak_rect[3],
                linewidth=1.6, edgecolor="yellow", facecolor="none", linestyle="--",
            ))
            # GT star
            gx = 256 + r["gt_px"][0]
            gy = 256 + r["gt_px"][1]
            ax.scatter([gx], [gy], marker="*", s=140, edgecolor="black",
                       facecolor="lime", linewidth=0.8, zorder=5)
            # Pred cross
            px = 256 + r["pred_px"][0]
            py = 256 + r["pred_px"][1]
            ax.scatter([px], [py], marker="x", s=80, color="red",
                       linewidth=1.8, zorder=5)

            # Distance in meters
            err = np.sqrt(
                (r["pred_px"][0] - r["gt_px"][0]) ** 2
                + (r["pred_px"][1] - r["gt_px"][1]) ** 2
            ) * r["mpp"]
            ax.set_xlim(0, 512)
            ax.set_ylim(512, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(MODEL_SPECS[j][0], fontsize=9)
            ax.set_xlabel(f"err = {err:.2f} m", fontsize=8)

        axes[i][0].text(
            -0.12, 0.5, sample_meta[i],
            transform=axes[i][0].transAxes, rotation=90,
            va="center", ha="right", fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[qualfig] wrote {out_path}")


def collate_batch(sample):
    """Wrap a single VIGOR sample into a 1-element batch matching forward_vigor's expectation."""
    (bev, sat, pano_gps, sat_gps, ori_angle,
     sat_delta, meter_per_pixel, resized_pano, rotated_pano, city) = sample
    def _b(t):
        if isinstance(t, torch.Tensor):
            return t.unsqueeze(0)
        if isinstance(t, np.ndarray):
            return torch.from_numpy(t).unsqueeze(0)
        return [t]
    return [
        _b(bev), _b(sat), _b(pano_gps), _b(sat_gps),
        torch.tensor([ori_angle], dtype=torch.float32),
        _b(sat_delta), _b(meter_per_pixel), _b(resized_pano),
        _b(rotated_pano), [city],
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=int, nargs="*", default=SAMPLE_INDICES)
    ap.add_argument("--out", type=Path,
                    default=Path("/home/yiru_fang/ICASSP_27_paper/figs/qualitative_comparison.pdf"))
    args_cli = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load VIGOR test dataset.
    data_args = make_args()
    vigor = VIGOR(data_args, "test")

    # Preload models sequentially to save GPU memory.
    all_results = []
    sample_meta = []
    for idx in args_cli.indices:
        sample = vigor[idx]
        sat_img_tensor = sample[1]  # (3, 512, 512)
        sat_img = sat_img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)

        batch = collate_batch(sample)
        # Move batch to device below inside forward.
        row_results = []
        for label, target, ckpt, teacher_ckpt in MODEL_SPECS:
            print(f"[qualfig] sample idx={idx}  model={label}")
            if target == "teacher":
                model, dino, m_args = load_teacher_model(teacher_ckpt, device)
            else:
                model, dino, m_args = load_student_model(ckpt, device)
            corr, pred_px, gt_px, mpp, feat_dims, argmax_i = infer_one(
                model, dino, m_args, batch, device,
            )
            row_results.append({
                "label": label,
                "corr": corr,
                "pred_px": pred_px,
                "gt_px": gt_px,
                "mpp": mpp,
                "feat_dims": feat_dims,
            })
            # free GPU memory
            del model, dino
            torch.cuda.empty_cache()

        all_results.append((sat_img, row_results))
        sample_meta.append(f"{sample[-1]}  #{idx}")

    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    render(all_results, sample_meta, args_cli.out)


if __name__ == "__main__":
    main()
