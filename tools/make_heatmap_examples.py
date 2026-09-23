"""Generate a rich set of heatmap visualizations for illustration.

Produces (for a hand-picked VIGOR test sample, index 87):

Raw heatmaps (no overlay):
  h01_teacher_raw.png
  h02_student_raw.png
  h03_reweighted_target_raw.png
  h04_reweighted_log.png (log-scale for visibility)

Heatmaps overlaid on satellite:
  h05_teacher_overlay.png
  h06_student_overlay.png
  h07_teacher_overlay_gt.png (with GT star)
  h08_student_overlay_gt_pred.png (with GT + prediction)

Side-by-side comparisons:
  h09_teacher_vs_student.png (2 panels)
  h10_teacher_reweight_student.png (3 panels)

All jet/hot colormaps. Fixes the peak-region alignment bug by using
corr_h/corr_w resolution instead of feat_h.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/home/yiru_fang/geokd-main")
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from train_geokd import (
    build_localization_model,
    load_checkpoint_state,
    load_teacher,
    make_backbone,
    forward_vigor,
    unwrap_model,
)
from dataset.VIGOR import VIGOR


OUT_DIR = Path("/home/yiru_fang/ICASSP_27_paper/figs/pipeline_examples")
SAMPLE_IDX = 87

TEACHER_CKPT = "/home/yiru_fang/geokd-main/ckpt/translation/vigor/cross-geodistill-dino.pth"
STUDENT_CKPT = REPO / "checkpoints/vigor/geokd/student/vigor-cross-peak-hard-peak-w16_best.pth"


def make_args():
    return SimpleNamespace(
        dataset="vigor", dataset_root="/home/yiru_fang/VIGOR",
        cross_area=True, image_size=512, sat_size=640, zoom=20, bev_size=512,
        grid_size=8, ori_noise=45,
        teacher_dino_model="vitb14", student_dino_model="vits14", student_width=0.5,
        init_from_teacher=False, levels=[0, 2], channels=[64, 16, 4],
        student_temp=0.06, teacher_temp=0.06, distill_mode="peak_hard",
        train=False, batch_size_per_gpu=1, seed=2023, clip=1.0,
        eval_target="student", lr=1e-4, min_lr=1e-5, weight_decay=1e-5,
        distributed=False, gpuid=[0], num_workers=0,
        save_path="/tmp/heatmaps", wandb=False, wandb_log_interval=20,
        visualize=False, save_visualization=False, vis_freq=100,
        supervision="distill", supervised_loss="contrastive",
        localization_sigma=2.0, localization_temperature=10.0, contrastive_weight=0.1,
        reset_scheduler=False, reset_optimizer=False, reset_epoch=False,
        peak_radius=2, peak_topk=1, peak_region_weight=16.0,
        hard_negative_topk=32, hard_negative_weight=1.0,
        boundary_weight=0.0, boundary_margin=1.0,
        uncertainty_weighting=False, uncertainty_weight_min=0.25, uncertainty_weight_power=1.0,
        freq_cutoff=0.25, freq_low_weight=1.0, freq_high_weight=1.0, freq_ce_weight=1.0,
        teacher_ckpts=None, teacher_dino_models=None, teacher_weights=None,
        student_ckpt=None, teacher_ckpt=TEACHER_CKPT,
        dino_model="vitb14", batch_size=1, name="heat", model_name="heat",
    )


def load_student(device):
    args = make_args()
    args.eval_target = "student"
    args.student_ckpt = str(STUDENT_CKPT)
    m = build_localization_model(args, dataset="vigor",
        dino_model=args.student_dino_model, student_width=args.student_width).to(device)
    state, _ = load_checkpoint_state(str(STUDENT_CKPT), map_location=device)
    m.load_state_dict(state, strict=True)
    m.eval()
    d = make_backbone(args.student_dino_model).to(device).eval()
    for p in d.parameters(): p.requires_grad = False
    return m, d, args


def load_teacher_m(device):
    args = make_args()
    args.eval_target = "teacher"
    m, _ = load_teacher(args, device)
    m.eval()
    d = make_backbone(args.teacher_dino_model).to(device).eval()
    for p in d.parameters(): p.requires_grad = False
    return m, d, args


def collate(sample):
    (bev, sat, pano_gps, sat_gps, ori_angle,
     sat_delta, mpp, resized_pano, rotated_pano, city) = sample
    def _b(t):
        if isinstance(t, torch.Tensor): return t.unsqueeze(0)
        if isinstance(t, np.ndarray): return torch.from_numpy(t).unsqueeze(0)
        return [t]
    return [
        _b(bev), _b(sat), _b(pano_gps), _b(sat_gps),
        torch.tensor([ori_angle], dtype=torch.float32),
        _b(sat_delta), _b(mpp), _b(resized_pano),
        _b(rotated_pano), [city],
    ]


def get_spatial_prob(model, dino, args, batch, device, temp=0.06):
    """Return (prob_full_512x512, prob_grid, corr_h, corr_w, gt_px, mpp)."""
    with torch.no_grad():
        out = forward_vigor(args, dino, model, batch, device)
        model_ref = unwrap_model(model)
        corr = model_ref.calc_corr_for_val(
            out["sat_feat"], out["sat_conf"], out["g2s_feat"], out["g2s_conf"],
        )
        b, corr_h, corr_w = corr.shape
        prob = F.softmax(corr.reshape(b, -1) / temp, dim=1).reshape(b, corr_h, corr_w)
        prob_full = F.interpolate(prob.unsqueeze(1), size=(512, 512),
                                  mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    sat_delta = out["sat_delta"].cpu().numpy()[0]
    gt_px = (256 + sat_delta[0] * 128.0, 256 + sat_delta[1] * 128.0)
    return prob_full, prob[0].cpu().numpy(), corr_h, corr_w, gt_px, float(out["meter_per_pixel"])


def compute_peak_rect(prob_grid, corr_h, corr_w, r=2):
    """Use corr_h/corr_w directly (fixes prior bug)."""
    argmax_flat = int(np.argmax(prob_grid))
    peak_row = argmax_flat // corr_w
    peak_col = argmax_flat % corr_w
    # px_per_feat should map corr grid to sat pixel; corr is at feat resolution
    px_per_feat_w = 512 / corr_w
    px_per_feat_h = 512 / corr_h
    size_w = (2 * r + 1) * px_per_feat_w
    size_h = (2 * r + 1) * px_per_feat_h
    # Cell (row, col) center in sat pixel coords: 512 * (col + 0.5) / corr_w
    cx = 512 * (peak_col + 0.5) / corr_w
    cy = 512 * (peak_row + 0.5) / corr_h
    x0 = cx - size_w / 2
    y0 = cy - size_h / 2
    return peak_row, peak_col, (x0, y0, size_w, size_h)


def reweight_target(teacher_prob_grid, peak_row, peak_col, r, lam):
    corr_h, corr_w = teacher_prob_grid.shape
    mask = np.zeros_like(teacher_prob_grid)
    for i in range(max(0, peak_row - r), min(corr_h, peak_row + r + 1)):
        for j in range(max(0, peak_col - r), min(corr_w, peak_col + r + 1)):
            mask[i, j] = 1.0
    w = 1.0 + (lam - 1.0) * mask
    q = w * teacher_prob_grid
    q /= q.sum()
    return q, mask


def upscale(grid, size=512):
    g = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).float()
    return F.interpolate(g, size=(size, size), mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def save_raw_heatmap(hm, path, cmap="hot", show_colorbar=False, title=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(hm, cmap=cmap)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {path}")


def save_overlay(sat, hm, path, alpha=0.5, peak_rect=None, gt_px=None,
                 pred_px=None, title=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(sat)
    norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    ax.imshow(norm, cmap="hot", alpha=alpha)
    if peak_rect is not None:
        ax.add_patch(Rectangle(
            (peak_rect[0], peak_rect[1]), peak_rect[2], peak_rect[3],
            linewidth=2.0, edgecolor="yellow", facecolor="none", linestyle="--",
        ))
    if gt_px is not None:
        ax.scatter([gt_px[0]], [gt_px[1]], marker="*", s=220,
                   edgecolor="black", facecolor="lime", linewidth=1.0, zorder=6)
    if pred_px is not None:
        ax.scatter([pred_px[0]], [pred_px[1]], marker="x", s=140,
                   color="red", linewidth=2.4, zorder=6)
    ax.set_xlim(0, 512); ax.set_ylim(512, 0)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {path}")


def save_side_by_side(imgs, titles, path, ncols=None):
    n = len(imgs)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.6 * nrows),
                             squeeze=False)
    for k, (im, ti) in enumerate(zip(imgs, titles)):
        ax = axes[k // ncols][k % ncols]
        ax.imshow(im)
        ax.set_title(ti, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.axis("off")
    # hide unused
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    args_data = make_args()
    vigor = VIGOR(args_data, "test")
    sample = vigor[SAMPLE_IDX]
    sat = sample[1].permute(1, 2, 0).numpy().astype(np.uint8)
    batch = collate(sample)

    # Teacher
    tm, td, ta = load_teacher_m(device)
    t_full, t_grid, corr_h, corr_w, gt_px, mpp = get_spatial_prob(tm, td, ta, batch, device)
    peak_row, peak_col, peak_rect = compute_peak_rect(t_grid, corr_h, corr_w, r=2)
    q_grid, _ = reweight_target(t_grid, peak_row, peak_col, r=2, lam=16.0)
    q_full = upscale(q_grid)

    # Teacher prediction (its own argmax)
    t_argmax = int(np.argmax(t_grid))
    t_pred_px = (512 * (t_argmax % corr_w + 0.5) / corr_w,
                 512 * (t_argmax // corr_w + 0.5) / corr_h)

    del tm, td
    torch.cuda.empty_cache()

    # Student
    sm, sd, sa = load_student(device)
    s_full, s_grid, sh, sw, _, _ = get_spatial_prob(sm, sd, sa, batch, device)
    s_argmax = int(np.argmax(s_grid))
    s_pred_px = (512 * (s_argmax % sw + 0.5) / sw,
                 512 * (s_argmax // sw + 0.5) / sh)

    # === Raw heatmaps ===
    save_raw_heatmap(t_full, OUT_DIR / "h01_teacher_raw.png", title="Teacher softmax score map")
    save_raw_heatmap(s_full, OUT_DIR / "h02_student_raw.png", title="Student softmax score map")
    save_raw_heatmap(q_full, OUT_DIR / "h03_reweighted_target_raw.png",
                     title=r"Reweighted target $q$ ($\lambda=16$)")
    save_raw_heatmap(np.log1p(q_full * 1e4),
                     OUT_DIR / "h04_reweighted_log.png",
                     title=r"Reweighted target $q$ (log scale)")

    # === Overlays on satellite ===
    save_overlay(sat, t_full, OUT_DIR / "h05_teacher_overlay.png", alpha=0.5,
                 title="Teacher heatmap on satellite")
    save_overlay(sat, s_full, OUT_DIR / "h06_student_overlay.png", alpha=0.5,
                 title="Student heatmap on satellite")
    save_overlay(sat, t_full, OUT_DIR / "h07_teacher_overlay_gt.png", alpha=0.5,
                 peak_rect=peak_rect, gt_px=gt_px, pred_px=t_pred_px,
                 title="Teacher: GT star, teacher argmax cross, peak box")
    save_overlay(sat, s_full, OUT_DIR / "h08_student_overlay_gt_pred.png", alpha=0.5,
                 peak_rect=peak_rect, gt_px=gt_px, pred_px=s_pred_px,
                 title="Student: GT star, student argmax cross, peak box")

    # === Side-by-side comparisons ===
    # 2-panel: teacher vs student raw heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(t_full, cmap="hot")
    axes[0].set_title("Teacher $p^T$", fontsize=12)
    axes[1].imshow(s_full, cmap="hot")
    axes[1].set_title("Student $p^S$", fontsize=12)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(OUT_DIR / "h09_teacher_vs_student.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'h09_teacher_vs_student.png'}")

    # 3-panel: teacher, reweighted target q, student
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(t_full, cmap="hot")
    axes[0].set_title(r"Teacher $p^T$", fontsize=12)
    axes[1].imshow(np.log1p(q_full * 1e4), cmap="hot")
    axes[1].set_title(r"Reweighted target $q$ ($\lambda=16$, log)", fontsize=12)
    axes[2].imshow(s_full, cmap="hot")
    axes[2].set_title(r"Student $p^S$", fontsize=12)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(OUT_DIR / "h10_teacher_reweight_student.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'h10_teacher_reweight_student.png'}")

    # 4-panel: satellite, teacher overlay, student overlay, peak region marked
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(sat); axes[0].set_title("Satellite tile", fontsize=12)
    axes[1].imshow(sat)
    axes[1].imshow((t_full - t_full.min()) / (t_full.max() - t_full.min() + 1e-8),
                   cmap="hot", alpha=0.45)
    axes[1].set_title("Teacher heatmap overlay", fontsize=12)
    axes[2].imshow(sat)
    axes[2].imshow((s_full - s_full.min()) / (s_full.max() - s_full.min() + 1e-8),
                   cmap="hot", alpha=0.45)
    axes[2].set_title("Student heatmap overlay", fontsize=12)
    axes[3].imshow(sat)
    axes[3].add_patch(Rectangle((peak_rect[0], peak_rect[1]),
                                peak_rect[2], peak_rect[3],
                                linewidth=2.0, edgecolor="yellow",
                                facecolor="none", linestyle="--"))
    axes[3].scatter([gt_px[0]], [gt_px[1]], marker="*", s=220,
                    edgecolor="black", facecolor="lime", linewidth=1.0, zorder=6)
    axes[3].scatter([s_pred_px[0]], [s_pred_px[1]], marker="x", s=140,
                    color="red", linewidth=2.4, zorder=6)
    axes[3].set_title("Peak region + GT + student pred", fontsize=12)
    for ax in axes:
        ax.set_xlim(0, 512); ax.set_ylim(512, 0)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(OUT_DIR / "h11_4panel_overlay.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'h11_4panel_overlay.png'}")

    print(f"\n[heatmaps] all files under {OUT_DIR}/")


if __name__ == "__main__":
    main()
