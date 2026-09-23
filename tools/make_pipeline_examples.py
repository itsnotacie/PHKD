"""Generate concrete example images illustrating the PHKD pipeline stages.

Renders (for a hand-picked VIGOR test sample):
  01_ground_panorama.png    - raw ground panorama input
  02_bev_projection.png     - BEV-projected panorama (student input)
  03_satellite.png          - satellite tile (search space)
  04_teacher_heatmap.png    - teacher softmax spatial map
  05_teacher_peak_region.png- peak region highlighted on satellite
  06_reweighted_target.png  - q distribution (teacher amplified around peak)
  07_student_heatmap.png    - student softmax spatial map
  08_hard_negatives.png     - student top-K outside peak, marked on satellite
  09_final_prediction.png   - GT and student prediction on satellite
Stored under: figs/pipeline_examples/
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
        save_path="/tmp/pipeline_examples", wandb=False, wandb_log_interval=20,
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
        dino_model="vitb14", batch_size=1, name="pipe", model_name="pipe",
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
    """Returns (corr_full, corr_h, corr_w, mpp, gt_px, feat_h)."""
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
        _, _, feat_h, _ = out["sat_feat"][args.levels[-1]].shape
    sat_delta = out["sat_delta"].cpu().numpy()[0]
    gt_px = (256 + sat_delta[0] * 128.0, 256 + sat_delta[1] * 128.0)
    return prob_full, prob[0].cpu().numpy(), (corr_h, corr_w), float(out["meter_per_pixel"]), gt_px, feat_h


def compute_peak_region(prob_grid, feat_h, r=2):
    """Return set of feature-grid cells in peak region and its bbox in sat pixels."""
    corr_h, corr_w = prob_grid.shape
    argmax_flat = int(np.argmax(prob_grid))
    peak_row = argmax_flat // corr_w
    peak_col = argmax_flat % corr_w
    px_per_feat = 512 / feat_h
    size = (2 * r + 1) * px_per_feat
    x0 = 256 + (peak_col - corr_w / 2) * px_per_feat - size / 2
    y0 = 256 + (peak_row - corr_h / 2) * px_per_feat - size / 2
    return peak_row, peak_col, (x0, y0, size), px_per_feat


def reweight_target(teacher_prob_grid, peak_row, peak_col, r, lam):
    """Apply peak reweighting: w=1+(lam-1)*I[peak], q = w*p / sum(w*p)."""
    corr_h, corr_w = teacher_prob_grid.shape
    mask = np.zeros_like(teacher_prob_grid)
    for i in range(max(0, peak_row - r), min(corr_h, peak_row + r + 1)):
        for j in range(max(0, peak_col - r), min(corr_w, peak_col + r + 1)):
            mask[i, j] = 1.0
    w = 1.0 + (lam - 1.0) * mask
    q = w * teacher_prob_grid
    q /= q.sum()
    return q, mask


def upscale_to_sat(grid, size=512):
    """Nearest-neighbor upscale probability grid to a 512x512 heatmap for overlay."""
    g = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).float()
    out = F.interpolate(g, size=(size, size), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def save_image_only(img, out_path, cmap=None):
    """Save an array as PNG with no axes/margin."""
    fig, ax = plt.subplots(figsize=(4, 4))
    if cmap:
        ax.imshow(img, cmap=cmap)
    else:
        ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis("off")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  wrote {out_path}")


def save_sat_with_overlay(sat_img, out_path, heatmap=None, peak_rect=None,
                          gt_px=None, pred_px=None, hard_neg_pxs=None,
                          heatmap_alpha=0.5, title=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(sat_img)
    if heatmap is not None:
        # normalize to [0,1]
        h = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        ax.imshow(h, cmap="hot", alpha=heatmap_alpha)
    if peak_rect is not None:
        ax.add_patch(Rectangle(
            (peak_rect[0], peak_rect[1]), peak_rect[2], peak_rect[2],
            linewidth=1.8, edgecolor="yellow", facecolor="none", linestyle="--",
        ))
    if gt_px is not None:
        ax.scatter([gt_px[0]], [gt_px[1]], marker="*", s=220,
                   edgecolor="black", facecolor="lime", linewidth=1.0, zorder=6)
    if pred_px is not None:
        ax.scatter([pred_px[0]], [pred_px[1]], marker="x", s=140,
                   color="red", linewidth=2.4, zorder=6)
    if hard_neg_pxs is not None:
        xs = [p[0] for p in hard_neg_pxs]
        ys = [p[1] for p in hard_neg_pxs]
        ax.scatter(xs, ys, marker="o", s=60, edgecolor="magenta", facecolor="none",
                   linewidth=1.4, zorder=6)
    ax.set_xlim(0, 512); ax.set_ylim(512, 0)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    args_data = make_args()
    vigor = VIGOR(args_data, "test")
    sample = vigor[SAMPLE_IDX]
    (bev, sat, pano_gps, sat_gps, ori_angle,
     sat_delta, mpp, resized_pano, rotated_pano, city) = sample
    batch = collate(sample)

    # (1) Ground panorama (original)
    save_image_only(np.asarray(resized_pano).astype(np.uint8),
                    OUT_DIR / "01_ground_panorama.png")

    # (2) BEV projection (student input)
    bev_img = bev.permute(1, 2, 0).numpy().astype(np.uint8)
    save_image_only(bev_img, OUT_DIR / "02_bev_projection.png")

    # (3) Satellite tile
    sat_img = sat.permute(1, 2, 0).numpy().astype(np.uint8)
    save_image_only(sat_img, OUT_DIR / "03_satellite.png")

    # Teacher forward
    tm, td, ta = load_teacher_m(device)
    t_prob_full, t_prob_grid, (th, tw), t_mpp, gt_px, feat_h = get_spatial_prob(
        tm, td, ta, batch, device, temp=0.06,
    )
    peak_row, peak_col, peak_rect, px_per_feat = compute_peak_region(t_prob_grid, feat_h, r=2)

    # (4) Teacher spatial heatmap (standalone, jet-colored)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(t_prob_full, cmap="hot")
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis("off")
    fig.savefig(OUT_DIR / "04_teacher_heatmap.png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / '04_teacher_heatmap.png'}")

    # (5) Peak region on satellite
    save_sat_with_overlay(
        sat_img, OUT_DIR / "05_teacher_peak_region.png",
        heatmap=t_prob_full, peak_rect=peak_rect, gt_px=gt_px,
        heatmap_alpha=0.45,
    )

    # (6) Reweighted target q
    q_grid, peak_mask = reweight_target(t_prob_grid, peak_row, peak_col, r=2, lam=16.0)
    q_full = upscale_to_sat(q_grid)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(q_full, cmap="hot")
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis("off")
    fig.savefig(OUT_DIR / "06_reweighted_target.png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / '06_reweighted_target.png'}")

    # Free teacher, load student
    del tm, td
    torch.cuda.empty_cache()
    sm, sd, sa = load_student(device)
    s_prob_full, s_prob_grid, (sh, sw), s_mpp, _, s_feat_h = get_spatial_prob(
        sm, sd, sa, batch, device, temp=0.06,
    )

    # (7) Student spatial heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(s_prob_full, cmap="hot")
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis("off")
    fig.savefig(OUT_DIR / "07_student_heatmap.png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / '07_student_heatmap.png'}")

    # (8) Hard-negative candidates: top-K of student EXCLUDING peak region
    K = 32
    flat = s_prob_grid.copy().flatten()
    # Zero out cells inside peak region so they can't be selected
    for i in range(max(0, peak_row - 2), min(sh, peak_row + 3)):
        for j in range(max(0, peak_col - 2), min(sw, peak_col + 3)):
            flat[i * sw + j] = -np.inf
    topk_idx = np.argsort(-flat)[:K]
    px_per_feat_s = 512 / s_feat_h
    hn_pxs = []
    for idx in topk_idx:
        row = idx // sw
        col = idx % sw
        x = 256 + (col - sw / 2) * px_per_feat_s
        y = 256 + (row - sh / 2) * px_per_feat_s
        hn_pxs.append((x, y))

    save_sat_with_overlay(
        sat_img, OUT_DIR / "08_hard_negatives.png",
        heatmap=s_prob_full, peak_rect=peak_rect, gt_px=gt_px,
        hard_neg_pxs=hn_pxs, heatmap_alpha=0.4,
    )

    # (9) Final prediction: student argmax + GT
    s_argmax = int(np.argmax(s_prob_grid))
    s_row = s_argmax // sw
    s_col = s_argmax % sw
    pred_px = (256 + (s_col - sw / 2) * px_per_feat_s,
               256 + (s_row - sh / 2) * px_per_feat_s)
    save_sat_with_overlay(
        sat_img, OUT_DIR / "09_final_prediction.png",
        gt_px=gt_px, pred_px=pred_px,
    )

    # Also a combined 3-panel summary
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(sat_img)
    axes[0].imshow((t_prob_full - t_prob_full.min()) /
                   (t_prob_full.max() - t_prob_full.min() + 1e-8),
                   cmap="hot", alpha=0.45)
    axes[0].add_patch(Rectangle((peak_rect[0], peak_rect[1]),
                                peak_rect[2], peak_rect[2],
                                linewidth=1.8, edgecolor="yellow",
                                facecolor="none", linestyle="--"))
    axes[0].set_title("Teacher heatmap + peak region", fontsize=10)

    axes[1].imshow(q_full, cmap="hot")
    axes[1].set_title(r"Reweighted target $q$ ($\lambda\!=\!16$)", fontsize=10)

    axes[2].imshow(sat_img)
    axes[2].imshow((s_prob_full - s_prob_full.min()) /
                   (s_prob_full.max() - s_prob_full.min() + 1e-8),
                   cmap="hot", alpha=0.4)
    xs = [p[0] for p in hn_pxs]; ys = [p[1] for p in hn_pxs]
    axes[2].scatter(xs, ys, marker="o", s=40, edgecolor="magenta",
                    facecolor="none", linewidth=1.2, zorder=6)
    axes[2].add_patch(Rectangle((peak_rect[0], peak_rect[1]),
                                peak_rect[2], peak_rect[2],
                                linewidth=1.8, edgecolor="yellow",
                                facecolor="none", linestyle="--"))
    axes[2].set_title(r"Student heatmap + hard negatives (K=32)", fontsize=10)

    for ax in axes:
        ax.set_xlim(0, 512); ax.set_ylim(512, 0)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(OUT_DIR / "10_summary_3panel.png", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {OUT_DIR / '10_summary_3panel.png'}")

    print(f"\n[pipeline] all files written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
