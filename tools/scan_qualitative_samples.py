"""Scan VIGOR test samples and find ones where PHKD outperforms all 6 baselines.

Runs each of the 7 models on the first N test samples, computes per-sample
localization error, and prints the top samples ranked by
(best baseline error) - (PHKD error), i.e. how much PHKD wins by.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path("/home/yiru_fang/geokd-main")
sys.path.insert(0, str(REPO))

from train_geokd import (
    build_localization_model,
    load_checkpoint_state,
    load_teacher,
    make_backbone,
    forward_vigor,
    unwrap_model,
)
from dataset.VIGOR import VIGOR


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
    ("DS2D2", "student",
        REPO / "checkpoints/vigor/geokd/student/vigor-cross-wavelet_best.pth", None),
    ("PHKD", "student",
        REPO / "checkpoints/vigor/geokd/student/vigor-cross-peak-hard-peak-w16_best.pth", None),
]

STUDENT_TEACHER_CKPT = (
    "/home/yiru_fang/geokd-main/ckpt/translation/vigor/cross-geodistill-dino.pth"
)


def make_args():
    return SimpleNamespace(
        dataset="vigor",
        dataset_root="/home/yiru_fang/VIGOR",
        cross_area=True,
        image_size=512, sat_size=640, zoom=20, bev_size=512, grid_size=8, ori_noise=45,
        teacher_dino_model="vitb14", student_dino_model="vits14", student_width=0.5,
        init_from_teacher=False, levels=[0, 2], channels=[64, 16, 4],
        student_temp=0.06, teacher_temp=0.06, distill_mode="peak_hard",
        train=False, batch_size_per_gpu=1, seed=2023, clip=1.0,
        eval_target="student", lr=1e-4, min_lr=1e-5, weight_decay=1e-5,
        distributed=False, gpuid=[0], num_workers=0,
        save_path="/tmp/qualfig", wandb=False, wandb_log_interval=20,
        visualize=False, save_visualization=False, vis_freq=100,
        supervision="distill", supervised_loss="contrastive",
        localization_sigma=2.0, localization_temperature=10.0, contrastive_weight=0.1,
        reset_scheduler=False, reset_optimizer=False, reset_epoch=False,
        peak_radius=2, peak_topk=1, peak_region_weight=4.0,
        hard_negative_topk=32, hard_negative_weight=1.0,
        boundary_weight=0.0, boundary_margin=1.0,
        uncertainty_weighting=False, uncertainty_weight_min=0.25, uncertainty_weight_power=1.0,
        freq_cutoff=0.25, freq_low_weight=1.0, freq_high_weight=1.0, freq_ce_weight=1.0,
        teacher_ckpts=None, teacher_dino_models=None, teacher_weights=None,
        student_ckpt=None, teacher_ckpt=STUDENT_TEACHER_CKPT,
        dino_model="vitb14", batch_size=1, name="scan", model_name="scan",
    )


def load_student(ckpt: Path, device: torch.device):
    args = make_args()
    args.eval_target = "student"
    args.student_ckpt = str(ckpt)
    model = build_localization_model(
        args, dataset="vigor",
        dino_model=args.student_dino_model, student_width=args.student_width,
    ).to(device)
    state, _ = load_checkpoint_state(str(ckpt), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    dino = make_backbone(args.student_dino_model).to(device).eval()
    for p in dino.parameters():
        p.requires_grad = False
    return model, dino, args


def load_teacher_m(ckpt: str, device: torch.device):
    args = make_args()
    args.eval_target = "teacher"
    args.teacher_ckpt = ckpt
    model, _ = load_teacher(args, device)
    model.eval()
    dino = make_backbone(args.teacher_dino_model).to(device).eval()
    for p in dino.parameters():
        p.requires_grad = False
    return model, dino, args


def collate(sample):
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


def predict_error(model, dino, args, batch, device):
    """Return localization error in meters for a single sample."""
    with torch.no_grad():
        out = forward_vigor(args, dino, model, batch, device)
        model_ref = unwrap_model(model)
        corr = model_ref.calc_corr_for_val(
            out["sat_feat"], out["sat_conf"], out["g2s_feat"], out["g2s_conf"],
        )
        b, corr_h, corr_w = corr.shape
        max_index = torch.argmax(corr.reshape(b, -1), dim=1)
        pred_col = (max_index % corr_w).float() - corr_w / 2
        pred_row = (max_index // corr_w).float() - corr_h / 2
        _, _, feat_h, _ = out["sat_feat"][args.levels[-1]].shape
        px_per_feat = 512 / feat_h
        pred_px_x = float(pred_col) * px_per_feat
        pred_px_y = float(pred_row) * px_per_feat
        mpp = float(out["meter_per_pixel"])
    sat_delta = out["sat_delta"].cpu().numpy()[0]
    gt_px_x = sat_delta[0] * 128.0
    gt_px_y = sat_delta[1] * 128.0
    err_px = np.sqrt((pred_px_x - gt_px_x) ** 2 + (pred_px_y - gt_px_y) ** 2)
    return err_px * mpp


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    device = torch.device("cuda:0")

    args = make_args()
    vigor = VIGOR(args, "test")
    print(f"[scan] test set size = {len(vigor)}; scanning first {N} samples")

    # Load all models once (memory permitting).
    models = []
    for label, target, ckpt, teacher_ckpt in MODEL_SPECS:
        if target == "teacher":
            m, d, ma = load_teacher_m(teacher_ckpt, device)
        else:
            m, d, ma = load_student(ckpt, device)
        models.append((label, m, d, ma))
        print(f"[scan] loaded {label}")

    errors = np.zeros((N, len(models)), dtype=np.float32)  # [n_samples, n_models]

    for i in range(N):
        sample = vigor[i]
        batch = collate(sample)
        for j, (label, m, d, ma) in enumerate(models):
            errors[i, j] = predict_error(m, d, ma, batch, device)
        if (i + 1) % 20 == 0:
            print(f"[scan] {i+1}/{N}")

    # Rank by PHKD win margin over BEST baseline.
    # PHKD is index 6, baselines are 1..5 (excluding Teacher at 0).
    phkd_idx = 6
    baseline_indices = list(range(1, 6))
    best_baseline_err = errors[:, baseline_indices].min(axis=1)
    margin = best_baseline_err - errors[:, phkd_idx]  # positive = PHKD wins

    # Also require PHKD absolute error be small.
    ranked = np.argsort(-margin)  # descending margin
    print("\n=== Top 30 samples where PHKD wins by biggest margin ===")
    print(f"{'idx':>5s}  {'margin(m)':>10s}  {'PHKD':>8s}  {'best_bl':>8s}  |  " +
          "  ".join(f"{lab[:6]:>6s}" for lab, *_ in models))
    for r in ranked[:30]:
        row_str = "  ".join(f"{errors[r, j]:6.2f}" for j in range(len(models)))
        print(f"{r:5d}  {margin[r]:10.3f}  {errors[r, phkd_idx]:8.3f}  "
              f"{best_baseline_err[r]:8.3f}  |  {row_str}")

    # Also print candidates with PHKD very small AND margin > 1m
    print("\n=== Best cases: PHKD <1m AND margin > 0.5m ===")
    good = np.where((errors[:, phkd_idx] < 1.0) & (margin > 0.5))[0]
    good = good[np.argsort(-margin[good])]
    for r in good[:20]:
        row_str = "  ".join(f"{errors[r, j]:6.2f}" for j in range(len(models)))
        print(f"{r:5d}  {margin[r]:10.3f}  {errors[r, phkd_idx]:8.3f}  "
              f"{best_baseline_err[r]:8.3f}  |  {row_str}")


if __name__ == "__main__":
    main()
