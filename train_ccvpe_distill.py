"""Distill the frozen GeoDistill DINOv2 teacher into a CCVPE (CNN) student on VIGOR.

Reuses the geokd VIGOR dataloader + teacher (DINOv2 ViT-B + DPT correlation heatmap)
and the CCVPE CVM_VIGOR student. Pure response distillation on the localization heatmap;
supports distill_mode ce / peak_hard / freq (reusing geokd's loss functions).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/yiru_fang/CCVPE")

import numpy as np
import torch
import torch.nn.functional as F

# Use file_system sharing so DataLoader workers survive if $TMPDIR (e.g. per-VSCode-session
# tmp dir) is cleaned mid-run; the default file_descriptor strategy loses its Unix socket.
torch.multiprocessing.set_sharing_strategy("file_system")

from dataset.VIGOR import VIGOR, fetch_dataloader
from model.dino import DINO
from model.geokd import build_localization_model, load_checkpoint_state
from model.loss import cross_entropy, boundary_distillation
from train_geokd import namespace_from_config, xy_metrics, print_colored

from models import CVM_VIGOR
# CCVPE's native soft-CE + soft-InfoNCE losses (label = Gaussian, not one-hot)
sys.path.insert(0, "/home/yiru_fang/CCVPE")
from losses import infoNCELoss as ccvpe_infoNCE, cross_entropy_loss as ccvpe_ce_soft

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _build_gauss_bump_and_pyramid(peak_x, peak_y, H=512, W=512, sigma=4.0, device=None):
    """Given per-sample peak (x=col, y=row) at (H, W) resolution, return a single-channel
    Gaussian bump [B, 1, H, W] and its 6-scale maxpool pyramid matching CCVPE deep supervision.
    """
    yy = torch.arange(H, device=device).float().view(1, -1, 1)  # [1, H, 1]
    xx = torch.arange(W, device=device).float().view(1, 1, -1)  # [1, 1, W]
    dy = yy - peak_y.view(-1, 1, 1)
    dx = xx - peak_x.view(-1, 1, 1)
    gauss = torch.exp(-(dy * dy + dx * dx) / (2.0 * sigma * sigma))  # [B, H, W]
    gt = gauss.unsqueeze(1)  # [B, 1, H, W]
    scales = [64, 32, 16, 8, 4, 2]
    bottlenecks = [F.max_pool2d(gt, k, stride=k) for k in scales]
    return gt, bottlenecks


def build_gaussian_pseudo_gt(corr_train, sigma=4.0):
    """Gaussian pseudo-GT centered at teacher's argmin over corr_train [B,512,512]."""
    b, H, W = corr_train.shape
    peak_idx = torch.argmin(corr_train.reshape(b, -1), dim=1)  # [B]
    peak_y = (peak_idx // W).float()
    peak_x = (peak_idx % W).float()
    return _build_gauss_bump_and_pyramid(peak_x, peak_y, H, W, sigma, corr_train.device)


def build_gaussian_true_gt(sat_delta, sigma=4.0, H=512, W=512, device=None):
    """Gaussian true-GT centered at the VIGOR ground-truth pixel location.
    VIGOR convention (see evaluate): pixel_col = W/2 + sat_delta[:,0]*W/4,
    pixel_row = H/2 + sat_delta[:,1]*H/4 for the 512x512 sat crop.
    """
    peak_x = (W / 2.0) + sat_delta[:, 0].float() * (W / 4.0)
    peak_y = (H / 2.0) + sat_delta[:, 1].float() * (H / 4.0)
    return _build_gauss_bump_and_pyramid(peak_x, peak_y, H, W, sigma, device)


def _pearson(a, b, dim=-1, eps=1e-8):
    a_c = a - a.mean(dim=dim, keepdim=True)
    b_c = b - b.mean(dim=dim, keepdim=True)
    num = (a_c * b_c).sum(dim=dim)
    den = torch.sqrt((a_c * a_c).sum(dim=dim) * (b_c * b_c).sum(dim=dim)) + eps
    return num / den


def dist_loss_at_scale(scores, labels):
    """DIST (Huang et al., NeurIPS 2022) Pearson-correlation loss, spatial dense variant.
    Skips the inter (cross-batch) term because Gaussian labels have zero variance across
    the batch at most spatial positions \u2192 division by zero. Intra Pearson is still
    scale-invariant, avoiding the log(K) floor issue of KL/CE.
    """
    return 1.0 - _pearson(scores, labels, dim=-1).mean()


def ccvpe_inputs_from_batch(batch, device):
    """Derive CCVPE (grd, sat) ImageNet-normalized tensors from a geokd VIGOR batch."""
    sat = batch[1].to(device).float()           # [B, 3, 512, 512], 0-255
    resized_pano = batch[7].to(device).float()  # [B, 320, 640, 3], 0-255
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    sat_c = ((sat / 255.0) - mean) / std
    grd = resized_pano.permute(0, 3, 1, 2).contiguous()
    grd_c = ((grd / 255.0) - mean) / std
    return grd_c, sat_c


def teacher_forward(args, dino, teacher, batch, device):
    batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
    _, sat, _, _, _, sat_delta, meter_per_pixel, resized_pano, _, _ = batch
    sat_img = 2 * (sat / 255.0) - 1.0
    pano_img = 2 * (resized_pano / 255.0) - 1.0
    pano_img = pano_img.contiguous().permute(0, 3, 1, 2)
    sat_feat_list = dino(sat_img.contiguous())
    pano_feat_list = dino(pano_img)
    sat_feat, sat_conf, g2s_feat, g2s_conf, _, _ = teacher(sat_feat_list, pano_feat_list, meter_per_pixel)
    # eval-corr: raw cos_sim (higher=better) — for argmax localization
    corr_eval = teacher.calc_corr_for_val(sat_feat, sat_conf, g2s_feat, g2s_conf)  # [B, H, W]
    # train-corr: 2-2·cos_sim (lower=better) at teacher's native feat resolution, then
    # upsample to 512x512 so the loss runs at the same grid as the student's eval-path argmax.
    corr_train_native = teacher.calc_corr_for_train(sat_feat, g2s_feat, batch_wise=False)[args.levels[-1]]  # [B, H, W]
    corr_train = F.interpolate(
        corr_train_native.unsqueeze(1), size=(512, 512), mode="bilinear", align_corners=False
    )[:, 0]  # [B, 512, 512]
    feat_h = sat_feat[args.levels[-1]].shape[2]
    return corr_eval, corr_train, sat_delta, meter_per_pixel, feat_h, g2s_feat


def student_logits_to_corr(logits_flattened):
    """CCVPE raw 512x512 logits → 2-2·logit distance-convention corr map (no standardization)."""
    b = logits_flattened.shape[0]
    logit_map = logits_flattened.reshape(b, 512, 512)
    # Keep raw magnitude: standardization here strips the scale info that the 262144-way
    # softmax needs at eval time (peak_ratio collapses to 1.0). Downsampling similarly
    # disconnects the training grid from the eval-path 512x512 argmax.
    return 2.0 - 2.0 * logit_map


def build_teacher(args, device):
    teacher = build_localization_model(args, dataset="vigor", dino_model=args.teacher_dino_model, student_width=None).to(device)
    state, _ = load_checkpoint_state(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    dino = DINO(model_name=args.teacher_dino_model).to(device).eval()
    for p in dino.parameters():
        p.requires_grad = False
    return teacher, dino


def evaluate(args, dino, teacher, student, loader, device, max_batches=None):
    student.eval()
    s_pu, s_pv, t_pu, t_pv, g_u, g_v = ([] for _ in range(6))
    has_teacher = teacher is not None
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch_gpu = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            sat_delta = batch_gpu[5]
            mpp = batch_gpu[6]
            # GT offset from sat centre, in metres (geokd convention)
            gt_u = sat_delta[:, 0] * mpp * 512 / 4
            gt_v = sat_delta[:, 1] * mpp * 512 / 4
            b = sat_delta.shape[0]
            if has_teacher:
                corr_t, _corr_t_train, _, _, feat_h, _ = teacher_forward(args, dino, teacher, batch, device)
                _, corr_h, corr_w = corr_t.shape
                t_idx = torch.argmax(corr_t.reshape(b, -1), dim=1)
                t_u = (t_idx % corr_w - corr_w / 2) * 512 / feat_h * mpp
                t_v = (t_idx // corr_w - corr_h / 2) * 512 / feat_h * mpp
                t_pu.append(t_u.cpu().numpy())
                t_pv.append(t_v.cpu().numpy())
            # Student prediction: CCVPE heatmap argmax in its native 512x512 frame
            grd_c, sat_c = ccvpe_inputs_from_batch(batch, device)
            heatmap = student(grd_c, sat_c)[1][:, 0]  # [B, 512, 512]
            s_idx = torch.argmax(heatmap.reshape(b, -1), dim=1)
            s_u = (s_idx % 512 - 512 / 2) * mpp
            s_v = (s_idx // 512 - 512 / 2) * mpp
            g_u.append(gt_u.cpu().numpy())
            g_v.append(gt_v.cpu().numpy())
            s_pu.append(s_u.cpu().numpy())
            s_pv.append(s_v.cpu().numpy())
    student_metrics = xy_metrics(s_pu, s_pv, g_u, g_v, device)
    teacher_metrics = xy_metrics(t_pu, t_pv, g_u, g_v, device) if has_teacher else (float("nan"), float("nan"))
    return student_metrics, teacher_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset/config_vigor_kd.yaml")
    parser.add_argument("--distill_mode", default="ce", choices=["ce", "peak_hard", "freq", "hard_label", "ccvpe_native", "dist", "supervised_gt", "gt_plus_kd"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--name", default="ccvpe-kd")
    parser.add_argument("--circular_padding", action="store_true", default=True)
    parser.add_argument("--boundary_weight", type=float, default=0.0)
    parser.add_argument("--boundary_margin", type=float, default=1.0)
    parser.add_argument("--eval_only", action="store_true", help="only run teacher-sanity eval on a few val batches, then exit")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="path to a student checkpoint to warm-start from")
    parser.add_argument("--kd_weight", type=float, default=1.0, help="weight for the teacher soft-heatmap KL (gt_plus_kd only)")
    parser.add_argument("--kd_temp", type=float, default=4.0, help="softmax temperature for the Hinton-style teacher KL (gt_plus_kd only)")
    cli = parser.parse_args()

    device = torch.device("cuda:0")
    args = namespace_from_config(cli.config)
    args.batch_size = cli.batch_size
    args.batch_size_per_gpu = cli.batch_size
    args.distributed = False
    args.train = True

    teacher, dino = (None, None) if cli.distill_mode == "supervised_gt" else build_teacher(args, device)
    student = CVM_VIGOR(device, circular_padding=cli.circular_padding).to(device)
    print_colored(f"CCVPE student params: {sum(p.numel() for p in student.parameters()) / 1e6:.2f}M")
    if cli.init_checkpoint is not None:
        ck = torch.load(cli.init_checkpoint, map_location=device, weights_only=False)
        student.load_state_dict(ck["model_state_dict"])
        print_colored(f"warm-started from {cli.init_checkpoint} (epoch={ck.get('epoch', '?')}, loss={ck.get('loss', '?')})")

    train_loader, val_loader = fetch_dataloader(args, VIGOR(args, "train"))
    optimizer = torch.optim.Adam(student.parameters(), lr=cli.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cli.epochs, eta_min=cli.lr * 0.1)

    if cli.eval_only:
        (s_mean, s_median), (t_mean, t_median) = evaluate(args, dino, teacher, student, val_loader, device, max_batches=15)
        print(f"[eval_only] teacher_sanity mean={t_mean:.4f} median={t_median:.4f} (expect ~2.6) | "
              f"untrained_student mean={s_mean:.4f}", flush=True)
        return

    out_dir = os.path.join("checkpoints", "vigor", "ccvpe", cli.name)
    os.makedirs(out_dir, exist_ok=True)
    best = float("inf")
    levels = [args.levels[-1]]

    for epoch in range(cli.epochs):
        student.train()
        total = 0.0
        for i, batch in enumerate(train_loader):
            if cli.distill_mode == "supervised_gt":
                # Pure supervised training with VIGOR true GT; skip teacher_forward entirely.
                batch_gpu = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
                sat_delta = batch_gpu[5]
                gt, gt_bottlenecks = build_gaussian_true_gt(sat_delta, sigma=4.0, H=512, W=512, device=device)
                grd_c, sat_c = ccvpe_inputs_from_batch(batch, device)
                out = student(grd_c, sat_c)
                gt_flat = torch.flatten(gt, start_dim=1)
                gt_flat = gt_flat / gt_flat.sum(dim=1, keepdim=True).clamp_min(1e-8)
                loss_ce = ccvpe_ce_soft(out[0], gt_flat)
                loss_nce = 0.0
                for k, gt_bk in enumerate(gt_bottlenecks):
                    scores = torch.flatten(out[3 + k].max(dim=1)[0], start_dim=1)
                    labels = torch.flatten(gt_bk, start_dim=1)
                    loss_nce = loss_nce + ccvpe_infoNCE(scores, labels)
                loss = loss_ce + (loss_nce / 6.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                total += loss.item()
                if i % 100 == 0:
                    print(f"[{cli.name}] epoch {epoch} it {i}/{len(train_loader)} loss {loss.item():.4f}", flush=True)
                continue
            with torch.no_grad():
                _corr_t_eval, corr_t_train, _, _, feat_h, g2s_feat_t = teacher_forward(args, dino, teacher, batch, device)
            grd_c, sat_c = ccvpe_inputs_from_batch(batch, device)
            out = student(grd_c, sat_c)
            if cli.distill_mode == "gt_plus_kd":
                # Route B stage-2: true-GT primary supervision (supervised_gt recipe) +
                # Hinton-style KL against teacher's full soft heatmap as auxiliary.
                batch_gpu = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
                sat_delta = batch_gpu[5]
                gt, gt_bottlenecks = build_gaussian_true_gt(sat_delta, sigma=4.0, H=512, W=512, device=device)
                gt_flat = torch.flatten(gt, start_dim=1)
                gt_flat = gt_flat / gt_flat.sum(dim=1, keepdim=True).clamp_min(1e-8)
                loss_ce = ccvpe_ce_soft(out[0], gt_flat)
                loss_nce = 0.0
                for k, gt_bk in enumerate(gt_bottlenecks):
                    scores = torch.flatten(out[3 + k].max(dim=1)[0], start_dim=1)
                    labels = torch.flatten(gt_bk, start_dim=1)
                    loss_nce = loss_nce + ccvpe_infoNCE(scores, labels)
                # Teacher soft heatmap: corr_t_train is 2-2·cos_sim, so cos_sim = 1 - corr_t_train/2.
                # Higher cos_sim = better match \u2192 softmax over 262144 pixels gives the soft target.
                teacher_logits = (1.0 - corr_t_train / 2.0).reshape(corr_t_train.shape[0], -1)
                T = cli.kd_temp
                kd_kl = F.kl_div(
                    F.log_softmax(out[0] / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                loss = loss_ce + (loss_nce / 6.0) + cli.kd_weight * kd_kl
                if i % 100 == 0:
                    print(f"[{cli.name}] epoch {epoch} it {i}/{len(train_loader)} "
                          f"total={loss.item():.4f} ce={loss_ce.item():.4f} "
                          f"nce/6={(loss_nce/6.0).item():.4f} kd_kl={kd_kl.item():.4f} "
                          f"kd_weighted={(cli.kd_weight * kd_kl).item():.4f}", flush=True)
            elif cli.distill_mode == "hard_label":
                # teacher pseudo-GT: argmin over 2-2·cos_sim (lower = better match) at 512x512
                b = corr_t_train.shape[0]
                teacher_gt = torch.argmin(corr_t_train.reshape(b, -1), dim=1)  # [B]
                loss = F.cross_entropy(out[0], teacher_gt)
            elif cli.distill_mode == "ccvpe_native":
                # CCVPE's native training recipe (soft-CE + 6-scale soft-InfoNCE) with teacher-
                # derived Gaussian pseudo-GT. matching_score_stacked{k} has 20 orientation
                # channels; without pose distillation we reduce over them with max.
                gt, gt_bottlenecks = build_gaussian_pseudo_gt(corr_t_train, sigma=4.0)
                gt_flat = torch.flatten(gt, start_dim=1)
                gt_flat = gt_flat / gt_flat.sum(dim=1, keepdim=True).clamp_min(1e-8)
                loss_ce = ccvpe_ce_soft(out[0], gt_flat)
                # matching_score_stacked{1..6} = out[3..8]
                loss_nce = 0.0
                for k, gt_bk in enumerate(gt_bottlenecks):
                    scores_ori_max = out[3 + k].max(dim=1)[0]  # [B, H, W]
                    scores = torch.flatten(scores_ori_max, start_dim=1)
                    labels = torch.flatten(gt_bk, start_dim=1)  # [B, H*W]
                    loss_nce = loss_nce + ccvpe_infoNCE(scores, labels)
                loss = loss_ce + (loss_nce / 6.0)
            elif cli.distill_mode == "dist":
                # CCVPE multi-scale supervision with DIST Pearson-correlation loss replacing
                # InfoNCE (Huang et al., NeurIPS 2022). Scale-invariant, avoids the log(K)
                # floor problem that KL/CE hits when student logits start near-uniform.
                gt, gt_bottlenecks = build_gaussian_pseudo_gt(corr_t_train, sigma=4.0)
                gt_flat = torch.flatten(gt, start_dim=1)
                gt_flat = gt_flat / gt_flat.sum(dim=1, keepdim=True).clamp_min(1e-8)
                loss_ce = ccvpe_ce_soft(out[0], gt_flat)
                loss_dist = 0.0
                for k, gt_bk in enumerate(gt_bottlenecks):
                    scores_ori_max = out[3 + k].max(dim=1)[0]  # [B, H, W]
                    scores = torch.flatten(scores_ori_max, start_dim=1)
                    labels = torch.flatten(gt_bk, start_dim=1)
                    loss_dist = loss_dist + dist_loss_at_scale(scores, labels)
                loss = loss_ce + (loss_dist / 6.0)
            else:
                corr_s = student_logits_to_corr(out[0])  # [B, 512, 512]
                loss = cross_entropy(
                    {levels[0]: corr_s}, {levels[0]: corr_t_train}, levels,
                    s_temp=args.student_temp, t_temp=args.teacher_temp, distill_mode=cli.distill_mode,
                )
            if cli.boundary_weight > 0:
                # out[8] = matching_score_stacked6: finest-resolution 20-channel windowed
                # cosine-similarity volume (pre max-pool). Channel-pooled attention is
                # architecture-agnostic, so this substitutes for CCVPE's g2s_feat.
                boundary_loss = boundary_distillation(
                    {"g2s_feat": {levels[0]: out[8]}},
                    {"g2s_feat": {levels[0]: g2s_feat_t[levels[0]]}},
                    levels, margin=cli.boundary_margin, feature_keys=("g2s_feat",),
                )
                loss = loss + cli.boundary_weight * boundary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
            if i % 100 == 0:
                print(f"[{cli.name}] epoch {epoch} it {i}/{len(train_loader)} loss {loss.item():.4f}", flush=True)
        scheduler.step()
        (mean_err, median_err), (t_mean, t_median) = evaluate(args, dino, teacher, student, val_loader, device)
        print(f"Epoch {epoch + 1}/{cli.epochs}: train_loss={total / max(len(train_loader),1):.4f}, "
              f"mean={mean_err:.4f}, median={median_err:.4f} | teacher_sanity_mean={t_mean:.4f} (expect ~2.6)", flush=True)
        torch.save({"epoch": epoch, "model_state_dict": student.state_dict(), "loss": mean_err,
                    "distill_mode": cli.distill_mode, "name": cli.name}, os.path.join(out_dir, "last.pth"))
        if mean_err < best:
            best = mean_err
            torch.save({"epoch": epoch, "model_state_dict": student.state_dict(), "loss": mean_err,
                        "distill_mode": cli.distill_mode, "name": cli.name}, os.path.join(out_dir, "best.pth"))
    print(f"[{cli.name}] done. best mean={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
