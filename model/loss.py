import math

import torch
import torch.nn.functional as F
import numpy as np



def cross_entropy(
    pred_dict,
    target_dict,
    levels,
    s_temp=0.2,
    t_temp=0.09,
    distill_mode="ce",
    uncertainty_weighting=False,
    uncertainty_weight_min=0.25,
    uncertainty_weight_power=1.0,
    peak_radius=2,
    peak_topk=1,
    peak_region_weight=4.0,
    hard_negative_topk=32,
    hard_negative_weight=1.0,
    freq_cutoff=0.25,
    freq_low_weight=1.0,
    freq_high_weight=1.0,
    freq_ce_weight=1.0,
    return_stats=False,
):
    if distill_mode == "peak_hard":
        return peak_hard_negative_distillation(
            pred_dict,
            target_dict,
            levels,
            s_temp=s_temp,
            t_temp=t_temp,
            peak_radius=peak_radius,
            peak_topk=peak_topk,
            peak_region_weight=peak_region_weight,
            hard_negative_topk=hard_negative_topk,
            hard_negative_weight=hard_negative_weight,
            return_stats=return_stats,
        )
    if distill_mode == "freq":
        return frequency_aligned_distillation(
            pred_dict,
            target_dict,
            levels,
            s_temp=s_temp,
            t_temp=t_temp,
            freq_cutoff=freq_cutoff,
            low_freq_weight=freq_low_weight,
            high_freq_weight=freq_high_weight,
            ce_weight=freq_ce_weight,
            return_stats=return_stats,
        )
    if distill_mode == "dist":
        return dist_pearson_intra_distillation(
            pred_dict,
            target_dict,
            levels,
            return_stats=return_stats,
        )
    if distill_mode == "mse":
        return mse_distillation(
            pred_dict,
            target_dict,
            levels,
            return_stats=return_stats,
        )
    if distill_mode == "dkd":
        return dkd_distillation(
            pred_dict,
            target_dict,
            levels,
            s_temp=s_temp,
            t_temp=t_temp,
            peak_radius=peak_radius,
            peak_topk=peak_topk,
            alpha=peak_region_weight,       # reused as TCKD weight (α)
            beta=hard_negative_weight,      # reused as NCKD weight (β)
            return_stats=return_stats,
        )
    if distill_mode == "peak_hard_ce":
        return peak_hard_plus_ce_distillation(
            pred_dict,
            target_dict,
            levels,
            s_temp=s_temp,
            t_temp=t_temp,
            peak_radius=peak_radius,
            peak_topk=peak_topk,
            peak_region_weight=peak_region_weight,
            hard_negative_topk=hard_negative_topk,
            hard_negative_weight=hard_negative_weight,
            ce_weight=freq_ce_weight,       # reuse freq_ce_weight for the CE blending weight
            return_stats=return_stats,
        )
    if distill_mode == "peak_hard_zscore":
        return peak_hard_negative_distillation(
            pred_dict,
            target_dict,
            levels,
            s_temp=s_temp,
            t_temp=t_temp,
            peak_radius=peak_radius,
            peak_topk=peak_topk,
            peak_region_weight=peak_region_weight,
            hard_negative_topk=hard_negative_topk,
            hard_negative_weight=hard_negative_weight,
            zscore=True,
            return_stats=return_stats,
        )
    if distill_mode == "ce_zscore":
        return zscore_ce_distillation(
            pred_dict,
            target_dict,
            levels,
            s_temp=s_temp,
            t_temp=t_temp,
            return_stats=return_stats,
        )
    if distill_mode == "wavelet":
        return wavelet_distillation(
            pred_dict,
            target_dict,
            levels,
            s_temp=s_temp,
            t_temp=t_temp,
            low_weight=freq_low_weight,
            high_weight=freq_high_weight,
            ce_weight=freq_ce_weight,
            return_stats=return_stats,
        )
    if distill_mode != "ce":
        raise ValueError(f"Unsupported distill_mode '{distill_mode}'.")

    ce_losses = []
    entropy_values = []
    confidence_values = []
    weight_values = []
    for _, level in enumerate(levels):
        pred = pred_dict[level]
        target = target_dict[level]
        pred = -(pred - 2) / 2
        target = -(target - 2) / 2

        b, h, w = pred.shape

        pred_map_flat = pred.reshape(b, -1)
        target_map_flat = target.reshape(b, -1)

        pred_map_log_softmax = F.log_softmax(pred_map_flat / s_temp, dim=1)
        target_map_softmax = F.softmax(target_map_flat / t_temp, dim=1)

        per_sample_loss = -torch.sum(target_map_softmax * pred_map_log_softmax, dim=1)
        if uncertainty_weighting:
            entropy = -torch.sum(target_map_softmax * torch.log(target_map_softmax.clamp_min(1e-12)), dim=1)
            norm_entropy = entropy / math.log(target_map_softmax.shape[1])
            confidence = (1.0 - norm_entropy).clamp(0.0, 1.0)
            weight = uncertainty_weight_min + (1.0 - uncertainty_weight_min) * confidence.pow(uncertainty_weight_power)
            loss = torch.sum(per_sample_loss * weight) / weight.sum().clamp_min(1e-6)

            entropy_values.append(norm_entropy.detach().mean())
            confidence_values.append(confidence.detach().mean())
            weight_values.append(weight.detach().mean())
        else:
            loss = torch.sum(per_sample_loss) / b
        ce_losses.append(loss)

    loss = torch.mean(torch.stack(ce_losses, dim=0).float())
    if not return_stats:
        return loss

    stats = {}
    if uncertainty_weighting:
        stats = {
            "distill/teacher_entropy": torch.mean(torch.stack(entropy_values)).item(),
            "distill/teacher_confidence": torch.mean(torch.stack(confidence_values)).item(),
            "distill/uncertainty_weight": torch.mean(torch.stack(weight_values)).item(),
        }
    return loss, stats


def _corr_to_logits(corr):
    return -(corr - 2) / 2


def _radial_freq_masks(h, w, cutoff, device):
    """Split the rfft2 spectrum into low/high frequency bands by normalized radius."""
    fy = torch.fft.fftfreq(h, device=device).abs().view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=device).view(1, -1)
    radius = torch.sqrt(fy ** 2 + fx ** 2)
    low_mask = (radius <= float(cutoff)).float()
    high_mask = 1.0 - low_mask
    return low_mask, high_mask


def frequency_aligned_distillation(
    pred_dict,
    target_dict,
    levels,
    s_temp=0.2,
    t_temp=0.09,
    freq_cutoff=0.25,
    low_freq_weight=1.0,
    high_freq_weight=1.0,
    ce_weight=1.0,
    return_stats=False,
):
    """SDKD-style distillation: response CE plus low/high frequency spectral alignment."""
    level_losses = []
    ce_values = []
    low_values = []
    high_values = []

    for _, level in enumerate(levels):
        pred = _corr_to_logits(pred_dict[level])
        target = _corr_to_logits(target_dict[level])

        b, h, w = pred.shape
        pred_flat = pred.reshape(b, -1)
        target_flat = target.reshape(b, -1)
        student_log_prob = F.log_softmax(pred_flat / s_temp, dim=1)
        teacher_prob = F.softmax(target_flat / t_temp, dim=1)
        ce = -torch.sum(teacher_prob * student_log_prob, dim=1).mean()

        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        spectral_diff = torch.abs(pred_fft - target_fft) ** 2

        low_mask, high_mask = _radial_freq_masks(h, w, freq_cutoff, pred.device)
        low_mask = low_mask.unsqueeze(0)
        high_mask = high_mask.unsqueeze(0)
        low_loss = (spectral_diff * low_mask).sum(dim=(1, 2)) / low_mask.sum().clamp_min(1.0)
        high_loss = (spectral_diff * high_mask).sum(dim=(1, 2)) / high_mask.sum().clamp_min(1.0)
        low_loss = low_loss.mean()
        high_loss = high_loss.mean()

        level_losses.append(
            float(ce_weight) * ce
            + float(low_freq_weight) * low_loss
            + float(high_freq_weight) * high_loss
        )
        ce_values.append(ce.detach())
        low_values.append(low_loss.detach())
        high_values.append(high_loss.detach())

    loss = torch.mean(torch.stack(level_losses, dim=0).float())
    if not return_stats:
        return loss

    stats = {
        "distill/freq_ce": torch.mean(torch.stack(ce_values)).item(),
        "distill/freq_low": torch.mean(torch.stack(low_values)).item(),
        "distill/freq_high": torch.mean(torch.stack(high_values)).item(),
    }
    return loss, stats


def _topk_region_mask(logits, h, w, topk=1, radius=2):
    b = logits.shape[0]
    numel = logits.shape[1]
    k = min(max(int(topk), 1), numel)
    indices = torch.topk(logits.detach(), k=k, dim=1).indices
    mask = torch.zeros((b, numel), dtype=torch.bool, device=logits.device)
    mask.scatter_(1, indices, True)
    mask = mask.reshape(b, 1, h, w).float()
    if radius > 0:
        kernel_size = int(radius) * 2 + 1
        mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=int(radius))
    return mask.reshape(b, -1).bool()


def _student_hard_negative_mask(student_logits, valid_negative_mask, topk):
    b, numel = student_logits.shape
    k = min(max(int(topk), 1), numel)
    masked_logits = student_logits.detach().masked_fill(~valid_negative_mask, float("-inf"))
    has_valid = torch.isfinite(masked_logits).any(dim=1)
    safe_logits = torch.where(torch.isfinite(masked_logits), masked_logits, torch.full_like(masked_logits, -1e4))
    indices = torch.topk(safe_logits, k=k, dim=1).indices
    mask = torch.zeros((b, numel), dtype=torch.bool, device=student_logits.device)
    mask.scatter_(1, indices, True)
    return mask & valid_negative_mask & has_valid[:, None]


def _zscore(x, eps=1e-7):
    """Per-sample Z-score standardization along last dim (Sun et al., CVPR 2024)."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def peak_hard_negative_distillation(
    pred_dict,
    target_dict,
    levels,
    s_temp=0.2,
    t_temp=0.09,
    peak_radius=2,
    peak_topk=1,
    peak_region_weight=4.0,
    hard_negative_topk=32,
    hard_negative_weight=1.0,
    zscore=False,
    return_stats=False,
):
    losses = []
    peak_ce_values = []
    hard_loss_values = []
    teacher_peak_mass_values = []
    student_hard_mass_values = []
    peak_fraction_values = []

    for _, level in enumerate(levels):
        pred = _corr_to_logits(pred_dict[level])
        target = _corr_to_logits(target_dict[level])

        b, h, w = pred.shape
        pred_flat = pred.reshape(b, -1)
        target_flat = target.reshape(b, -1)

        if zscore:
            pred_scaled = _zscore(pred_flat) / s_temp
            target_scaled = _zscore(target_flat) / t_temp
        else:
            pred_scaled = pred_flat / s_temp
            target_scaled = target_flat / t_temp

        student_log_prob = F.log_softmax(pred_scaled, dim=1)
        student_prob = student_log_prob.exp()
        teacher_prob = F.softmax(target_scaled, dim=1)

        peak_mask = _topk_region_mask(target_flat, h, w, topk=peak_topk, radius=peak_radius)
        token_weight = torch.ones_like(teacher_prob)
        token_weight = token_weight + (float(peak_region_weight) - 1.0) * peak_mask.float()
        weighted_teacher = teacher_prob * token_weight
        peak_ce = -torch.sum(weighted_teacher * student_log_prob, dim=1)
        peak_ce = peak_ce / weighted_teacher.sum(dim=1).clamp_min(1e-6)
        peak_ce = peak_ce.mean()

        valid_negative_mask = ~peak_mask
        hard_mask = _student_hard_negative_mask(pred_flat, valid_negative_mask, hard_negative_topk)
        hard_mass = torch.sum(student_prob * hard_mask.float(), dim=1)
        hard_loss = hard_mass.mean()

        loss = peak_ce + float(hard_negative_weight) * hard_loss
        losses.append(loss)

        peak_ce_values.append(peak_ce.detach())
        hard_loss_values.append(hard_loss.detach())
        teacher_peak_mass_values.append(torch.sum(teacher_prob.detach() * peak_mask.float(), dim=1).mean())
        student_hard_mass_values.append(hard_mass.detach().mean())
        peak_fraction_values.append(peak_mask.float().mean())

    loss = torch.mean(torch.stack(losses, dim=0).float())
    if not return_stats:
        return loss

    stats = {
        "distill/peak_ce": torch.mean(torch.stack(peak_ce_values)).item(),
        "distill/hard_negative_loss": torch.mean(torch.stack(hard_loss_values)).item(),
        "distill/teacher_peak_mass": torch.mean(torch.stack(teacher_peak_mass_values)).item(),
        "distill/student_hard_mass": torch.mean(torch.stack(student_hard_mass_values)).item(),
        "distill/peak_region_fraction": torch.mean(torch.stack(peak_fraction_values)).item(),
    }
    return loss, stats


def _channel_attention(feat, eps=1e-6):
    """Channel-pooled, per-sample standardized spatial attention map [B, H, W]."""
    att = feat.pow(2).sum(dim=1)
    b = att.shape[0]
    flat = att.reshape(b, -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp_min(eps)
    flat = (flat - mean) / std
    return flat.reshape_as(att)


def boundary_distillation(
    student_feats,
    teacher_feats,
    levels,
    margin=1.0,
    feature_keys=("sat_feat", "g2s_feat"),
    return_stats=False,
):
    """GaitKD-style activation-boundary distillation on channel-pooled attention maps.

    Preserves the teacher-induced active/inactive spatial partitioning instead of
    directly regressing features, so it is agnostic to teacher/student channel widths.
    """
    losses = []
    active_fracs = []

    for _, level in enumerate(levels):
        for key in feature_keys:
            student_feat = student_feats[key][level]
            teacher_feat = teacher_feats[key][level].detach()

            att_s = _channel_attention(student_feat)
            att_t = _channel_attention(teacher_feat)

            if att_s.shape[-2:] != att_t.shape[-2:]:
                att_s = F.interpolate(
                    att_s.unsqueeze(1), size=att_t.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)

            active = (att_t > 0).float()
            pos = active * F.relu(float(margin) - att_s).pow(2)
            neg = (1.0 - active) * F.relu(float(margin) + att_s).pow(2)
            losses.append((pos + neg).mean())
            active_fracs.append(active.mean().detach())

    loss = torch.mean(torch.stack(losses, dim=0).float())
    if not return_stats:
        return loss

    stats = {
        "distill/boundary_loss": loss.item(),
        "distill/boundary_active_frac": torch.mean(torch.stack(active_fracs)).item(),
    }
    return loss, stats


def dist_pearson_intra_distillation(
    pred_dict,
    target_dict,
    levels,
    return_stats=False,
):
    """DIST-style intra-image Pearson correlation on raw cos-sim logits."""
    losses = []
    pearson_values = []
    for level in levels:
        pred = _corr_to_logits(pred_dict[level])
        target = _corr_to_logits(target_dict[level])
        b = pred.shape[0]
        p = pred.reshape(b, -1)
        t = target.reshape(b, -1)
        p_center = p - p.mean(dim=1, keepdim=True)
        t_center = t - t.mean(dim=1, keepdim=True)
        num = (p_center * t_center).sum(dim=1)
        den = (p_center.norm(dim=1) * t_center.norm(dim=1)).clamp_min(1e-6)
        pearson = num / den
        losses.append((1.0 - pearson).mean())
        pearson_values.append(pearson.detach().mean())

    loss = torch.mean(torch.stack(losses, dim=0).float())
    if not return_stats:
        return loss
    stats = {
        "distill/dist_pearson": torch.mean(torch.stack(pearson_values)).item(),
    }
    return loss, stats


def mse_distillation(
    pred_dict,
    target_dict,
    levels,
    return_stats=False,
):
    """Plain L2 on raw cos-sim logits, no softmax or temperature."""
    losses = []
    for level in levels:
        pred = _corr_to_logits(pred_dict[level])
        target = _corr_to_logits(target_dict[level])
        losses.append(F.mse_loss(pred, target))
    loss = torch.mean(torch.stack(losses, dim=0).float())
    if not return_stats:
        return loss
    stats = {"distill/mse": loss.item()}
    return loss, stats


def dkd_distillation(
    pred_dict,
    target_dict,
    levels,
    s_temp=0.06,
    t_temp=0.06,
    peak_radius=2,
    peak_topk=1,
    alpha=1.0,
    beta=8.0,
    return_stats=False,
):
    """DKD adapted to heatmap: TCKD on binary [peak, off] + NCKD on non-peak renormalized."""
    losses = []
    tckd_vals = []
    nckd_vals = []
    for level in levels:
        pred = _corr_to_logits(pred_dict[level])
        target = _corr_to_logits(target_dict[level])
        b, h, w = pred.shape
        p = pred.reshape(b, -1)
        t = target.reshape(b, -1)
        s_prob = F.softmax(p / s_temp, dim=1)
        t_prob = F.softmax(t / t_temp, dim=1)

        peak_mask = _topk_region_mask(t, h, w, topk=peak_topk, radius=peak_radius).float()

        s_peak = (s_prob * peak_mask).sum(dim=1).clamp_min(1e-8)
        s_off = (1.0 - s_peak).clamp_min(1e-8)
        t_peak = (t_prob * peak_mask).sum(dim=1).clamp_min(1e-8)
        t_off = (1.0 - t_peak).clamp_min(1e-8)
        tckd = (t_peak * (torch.log(t_peak) - torch.log(s_peak)) +
                t_off * (torch.log(t_off) - torch.log(s_off))).mean()
        tckd = tckd * (t_temp * t_temp)

        neg_mask = 1.0 - peak_mask
        s_neg = s_prob * neg_mask
        t_neg = t_prob * neg_mask
        s_neg_norm = s_neg / s_neg.sum(dim=1, keepdim=True).clamp_min(1e-8)
        t_neg_norm = t_neg / t_neg.sum(dim=1, keepdim=True).clamp_min(1e-8)
        nckd = (t_neg_norm * (torch.log(t_neg_norm.clamp_min(1e-8)) - torch.log(s_neg_norm.clamp_min(1e-8))))
        nckd = nckd.sum(dim=1).mean() * (t_temp * t_temp)

        losses.append(float(alpha) * tckd + float(beta) * nckd)
        tckd_vals.append(tckd.detach())
        nckd_vals.append(nckd.detach())

    loss = torch.mean(torch.stack(losses, dim=0).float())
    if not return_stats:
        return loss
    stats = {
        "distill/dkd_tckd": torch.mean(torch.stack(tckd_vals)).item(),
        "distill/dkd_nckd": torch.mean(torch.stack(nckd_vals)).item(),
    }
    return loss, stats


def peak_hard_plus_ce_distillation(
    pred_dict,
    target_dict,
    levels,
    s_temp=0.06,
    t_temp=0.06,
    peak_radius=2,
    peak_topk=1,
    peak_region_weight=4.0,
    hard_negative_topk=32,
    hard_negative_weight=1.0,
    ce_weight=1.0,
    return_stats=False,
):
    """peak_hard (best current mode) blended with plain Hinton soft-CE on full map."""
    peak_loss, peak_stats = peak_hard_negative_distillation(
        pred_dict,
        target_dict,
        levels,
        s_temp=s_temp,
        t_temp=t_temp,
        peak_radius=peak_radius,
        peak_topk=peak_topk,
        peak_region_weight=peak_region_weight,
        hard_negative_topk=hard_negative_topk,
        hard_negative_weight=hard_negative_weight,
        return_stats=True,
    )

    ce_losses = []
    for level in levels:
        pred = _corr_to_logits(pred_dict[level])
        target = _corr_to_logits(target_dict[level])
        b = pred.shape[0]
        p = pred.reshape(b, -1)
        t = target.reshape(b, -1)
        s_log = F.log_softmax(p / s_temp, dim=1)
        t_prob = F.softmax(t / t_temp, dim=1)
        ce_losses.append(-(t_prob * s_log).sum(dim=1).mean())
    ce = torch.mean(torch.stack(ce_losses, dim=0).float())

    loss = peak_loss + float(ce_weight) * ce
    if not return_stats:
        return loss
    peak_stats["distill/blend_ce"] = ce.item()
    peak_stats["distill/blend_ce_weight"] = float(ce_weight)
    return loss, peak_stats


def zscore_ce_distillation(
    pred_dict,
    target_dict,
    levels,
    s_temp=2.0,
    t_temp=2.0,
    return_stats=False,
):
    """Plain Hinton soft-CE with Z-score standardized logits (Sun et al., CVPR 2024)."""
    losses = []
    for level in levels:
        pred = _corr_to_logits(pred_dict[level])
        target = _corr_to_logits(target_dict[level])
        b = pred.shape[0]
        p = pred.reshape(b, -1)
        t = target.reshape(b, -1)
        p_z = _zscore(p) / s_temp
        t_z = _zscore(t) / t_temp
        s_log = F.log_softmax(p_z, dim=1)
        t_prob = F.softmax(t_z, dim=1)
        losses.append(-(t_prob * s_log).sum(dim=1).mean())
    loss = torch.mean(torch.stack(losses, dim=0).float())
    if not return_stats:
        return loss
    return loss, {"distill/ce_zscore": loss.item()}


def _haar_dwt2d(x):
    """One-level 2D Haar DWT along the last two dims.

    x: [B, H, W] with H and W even. Returns four sub-bands [B, H/2, W/2]:
    LL (low-pass), LH (horizontal detail), HL (vertical detail), HH (diagonal detail).
    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a + b - c - d) * 0.5
    hl = (a - b + c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    if squeeze:
        ll, lh, hl, hh = [t.squeeze(1) for t in (ll, lh, hl, hh)]
    return ll, lh, hl, hh


def wavelet_distillation(
    pred_dict,
    target_dict,
    levels,
    s_temp=0.2,
    t_temp=0.09,
    low_weight=1.0,
    high_weight=1.0,
    ce_weight=1.0,
    return_stats=False,
):
    """Simplified DS²D²-inspired wavelet decoupling: Haar DWT split LL/{LH,HL,HH} + spatial CE.

    Args:
        low_weight  : weight on L2 loss over LL band (structural).
        high_weight : weight on L2 loss over LH+HL+HH bands (edges / details).
        ce_weight   : weight on standard softmax CE (spatial supervision, same as freq mode).
    """
    losses = []
    low_values = []
    high_values = []
    ce_values = []
    for level in levels:
        pred = _corr_to_logits(pred_dict[level])
        target = _corr_to_logits(target_dict[level])
        b, h, w = pred.shape

        # need even H, W for Haar; if odd, drop last row/col
        if h % 2 == 1:
            pred = pred[..., :-1, :]
            target = target[..., :-1, :]
        if w % 2 == 1:
            pred = pred[..., :, :-1]
            target = target[..., :, :-1]

        p_ll, p_lh, p_hl, p_hh = _haar_dwt2d(pred)
        t_ll, t_lh, t_hl, t_hh = _haar_dwt2d(target)

        low = F.mse_loss(p_ll, t_ll)
        high = (F.mse_loss(p_lh, t_lh) + F.mse_loss(p_hl, t_hl) + F.mse_loss(p_hh, t_hh)) / 3.0

        p_flat = pred_dict[level].reshape(b, -1)
        t_flat = target_dict[level].reshape(b, -1)
        p_flat = _corr_to_logits(p_flat)
        t_flat = _corr_to_logits(t_flat)
        s_log = F.log_softmax(p_flat / s_temp, dim=1)
        t_prob = F.softmax(t_flat / t_temp, dim=1)
        ce = -(t_prob * s_log).sum(dim=1).mean()

        losses.append(float(ce_weight) * ce + float(low_weight) * low + float(high_weight) * high)
        low_values.append(low.detach())
        high_values.append(high.detach())
        ce_values.append(ce.detach())

    loss = torch.mean(torch.stack(losses, dim=0).float())
    if not return_stats:
        return loss
    stats = {
        "distill/wavelet_low": torch.mean(torch.stack(low_values)).item(),
        "distill/wavelet_high": torch.mean(torch.stack(high_values)).item(),
        "distill/wavelet_ce": torch.mean(torch.stack(ce_values)).item(),
    }
    return loss, stats


def multi_scale_contrastive_loss(corr_maps, levels):
    '''
    corr_maps: dict, key -- level; value -- corr map with shape of [M, N, H, W]
    '''
    matching_losses = []

    for _, level in enumerate(levels):
        corr = corr_maps[level]
        M, N, H, W = corr.shape
        assert M == N
        dis = torch.min(corr.reshape(M, N, -1), dim=-1)[0]
        pos = torch.diagonal(dis) # [M]  # it is also the predicted distance
        pos_neg = pos.reshape(-1, 1) - dis
        loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (M * (N-1))
        matching_losses.append(loss)

    return torch.mean(torch.stack(matching_losses, dim=0))


def kitti_localization_ce_loss(
    model_ref,
    sat_feat_dict,
    g2s_feat_dict,
    gt_shift_u,
    gt_shift_v,
    levels,
    shift_range_lon,
    shift_range_lat,
    meters_per_pixel_table,
    sigma_px=2.0,
    temperature=10.0,
):
    '''
    Spatial cross-entropy between softmax(-tau * corr) and a Gaussian target
    centred at the GT shift on the self-pair correlation map. Encourages the
    corr peak to land at (gt_shift_u, gt_shift_v).

    corr uses model_ref.calc_corr_for_train(batch_wise=False), i.e.
    corr = 2 - 2*cos_sim so LOWER is better; we negate before softmax.

    Args:
        model_ref: unwrapped KittiLocalizationNet.
        sat_feat_dict, g2s_feat_dict: dicts of features by level.
        gt_shift_u, gt_shift_v: [B, 1] in normalized [-1, 1].
        levels: list of ints to use.
        shift_range_lon, shift_range_lat: floats, meters.
        meters_per_pixel_table: dict level -> meter_per_pixel of the sat feature.
        sigma_px: Gaussian sigma in corr-map pixels for the soft target.
        temperature: softmax sharpness on the negated corr.
    '''
    losses = []
    stats = {}
    corr_maps = model_ref.calc_corr_for_train(
        sat_feat_dict,
        g2s_feat_dict,
        batch_wise=False,
    )
    for level in levels:
        corr = corr_maps[level]  # [B, H, W], lower = better
        B, H, W = corr.shape
        device = corr.device
        dtype = corr.dtype

        mpp = float(meters_per_pixel_table[level])
        gt_u_m = gt_shift_u.view(-1) * shift_range_lon
        gt_v_m = gt_shift_v.view(-1) * shift_range_lat
        gt_col = gt_u_m / mpp + W / 2.0 - 0.5
        gt_row = -gt_v_m / mpp + H / 2.0 - 0.5

        ii = torch.arange(H, device=device, dtype=dtype)
        jj = torch.arange(W, device=device, dtype=dtype)
        gi, gj = torch.meshgrid(ii, jj, indexing="ij")  # [H, W]
        d2 = (gi[None] - gt_row[:, None, None]) ** 2 + (gj[None] - gt_col[:, None, None]) ** 2
        target_logits = -d2 / (2.0 * float(sigma_px) ** 2)
        target = F.softmax(target_logits.view(B, -1), dim=-1)

        pred_logits = -corr.view(B, -1) * float(temperature)
        log_pred = F.log_softmax(pred_logits, dim=-1)

        ce = -(target * log_pred).sum(dim=-1).mean()
        losses.append(ce)
        stats[f"loc_ce_l{level}"] = float(ce.detach().item())

    total = torch.mean(torch.stack(losses, dim=0))
    stats["loc_ce"] = float(total.detach().item())
    return total, stats


def calculate_errors(true_labels, logits, num_classes=45):
    """
    Calculate the average error and median error
    true_labels: true label tensor, shape is [batch_size]
    predicted_labels: predicted label tensor, shape is [batch_size]
    """
    predicted_labels = torch.argmax(logits, dim=1)
    predicted_labels -= num_classes
    errors = torch.abs(true_labels - predicted_labels)

    mean_error = errors.float().mean()

    median_error = torch.median(errors.float())

    return errors, mean_error, median_error

def generate_soft_labels(true_labels, num_classes, sigma=1.0):
    """
    Generate soft labels based on Gaussian distribution, support batch labels
    true_labels: true labels, should be an integer array (such as [20, 21, 19, ...])
    num_classes: number of categories, assumed to be 90
    sigma: standard deviation of Gaussian distribution, controls the smoothness of soft labels
    """
    batch_size = true_labels.size(0)

    class_indices = torch.arange(num_classes, device=true_labels.device)  # [0, 1, ..., 89]

    soft_labels = []

    for i in range(batch_size):
        true_label = true_labels[i]
        distances = torch.abs(class_indices - true_label)

        soft_label = torch.exp(-distances ** 2 / (2 * sigma ** 2))

        soft_label /= soft_label.sum()

        soft_labels.append(soft_label)

    return torch.stack(soft_labels)

