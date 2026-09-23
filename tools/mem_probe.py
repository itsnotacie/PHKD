"""One-off memory probe: sweep batch sizes, report peak GPU memory for one training step.

Runs the real VIGOR train forward/backward (teacher + student + distill loss) on a single GPU
so we can pick batch_size_per_gpu that lands at ~80-90% of the card before launching DDP.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from train_geokd import (
    namespace_from_config,
    validate_args,
    load_teachers,
    build_student,
    compute_teacher_weights,
    forward_vigor,
    forward_kitti,
    make_backbone,
)
from model.dino import DINO
from model.loss import cross_entropy, boundary_distillation, multi_scale_contrastive_loss


def build(args, device):
    teachers, states, names = load_teachers(args, device) if args.supervision == "distill" or args.init_from_teacher else ([], [], [])
    teacher_state = states[0] if states else None
    student, _ = build_student(args, teacher_state, device)
    student_dino = make_backbone(args.student_dino_model).to(device).eval()
    for p in student_dino.parameters():
        p.requires_grad = False
    teacher_dinos = []
    cache = {}
    for name in names:
        if name not in cache:
            d = make_backbone(name).to(device).eval()
            for p in d.parameters():
                p.requires_grad = False
            cache[name] = d
        teacher_dinos.append(cache[name])
    weights = compute_teacher_weights(args, len(teachers))
    return teachers, teacher_dinos, weights, student, student_dino


def one_step(args, batch, device, teachers, teacher_dinos, weights, student, student_dino, optimizer):
    optimizer.zero_grad(set_to_none=True)
    fwd = forward_vigor if args.dataset == "vigor" else forward_kitti
    student_out = fwd(args, student_dino, student, batch, device)
    if args.supervision == "supervised":
        student_corr = student.calc_corr_for_train(student_out["sat_feat"], student_out["g2s_feat"], batch_wise=True)
        loss = multi_scale_contrastive_loss(student_corr, args.levels)
    else:
        teacher_corrs = []
        primary = None
        with torch.no_grad():
            for i, (t, td) in enumerate(zip(teachers, teacher_dinos)):
                t_out = fwd(args, td, t, batch, device)
                teacher_corrs.append(t.calc_corr_for_train(t_out["sat_feat"], t_out["g2s_feat"], batch_wise=False))
                if i == 0:
                    primary = t_out
        student_corr = student.calc_corr_for_train(student_out["sat_feat"], student_out["g2s_feat"], batch_wise=False)
        loss = 0.0
        for tw, tc in zip(weights, teacher_corrs):
            loss = loss + tw * cross_entropy(
                student_corr, tc, args.levels,
                s_temp=args.student_temp, t_temp=args.teacher_temp, distill_mode=args.distill_mode,
                uncertainty_weighting=args.uncertainty_weighting, uncertainty_weight_min=args.uncertainty_weight_min,
                uncertainty_weight_power=args.uncertainty_weight_power, peak_radius=args.peak_radius,
                peak_topk=args.peak_topk, peak_region_weight=args.peak_region_weight,
                hard_negative_topk=args.hard_negative_topk, hard_negative_weight=args.hard_negative_weight,
                freq_cutoff=args.freq_cutoff, freq_low_weight=args.freq_low_weight,
                freq_high_weight=args.freq_high_weight, freq_ce_weight=args.freq_ce_weight,
            )
        if args.boundary_weight > 0:
            loss = loss + args.boundary_weight * boundary_distillation(student_out, primary, args.levels, margin=args.boundary_margin)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=args.clip)
    optimizer.step()
    return float(loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batches", default="8,16,24,32", help="comma-separated batch sizes to try")
    parser.add_argument("--steps", type=int, default=3, help="steps per batch size (last is measured)")
    cli = parser.parse_args()

    device = torch.device("cuda:0")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Device: {torch.cuda.get_device_name(0)}  total={total_mem:.1f} GB")

    for bs in [int(x) for x in cli.batches.split(",")]:
        args = namespace_from_config(cli.config)
        args.distributed = False
        args.wandb = False
        args.train = True
        args.batch_size = bs
        args.batch_size_per_gpu = bs
        args.num_workers = 8
        validate_args(args)

        try:
            teachers, teacher_dinos, weights, student, student_dino = build(args, device)
            optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            from dataset.VIGOR import VIGOR, fetch_dataloader
            if args.dataset == "vigor":
                train_loader, _ = fetch_dataloader(args, VIGOR(args, "train"))
            else:
                raise SystemExit("mem_probe currently wired for vigor")

            torch.cuda.reset_peak_memory_stats(device)
            loss_val = None
            it = iter(train_loader)
            for step in range(cli.steps):
                batch = next(it)
                loss_val = one_step(args, batch, device, teachers, teacher_dinos, weights, student, student_dino, optimizer)
            torch.cuda.synchronize(device)
            peak_alloc = torch.cuda.max_memory_allocated(device) / 1e9
            peak_resv = torch.cuda.max_memory_reserved(device) / 1e9
            print(f"batch={bs:3d}  loss={loss_val:.4f}  peak_alloc={peak_alloc:.1f}GB  peak_reserved={peak_resv:.1f}GB  ({100*peak_resv/total_mem:.0f}% of card)")
        except RuntimeError as e:
            msg = str(e).splitlines()[0]
            print(f"batch={bs:3d}  OOM/ERROR: {msg}")
        finally:
            del_vars = ["student", "student_dino", "teachers", "teacher_dinos", "optimizer", "train_loader"]
            for v in del_vars:
                if v in locals():
                    del locals()[v]
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()
