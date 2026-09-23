"""Measure efficiency metrics (params / FLOPs / latency / FPS) for teacher and student.

Runs the real VIGOR inference path (DINO backbone + localization head + val correlation)
at batch size 1 on one GPU. Backbones are frozen; this reflects deployment cost.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from train_geokd import namespace_from_config, build_localization_model
from model.dino import DINO


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def measure(args, dino_name, student_width, device, tag):
    dino = DINO(model_name=dino_name).to(device).eval()
    net = build_localization_model(args, dataset="vigor", dino_model=dino_name, student_width=student_width).to(device).eval()
    for p in dino.parameters():
        p.requires_grad = False
    for p in net.parameters():
        p.requires_grad = False

    dino_p = count_params(dino)
    head_p = count_params(net)

    sat = torch.rand(1, 3, args.image_size, args.image_size, device=device) * 2 - 1
    pano = torch.rand(1, 3, 320, 640, device=device) * 2 - 1
    mpp = torch.tensor([0.14], device=device)

    def forward():
        sat_feat_list = dino(sat.contiguous())
        pano_feat_list = dino(pano.contiguous())
        sat_feat, sat_conf, g2s_feat, g2s_conf, _, _ = net(sat_feat_list, pano_feat_list, mpp)
        return net.calc_corr_for_val(sat_feat, sat_conf, g2s_feat, g2s_conf)

    with torch.no_grad():
        for _ in range(5):
            forward()
        torch.cuda.synchronize(device)
        n = 30
        start = time.perf_counter()
        for _ in range(n):
            forward()
        torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - start) / n * 1000

    flops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            forward()
        flops = counter.get_total_flops()
    except Exception as exc:  # noqa: BLE001
        flops = f"ERR {exc}"

    gflops = f"{flops / 1e9:.1f}" if isinstance(flops, int) else str(flops)
    print(f"[{tag}] backbone={dino_name} head_width={student_width}")
    print(f"  params : backbone={dino_p / 1e6:.2f}M  head={head_p / 1e6:.2f}M  total={(dino_p + head_p) / 1e6:.2f}M")
    print(f"  latency: {latency_ms:.2f} ms/img (bs1)   FPS={1000 / latency_ms:.1f}")
    print(f"  FLOPs  : {gflops} GFLOPs (fwd, bs1)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset/config_vigor_kd.yaml")
    cli = parser.parse_args()
    args = namespace_from_config(cli.config)
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    measure(args, args.teacher_dino_model, None, device, "TEACHER")
    measure(args, args.student_dino_model, args.student_width, device, "STUDENT")


if __name__ == "__main__":
    main()
