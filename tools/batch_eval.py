"""Batch-evaluate every VIGOR / KITTI checkpoint on the held-out test set(s).

Reads each ckpt's stored metadata (backbone, student_width, ...) so the eval
config matches the training-time architecture.  Dispatches jobs round-robin
across the requested GPU list, writes one row per (ckpt, split) to a CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml


REPO = Path("/home/yiru_fang/geokd-main")
PY = REPO / ".venv" / "bin" / "python"
TRAIN_SCRIPT = REPO / "train_geokd.py"
LOG_DIR = REPO / "runs" / "logs" / "batch_eval"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------- ckpt inspection --------------------------------

def read_ckpt_meta(path: Path) -> Dict[str, Any]:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    return {k: obj.get(k) for k in [
        "teacher_dino_model", "student_dino_model", "student_width",
        "distill_mode", "supervision", "batch_size_per_gpu",
        "teacher_ckpt", "experiment_name", "model_name",
    ]}


# ---------------------------- config builders --------------------------------

def base_vigor_data() -> Dict[str, Any]:
    return {
        "dataset": "vigor",
        "dataset_root": "/home/yiru_fang/VIGOR",
        "cross_area": True,
        "image_size": 512,
        "sat_size": 640,
        "zoom": 20,
        "bev_size": 512,
        "grid_size": 8,
        "ori_noise": 45,
    }


def base_kitti_data(cross_area: bool) -> Dict[str, Any]:
    return {
        "dataset": "kitti",
        "dataset_root": "/home/yiru_fang/KITTI/KITTI",
        "train_file": "/home/yiru_fang/KITTI/dataLoader/train_files.txt",
        "test1_file": "/home/yiru_fang/KITTI/dataLoader/test1_files.txt",
        "test2_file": "/home/yiru_fang/KITTI/dataLoader/test2_files.txt",
        "cross_area": cross_area,
        "rotation_range": 0,
        "shift_range_lat": 20.0,
        "shift_range_lon": 20.0,
        "image_size": 512,
        "sat_size": 640,
        "zoom": 20,
        "bev_size": 300,
    }


def build_config(job: Dict[str, Any]) -> Dict[str, Any]:
    """job keys: dataset, target (teacher|student), ckpt, cross_area,
                 teacher_dino_model, student_dino_model, student_width, name.
    """
    if job["dataset"] == "vigor":
        data = base_vigor_data()
    else:
        data = base_kitti_data(job["cross_area"])

    if job["target"] == "teacher":
        teacher_ckpt = str(job["ckpt"])
        student_ckpt = None
    else:
        teacher_ckpt = "/home/yiru_fang/geokd-main/ckpt/translation/vigor/cross-geodistill-dino.pth"  # unused in eval
        student_ckpt = str(job["ckpt"])

    return {
        "data": data,
        "model": {
            "teacher_ckpt": teacher_ckpt,
            "student_ckpt": student_ckpt,
            "teacher_dino_model": job["teacher_dino_model"],
            "student_dino_model": job["student_dino_model"],
            "student_width": job["student_width"],
            "init_from_teacher": False,
            "levels": [0, 2],
            "channels": [64, 16, 4],
        },
        "distill": {
            "student_temp": 0.06,
            "teacher_temp": 0.06,
            "distill_mode": "peak_hard",
        },
        "train": {
            "train": False,
            "batch_size_per_gpu": job.get("batch_size", 32),
            "epochs": 1,
            "seed": 2023,
            "clip": 1.0,
        },
        "eval": {"target": job["target"]},
        "optim": {"lr": 1e-4, "min_lr": 1e-5, "weight_decay": 1e-5},
        "runtime": {
            "distributed": False,
            "gpuid": [0],
            "num_workers": 8,
        },
        "logging": {
            "wandb": False,
            "wandb_log_interval": 20,
            "visualize": False,
            "save_visualization": False,
            "vis_freq": 100,
        },
        "output": {
            "save_path": "/tmp/batch_eval_output",
            "name": job["name"],
        },
    }


# ---------------------------- job dispatch ----------------------------------

_MEAN_RE = re.compile(r"Test mean distance:\s*([0-9.]+)")
_MEDIAN_RE = re.compile(r"Test median distance:\s*([0-9.]+)")
_FPS_RE = re.compile(r"Test FPS:\s*([0-9.]+)")


def run_one(job: Dict[str, Any], gpu: int) -> Dict[str, Any]:
    cfg = build_config(job)
    tag = f"{job['name']}_gpu{gpu}"
    log_path = LOG_DIR / f"{tag}.log"

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        cfg_path = f.name

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    try:
        with open(log_path, "w") as flog:
            proc = subprocess.run(
                [str(PY), str(TRAIN_SCRIPT), "--config", cfg_path],
                stdout=flog, stderr=subprocess.STDOUT, env=env, check=False,
            )
        rc = proc.returncode
    finally:
        os.unlink(cfg_path)

    dur = time.time() - t0
    txt = log_path.read_text(errors="replace")
    mean = _MEAN_RE.search(txt)
    median = _MEDIAN_RE.search(txt)
    fps = _FPS_RE.search(txt)
    return {
        **job,
        "gpu": gpu,
        "rc": rc,
        "sec": round(dur, 1),
        "mean": float(mean.group(1)) if mean else None,
        "median": float(median.group(1)) if median else None,
        "fps": float(fps.group(1)) if fps else None,
        "log": str(log_path),
    }


# ---------------------------- ckpt enumeration ------------------------------

def enumerate_jobs() -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    # VIGOR students (cross-area test 26848)
    vigor_student_dir = REPO / "checkpoints" / "vigor" / "geokd" / "student"
    for p in sorted(vigor_student_dir.glob("*_best.pth")):
        name_stem = p.stem
        # Skip teacher-only ckpts saved into student dir (dinov3-teacher-*).
        is_teacher = "teacher" in name_stem
        meta = read_ckpt_meta(p)
        if is_teacher:
            jobs.append({
                "dataset": "vigor",
                "target": "teacher",
                "ckpt": p,
                "cross_area": True,
                "teacher_dino_model": meta.get("teacher_dino_model") or "vitl14",
                "student_dino_model": meta.get("student_dino_model") or "vits14",
                "student_width": 1.0,
                "name": f"eval_vigor_teacher__{name_stem}",
                "batch_size": 16,
            })
        else:
            jobs.append({
                "dataset": "vigor",
                "target": "student",
                "ckpt": p,
                "cross_area": True,
                "teacher_dino_model": meta.get("teacher_dino_model") or "vitb14",
                "student_dino_model": meta.get("student_dino_model") or "vits14",
                "student_width": meta.get("student_width") or 0.5,
                "name": f"eval_vigor_student__{name_stem}",
                "batch_size": 32,
            })

    # VIGOR main teacher (external ckpt)
    ext_teacher = REPO / "ckpt" / "translation" / "vigor" / "cross-geodistill-dino.pth"
    jobs.append({
        "dataset": "vigor",
        "target": "teacher",
        "ckpt": ext_teacher,
        "cross_area": True,
        "teacher_dino_model": "vitb14",
        "student_dino_model": "vits14",
        "student_width": 1.0,
        "name": "eval_vigor_teacher__cross-geodistill-dino",
        "batch_size": 16,
    })

    # KITTI teachers (2 files, evaluate on BOTH test1 and test2)
    kitti_dir = REPO / "checkpoints" / "kitti" / "geokd" / "student"
    kitti_teachers = [
        kitti_dir / "kitti-teacher-vitl14_best.pth",             # v1 contrastive
        kitti_dir / "kitti-teacher-vitl14-localce_best_ep10_backup.pth",  # v2 10ep (localce, backup preserved)
        kitti_dir / "kitti-teacher-vitl14-localce-ext_best.pth",  # v2-ext 30ep
    ]
    for p in kitti_teachers:
        if not p.exists():
            continue
        for cross_area in (False, True):  # test1 (same-area), test2 (cross-area)
            split = "test2_cross" if cross_area else "test1_same"
            jobs.append({
                "dataset": "kitti",
                "target": "teacher",
                "ckpt": p,
                "cross_area": cross_area,
                "teacher_dino_model": "vitl14",
                "student_dino_model": "vitl14",
                "student_width": 1.0,
                "name": f"eval_kitti_teacher__{p.stem}__{split}",
                "batch_size": 16,
            })

    # KITTI students: auto-discover all *_best.pth that are NOT teacher ckpts.
    kitti_students = sorted(
        p for p in kitti_dir.glob("*_best.pth")
        if "teacher" not in p.stem
    )
    for p in kitti_students:
        if not p.exists():
            continue
        meta = read_ckpt_meta(p)
        for cross_area in (False, True):
            split = "test2_cross" if cross_area else "test1_same"
            jobs.append({
                "dataset": "kitti",
                "target": "student",
                "ckpt": p,
                "cross_area": cross_area,
                "teacher_dino_model": meta.get("teacher_dino_model") or "vitl14",
                "student_dino_model": meta.get("student_dino_model") or "vits14",
                "student_width": meta.get("student_width") or 0.5,
                "name": f"eval_kitti_student__{p.stem}__{split}",
                "batch_size": 32,
            })

    return jobs


# ---------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                    help="comma-separated GPU ids to use")
    ap.add_argument("--out", type=str, default=str(REPO / "runs" / "batch_eval_results.csv"))
    ap.add_argument("--limit", type=int, default=0,
                    help="only run first N jobs (0=all, for smoke test)")
    ap.add_argument("--filter", type=str, default="",
                    help="comma-separated substrings; keep jobs whose name matches ANY")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    jobs = enumerate_jobs()
    if args.filter:
        subs = [s for s in args.filter.split(",") if s.strip()]
        jobs = [j for j in jobs if any(s in j["name"] for s in subs)]
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"[batch_eval] enumerated {len(jobs)} jobs, using GPUs {gpus}", flush=True)
    for j in jobs:
        print(f"  - {j['name']}  ({j['dataset']}/{j['target']} bs={j['batch_size']})")

    # GPU allocation via a semaphore-like Queue
    from queue import Queue
    gpu_q: Queue = Queue()
    for g in gpus:
        gpu_q.put(g)

    lock = threading.Lock()
    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "dataset", "target", "cross_area", "mean", "median",
                  "fps", "sec", "gpu", "rc", "ckpt", "log"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def worker(job):
        gpu = gpu_q.get()
        try:
            print(f"[start] {job['name']} on GPU {gpu}", flush=True)
            res = run_one(job, gpu)
            with lock:
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow({
                        "name": res["name"],
                        "dataset": res["dataset"],
                        "target": res["target"],
                        "cross_area": res["cross_area"],
                        "mean": res["mean"],
                        "median": res["median"],
                        "fps": res["fps"],
                        "sec": res["sec"],
                        "gpu": res["gpu"],
                        "rc": res["rc"],
                        "ckpt": str(res["ckpt"]),
                        "log": res["log"],
                    })
            status = "OK" if res["rc"] == 0 and res["mean"] is not None else f"FAIL rc={res['rc']}"
            print(f"[done ] {job['name']}  GPU{gpu}  {status}  mean={res['mean']}  median={res['median']}  ({res['sec']}s)", flush=True)
            return res
        finally:
            gpu_q.put(gpu)

    results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futures = [ex.submit(worker, j) for j in jobs]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"[error] {e}", flush=True)

    print(f"\n[batch_eval] complete: {len(results)} results -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
