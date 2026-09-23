# PHKD: Peak-and-Hard Knowledge Distillation for Fine-Grained Cross-View Geo-Localization

This repository contains the training and evaluation code for **PHKD** (Peak-and-Hard Knowledge Distillation), a spatial-heatmap distillation framework for lightweight cross-view geo-localization models on VIGOR and KITTI cross-area benchmarks.

The method distills a fixed **GeoDistill** teacher (DINOv2 ViT-B/14 + full DPT head) into a compact student (DINOv2 ViT-S/14 + SlimDPT half-width head), reducing parameters by 74.8% and FLOPs by 73.3% with essentially no loss in mean localization error and a measurable improvement in median error.

- **VIGOR Cross-Area**: 2.6575 m mean / 1.1585 m median (best among all evaluated KD baselines; median beats the teacher).
- **KITTI cross-area**: 11.112 m mean / 10.4101 m median (matches strongest baseline on mean; best on median).

## 1. Repository Layout

```text
phkd/
├── train_geokd.py              # unified train/validate/test entry point
├── geometry.py                 # BEV projection utilities
├── VGG.py                      # legacy component still imported by some networks
├── README.md                   # this file
├── model/
│   ├── geokd.py                # student builder / teacher-to-student weight copy
│   ├── dino.py                 # DINOv2 wrapper (torch.hub)
│   ├── dpt.py                  # full-width DPT head (teacher head)
│   ├── slim_dpt.py             # SlimDPT head (student head, half channels)
│   ├── loss.py                 # all distillation losses (peak_hard, ce, mse, dist, dkd, ...)
│   ├── network_vigor.py        # VIGOR localization network + correlation
│   ├── network_kitti_dino.py   # KITTI localization network + correlation
│   └── efficientnet_pytorch/   # legacy dependency
├── dataset/
│   ├── VIGOR.py                # VIGOR dataset + dataloader
│   ├── KITTI.py                # KITTI dataset + dataloader
│   ├── config_*_kd.yaml        # legacy KD configs (kept for reference)
│   ├── config_vigor_*.yaml     # 31 VIGOR training configs
│   ├── config_kitti_*.yaml     # 29 KITTI training configs
│   ├── RESULTS.md              # historical sweep results
│   └── PROGRESS_DINOv3_vs_DINOv2.md
├── tools/
│   ├── batch_eval.py           # multi-GPU batch evaluation on test sets
│   ├── run_serial_queue.sh     # serial job queue (used to chain training runs)
│   ├── make_kitti_baseline_configs.py
│   ├── make_qualitative_figure.py  # 2x7 qualitative comparison figure
│   ├── make_heatmap_examples.py    # 11 pipeline heatmap illustrations
│   ├── make_pipeline_examples.py   # 10 pipeline-stage example images
│   ├── assemble_pipeline_figure.py # composes pipeline_concrete_{2row, 6panel, horizontal}
│   ├── scan_qualitative_samples.py # scan test set for best PHKD-vs-baselines cases
│   └── efficiency.py           # parameters/FLOPs/latency measurement
├── utils/
│   └── util.py                 # seed, metrics, visualization helpers
└── docs/                       # extra design notes
```

**Not shipped** (large or environment-specific): `checkpoints/`, `ckpt/`, `runs/`, `wandb/`, `.venv/`.

## 2. Environment

- Python 3.10+ (tested with 3.12)
- PyTorch 2.x with CUDA (tested with 2.1+cu121)
- 1x NVIDIA H100 80 GB is sufficient for VIGOR student training; KITTI needs less memory.

Suggested install:

```bash
python -m venv .venv
source .venv/bin/activate

pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.0 torchvision==0.16.0

pip install numpy scipy scikit-learn opencv-python-headless pillow \
            pyyaml tqdm einops matplotlib pandas wandb
```

`model/dino.py` loads DINOv2 via `torch.hub.load("facebookresearch/dinov2", ...)`. The first run downloads the checkpoint into `~/.cache/torch/hub`; make sure the machine can reach GitHub or pre-populate the cache.

## 3. Datasets

Set your dataset roots in each config YAML (default paths shown here).

### VIGOR

```
VIGOR/
├── NewYork/     Seattle/     SanFrancisco/     Chicago/
│   ├── pano_label_balanced__corrected.txt
│   ├── same_area_balanced_train__corrected.txt
│   ├── same_area_balanced_test__corrected.txt
│   ├── satellite_list.txt
│   ├── panorama/*
│   └── satellite/*
```

Update `dataset_root` in the VIGOR configs (default `/home/yiru_fang/VIGOR`).
Cross-area split: NewYork + Seattle → train (51,518 panoramas); SanFrancisco + Chicago → test (53,692 panoramas).

### KITTI

Follows the [HighlyAccurate](https://github.com/shiyujiao/HighlyAccurate) split:

```
KITTI/
├── KITTI/                       # image data
└── dataLoader/
    ├── train_files.txt
    ├── test1_files.txt          # same-area test  (3,773 panoramas)
    └── test2_files.txt          # cross-area test (7,542 panoramas, unseen trajectories)
```

Update `dataset_root`, `train_file`, `test1_file`, `test2_file` in KITTI configs.

### Teacher checkpoints

Place the fixed GeoDistill teacher weights at:

- VIGOR: `ckpt/translation/vigor/cross-geodistill-dino.pth`
- KITTI: `checkpoints/kitti/geokd/student/kitti-teacher-vitl14-localce-ext_best.pth`

(Or point each config's `model.teacher_ckpt` at your own path.)

## 4. Reproducing the Paper Results

### 4.1 VIGOR PHKD (best VIGOR model)

Config: `dataset/config_vigor_peak_hard_peak_w16.yaml` — Peak-and-Hard KD with peak weight λ=16, K=32 hard negatives, radius r=2, 10 epochs, batch size 56.

```bash
CUDA_VISIBLE_DEVICES=0 python train_geokd.py \
    --config dataset/config_vigor_peak_hard_peak_w16.yaml
```

Saves best-val checkpoint to
`checkpoints/vigor/geokd/student/vigor-cross-peak-hard-peak-w16_best.pth`.

Expected VIGOR Cross-Area test error: **2.6575 m mean / 1.1585 m median**.

### 4.2 KITTI PHKD (best KITTI model)

Config: `dataset/config_kitti_phkd_30ep.yaml` — PHKD with default λ=4, K=32, r=2, **30 epochs**, batch size 32. Longer training on KITTI is what gives the improvement over the 10-epoch baselines.

```bash
CUDA_VISIBLE_DEVICES=0 python train_geokd.py \
    --config dataset/config_kitti_phkd_30ep.yaml
```

Expected KITTI test2 (cross-area) error: **11.112 m mean / 10.4101 m median**.

### 4.3 Ablation configs (Table 3)

| Row | Config |
| --- | --- |
| KD baseline | `dataset/config_vigor_hinton_t1.yaml` |
| Peak only  | `dataset/config_vigor_peak_only_w16.yaml` |
| Hard only  | `dataset/config_vigor_hard_only.yaml` |
| PHKD       | `dataset/config_vigor_peak_hard_peak_w16.yaml` |

### 4.4 Hyperparameter sensitivity (Table 4)

Vary a single hyperparameter around the reported setting:

```bash
# λ sweep
python train_geokd.py --config dataset/config_vigor_peak_hard.yaml         # λ=4
python train_geokd.py --config dataset/config_vigor_peak_hard_peak_w8.yaml # λ=8
python train_geokd.py --config dataset/config_vigor_peak_hard_peak_w16.yaml # λ=16

# K sweep
python train_geokd.py --config dataset/config_vigor_peak_hard_topk4.yaml   # K=4
python train_geokd.py --config dataset/config_vigor_peak_hard_topk8.yaml   # K=8
python train_geokd.py --config dataset/config_vigor_peak_hard_hn_topk64.yaml # K=64

# r sweep
python train_geokd.py --config dataset/config_vigor_peak_hard_r3.yaml      # r=3
```

### 4.5 Table 1 baselines

Every reported baseline has a matching config: `config_vigor_hinton_t1.yaml` (KD), `config_vigor_mse.yaml` (Logit MSE), `config_vigor_dist.yaml` (DIST), `config_vigor_dkd.yaml` (DKD), `config_vigor_ce_zscore.yaml` (Logit Std.), `config_vigor_boundary.yaml` (AB), `config_vigor_freq.yaml` (SDKD), `config_vigor_wavelet.yaml` (DS²D²), and analogous `config_kitti_*.yaml`. Run each with `python train_geokd.py --config <yaml>`.

## 5. Batch Evaluation

`tools/batch_eval.py` dispatches evaluation over multiple GPUs and writes a CSV with one row per (checkpoint, test split).

```bash
# Evaluate every discovered checkpoint on all GPUs
python tools/batch_eval.py --gpus 0,1,2,3,4,5,6,7 \
    --out runs/batch_eval_results.csv

# Filter to specific checkpoints (comma-separated substrings; matches ANY)
python tools/batch_eval.py --gpus 0 \
    --filter "peak-hard-peak-w16,phkd-30ep" \
    --out runs/batch_eval_results_filtered.csv
```

Auto-discovers:

- `checkpoints/vigor/geokd/student/*_best.pth` — VIGOR students + any teacher-labeled ckpts.
- `checkpoints/kitti/geokd/student/kitti-teacher-*_best.pth` — KITTI teachers, evaluated on both test1 (same-area) and test2 (cross-area).
- `checkpoints/kitti/geokd/student/*_best.pth` (non-teacher) — KITTI students, on both splits.

Metadata (backbone, student width, distill mode) is read from each checkpoint so the eval config always matches the training-time architecture.

Result columns: `name, dataset, target, cross_area, mean, median, fps, sec, gpu, rc, ckpt, log`.

## 6. Figure Generation

All figure scripts write into `figs/`.

### 6.1 Qualitative comparison (2×7)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/make_qualitative_figure.py \
    --indices 87 166 \
    --out figs/qualitative_comparison.pdf
```

Columns are Teacher / KD / DIST / DKD / AB / DS²D² / **PHKD**, showing satellite tile + softmax score overlay + GT ★ + predicted × + teacher peak region (yellow dashed box). Indices 87 and 166 are the samples where PHKD wins by the largest margin over the strongest baseline (found via `tools/scan_qualitative_samples.py`).

### 6.2 Pipeline example images

```bash
CUDA_VISIBLE_DEVICES=0 python tools/make_pipeline_examples.py    # 10 stage images
CUDA_VISIBLE_DEVICES=0 python tools/make_heatmap_examples.py     # 11 heatmap variants
python tools/assemble_pipeline_figure.py                          # 3 composite layouts
```

Outputs land in `figs/pipeline_examples/`:

- `01_ground_panorama.png … 10_summary_3panel.png` — stage-by-stage inputs and heatmaps.
- `h01_teacher_raw.png … h11_4panel_overlay.png` — raw heatmaps, sat overlays, side-by-side comparisons.
- `pipeline_concrete_2row.{png,pdf}`, `pipeline_concrete_6panel.png`, `pipeline_concrete_horizontal.png` — assembled pipeline figures using the actual example images.

### 6.3 Sample-picking helper

```bash
CUDA_VISIBLE_DEVICES=0 python tools/scan_qualitative_samples.py 300
```

Scans the first N VIGOR cross-area test samples, computes per-sample localization error for the teacher and 6 student models, and prints the samples with the largest PHKD-vs-baseline margin.

## 7. Serial Queue Runner (multi-run scheduling)

Chain long training runs behind a "wait until PID X exits" gate. Example (three GPUs, each running two configs sequentially):

```bash
# Wait for PID 1 (the init PID, always present) — starts immediately
nohup bash tools/run_serial_queue.sh 0 1 q_gpu0 \
    dataset/config_vigor_peak_hard_peak_w16.yaml \
    dataset/config_kitti_phkd_30ep.yaml \
    > runs/logs/DRIVER_q_gpu0.log 2>&1 &

nohup bash tools/run_serial_queue.sh 1 1 q_gpu1 \
    dataset/config_vigor_peak_only_w16.yaml \
    dataset/config_kitti_phkd_boundary.yaml \
    > runs/logs/DRIVER_q_gpu1.log 2>&1 &
```

State file at `/tmp/<queue_tag>.state` and per-child logs under `runs/logs/`.

## 8. Config Reference (key fields)

```yaml
data:
  dataset: vigor | kitti
  dataset_root: <path>
  cross_area: true    # cross-area (test2 for KITTI) vs same-area

model:
  teacher_ckpt: <path to teacher .pth>
  student_ckpt: null                    # or resume path
  teacher_dino_model: vitb14 | vitl14   # KITTI teacher uses vitl14
  student_dino_model: vits14
  student_width: 0.5                    # SlimDPT half-channel
  init_from_teacher: false              # true = teacher-slice init (usually worse on VIGOR)
  levels: [0, 2]                        # DPT output levels used for correlation
  channels: [64, 16, 4]                 # SlimDPT intermediate widths

distill:
  student_temp: 0.06                    # τ in softmax
  teacher_temp: 0.06
  distill_mode: peak_hard | ce | mse | dist | dkd | ce_zscore | freq | wavelet
  # PHKD-specific:
  peak_radius: 2                        # r
  peak_topk: 1                          # number of teacher peak locations (1 = single peak)
  peak_region_weight: 16.0              # λ  (VIGOR best) / 4.0 (KITTI best)
  hard_negative_topk: 32                # K
  hard_negative_weight: 1.0             # β

train:
  train: true
  supervision: distill                  # supervised = No-KD baseline
  batch_size_per_gpu: 56                # VIGOR / 32 for KITTI
  epochs: 10                            # VIGOR / 30 for KITTI PHKD
  seed: 2023

optim:
  lr: 0.0001
  min_lr: 0.00001                       # cosine schedule minimum
  weight_decay: 0.00001

output:
  save_path: checkpoints
  name: <experiment-name>               # ckpt saved as <name>_best.pth
```

## 9. Loss Reference (`model/loss.py`)

| `distill_mode` | Meaning |
| --- | --- |
| `peak_hard` | Peak-reweighted target + top-K hard-negative suppression (PHKD, main method) |
| `ce` | Standard soft-target cross-entropy (Hinton KD) — uses `student_temp` / `teacher_temp` |
| `mse` | MSE between teacher and student logits |
| `dist` | DIST loss (intra + inter class distance matching) |
| `dkd` | DKD (reuses `peak_region_weight`=α, `hard_negative_weight`=β) |
| `ce_zscore` | Logit standardization + soft-target CE |
| `freq` | SDKD (frequency-decomposed CE) |
| `wavelet` | DS²D² (wavelet decomposition) |
| `peak_hard_ce`, `peak_hard_zscore` | Peak-hard combined with an auxiliary loss term |
| `peak_hard_wavelet_allwin` | multi-peak / multi-K PHKD variant explored during ablation |

## 10. Trained Checkpoints (not shipped)

Trained checkpoints are large and are not included. Re-run any config to reproduce; each config writes to `checkpoints/<dataset>/geokd/student/<name>_best.pth`.

Key checkpoints referenced in the paper:

- `vigor-cross-peak-hard-peak-w16_best.pth` — PHKD @ VIGOR (Table 1 row).
- `vigor-cross-peak-only-w16_best.pth`      — Peak-only @ VIGOR (Table 3 ablation).
- `vigor-cross-hard-only_best.pth`          — Hard-only @ VIGOR (Table 3 ablation).
- `vigor-cross-hinton-t1_best.pth`          — KD @ VIGOR (Table 3 baseline).
- `kitti-cross-phkd-30ep_best.pth`          — PHKD @ KITTI (Table 1 row, 30 epochs).
- `kitti-cross-supervised-baseline_best.pth` — No-KD @ KITTI (Table 1 row).
- `kitti-teacher-vitl14-localce-ext_best.pth` — KITTI teacher.

## 11. Citation

If you find PHKD useful, please cite our paper (ICASSP 2026 submission).

```
@inproceedings{phkd2026,
  title  = {PHKD: Distilling Lightweight Models for Fine-Grained Cross-View Geo-Localization},
  author = {Li, Yuqi and Fang, Yiru and Tong, Shaowen and Feng, Xiaoqin and
            Yang, Chuanguang and Duan, Huiran and Tian, Yingli},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year   = {2026}
}
```

## 12. Acknowledgements

This code base builds on the excellent [GeoDistill](https://github.com/Yujiao-Shi/GeoDistill) and [HighlyAccurate](https://github.com/shiyujiao/HighlyAccurate) repositories. The teacher checkpoints for VIGOR follow the GeoDistill release; DINOv2 backbones follow [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).
