"""Generate KITTI baseline configs mirroring the VIGOR baselines.

Each config uses the best KITTI teacher (vitl14-localce-ext) and
matches the KITTI data/model/train template. Only the distill
section (and name) differs across baselines.
"""

from pathlib import Path

ROOT = Path("/home/yiru_fang/geokd-main/dataset")
TEACHER = (
    "/home/yiru_fang/geokd-main/checkpoints/kitti/geokd/student/"
    "kitti-teacher-vitl14-localce-ext_best.pth"
)

DATA_MODEL_TRAIN_TEMPLATE = """data:
  dataset: kitti
  dataset_root: /home/yiru_fang/KITTI/KITTI
  train_file: /home/yiru_fang/KITTI/dataLoader/train_files.txt
  test1_file: /home/yiru_fang/KITTI/dataLoader/test1_files.txt
  test2_file: /home/yiru_fang/KITTI/dataLoader/test2_files.txt
  cross_area: true
  rotation_range: 0
  shift_range_lat: 20.0
  shift_range_lon: 20.0
  image_size: 512
  sat_size: 640
  zoom: 20
  bev_size: 300

model:
  teacher_ckpt: {teacher_ckpt}
  student_ckpt: null
  teacher_dino_model: vitl14
  student_dino_model: vits14
  student_width: 0.5
  init_from_teacher: false
  levels: [0, 2]
  channels: [64, 16, 4]

{distill}

train:
  train: true
  supervision: {supervision}
  batch_size_per_gpu: 32
  epochs: 10
  seed: 2023
  clip: 1.0

eval:
  target: student

optim:
  lr: 0.0001
  min_lr: 0.00001
  weight_decay: 0.00001

runtime:
  distributed: false
  gpuid: [0]
  num_workers: 16

logging:
  wandb: false
  wandb_log_interval: 20
  visualize: false
  save_visualization: false
  vis_freq: 100

output:
  save_path: checkpoints
  name: {name}
"""

# Distill sections copied 1:1 from the corresponding VIGOR baselines,
# so the loss dispatcher receives the same hyperparameters on KITTI.
BASELINES = {
    "kitti_hinton_t1": dict(
        name="kitti-cross-hinton-t1",
        supervision="distill",
        distill="""distill:
  student_temp: 1.0
  teacher_temp: 1.0
  distill_mode: ce""",
    ),
    "kitti_hinton_t4": dict(
        name="kitti-cross-hinton-t4",
        supervision="distill",
        distill="""distill:
  student_temp: 4.0
  teacher_temp: 4.0
  distill_mode: ce""",
    ),
    "kitti_mse": dict(
        name="kitti-cross-mse",
        supervision="distill",
        distill="""distill:
  student_temp: 0.06
  teacher_temp: 0.06
  distill_mode: mse""",
    ),
    "kitti_dist": dict(
        name="kitti-cross-dist",
        supervision="distill",
        distill="""distill:
  student_temp: 0.06
  teacher_temp: 0.06
  distill_mode: dist""",
    ),
    "kitti_dkd": dict(
        name="kitti-cross-dkd",
        supervision="distill",
        distill="""distill:
  student_temp: 0.06
  teacher_temp: 0.06
  distill_mode: dkd
  peak_radius: 2
  peak_topk: 1
  peak_region_weight: 1.0
  hard_negative_weight: 8.0""",
    ),
    "kitti_ce_zscore": dict(
        name="kitti-cross-ce-zscore",
        supervision="distill",
        distill="""distill:
  student_temp: 2.0
  teacher_temp: 2.0
  distill_mode: ce_zscore""",
    ),
    "kitti_boundary": dict(
        name="kitti-cross-boundary",
        supervision="distill",
        distill="""distill:
  student_temp: 0.06
  teacher_temp: 0.06
  distill_mode: peak_hard
  peak_radius: 2
  peak_topk: 1
  peak_region_weight: 4.0
  hard_negative_topk: 32
  hard_negative_weight: 1.0
  boundary_weight: 0.5
  boundary_margin: 1.0""",
    ),
    "kitti_freq": dict(
        name="kitti-cross-freq",
        supervision="distill",
        distill="""distill:
  student_temp: 0.06
  teacher_temp: 0.06
  distill_mode: freq
  freq_cutoff: 0.25
  freq_low_weight: 1.0
  freq_high_weight: 1.0
  freq_ce_weight: 1.0""",
    ),
    "kitti_wavelet": dict(
        name="kitti-cross-wavelet",
        supervision="distill",
        distill="""distill:
  student_temp: 0.06
  teacher_temp: 0.06
  distill_mode: wavelet
  freq_low_weight: 1.0
  freq_high_weight: 1.0
  freq_ce_weight: 1.0""",
    ),
    # PHKD retrained with the new best KITTI teacher, for a fair Table 1 comparison.
    "kitti_peak_hard_v2": dict(
        name="kitti-cross-phkd",
        supervision="distill",
        distill="""distill:
  student_temp: 0.06
  teacher_temp: 0.06
  distill_mode: peak_hard
  peak_radius: 2
  peak_topk: 1
  peak_region_weight: 4.0
  hard_negative_topk: 32
  hard_negative_weight: 1.0""",
    ),
    "kitti_supervised": dict(
        name="kitti-cross-supervised-baseline",
        supervision="supervised",
        # teacher_ckpt overridden to null for the No-KD baseline.
        distill="""distill:
  distill_mode: ce""",
    ),
}


def main():
    for key, cfg in BASELINES.items():
        teacher = "null" if cfg["supervision"] == "supervised" else TEACHER
        text = DATA_MODEL_TRAIN_TEMPLATE.format(
            teacher_ckpt=teacher,
            distill=cfg["distill"],
            supervision=cfg["supervision"],
            name=cfg["name"],
        )
        out = ROOT / f"config_{key}.yaml"
        out.write_text(text)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
