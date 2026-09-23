# GeoKD 实验结果追踪

单卡（GPU 0）串行跑，batch 56，VIGOR cross-area，10 epochs。指标为验证集定位误差（米，越小越好）。
每个 run 完成一个 epoch 会在对应 log 里打印 `Epoch x/10: train_loss=, mean=, median=, fps=`。

## 对照矩阵

| 实验 | 初始化 | 训练 | distill_mode | 状态 | best mean | best median | log |
|---|---|---|---|---|---|---|---|
| T teacher 上界 | — | eval only | — | ✅ 完成 | **2.6409** | **1.2132** | runs/chain_config_vigor_T_teacher_gpu0.log |
| A 无KD下界 | 随机 | supervised | — | **运行中 2卡DDP**(GPU0+2, b32x2) | | | runs/vigor_A_supervised_2gpu.log |
| B 仅初始化 | teacher切片 | eval only | — | 暂缓 | | | |
| C 仅蒸馏 | 随机 | distill | peak_hard | ✅ 完成(10ep) | **3.6268** | **1.8022** | runs/chain_config_vigor_C_gpu0.log |
| **D 当前方法** | teacher切片 | distill | peak_hard | ✅ 完成(10ep) | **3.7588** | **1.8516** | runs/chain_config_vigor_kd_gpu0.log |
| freq (SDKD) | teacher切片 | distill | freq | ✅ 完成(10ep) | **3.7993** | **1.8859** | runs/chain_config_vigor_freq_gpu0.log |
| boundary (GaitKD) | teacher切片 | distill | peak_hard+boundary | ✅ 完成(10ep) | **3.7522** | **1.8492** | runs/chain_config_vigor_boundary_gpu0.log |

判定：C>A 且 D>B ⇒ 蒸馏本身有效。

## 逐 epoch 记录

### D (peak_hard, init_from_teacher) — 运行中
启动 2026-08-03 22:04，GPU0 ~68GB/84%，736 it/epoch，~4.1 s/it（≈50min/epoch）。

| epoch | train_loss | mean | median | fps |
|---|---|---|---|---|
| _待第一个 epoch 完成_ | | | | |

## 备注
- 只用 GPU 0/6（其余卡他用）。
- D: GPU0, batch 56 (84%). A: GPU6, batch 32 / epochs 4（b32 对 O(B²) loss 吞吐最优 ~9.4h/ep）。
- 标签：已确认 SliceMatch 修正版（byte 级一致）+ 修正分辨率。
- **自动链式**（tools/chain.sh, GPU0）：D 跑完 → 自动起 freq → 再起 boundary（C/T 上轮已完成，不重跑）。日志 runs/chain_gpu0.log。
- A 监督很慢 ~10h/epoch（batch_wise B 循环是 GPU 串行瓶颈，非 batch 问题）；可早停。
- wandb 离线：`wandb/offline-run-*`。
