# CCVPE 蒸馏损失候选清单（SOTA 调研）

> 目标：为下一步"用不同 loss 蒸馏 CCVPE"的对比实验准备候选。
>
> **前置提醒**：当前 CCVPE 蒸馏 pipeline 有 mode collapse（freq / peak_hard+boundary 完整跑完 10 epoch，mean 稳定在 ~44–49m 不下降），根因未定位。**建议先修 pipeline，再跑 loss ablation**，否则任何 loss 得到的都是 collapse 后的假信号。

---

## 1. 任务特征回顾（决定哪些 loss 合适）

CCVPE 蒸馏任务不同于 RepDistiller 里 CIFAR-100 分类的地方：

| 维度 | CCVPE 任务 |
|---|---|
| Teacher 输出 | **相关性热图**（correlation heatmap，本质是 sat 上每像素的匹配打分），非 logits |
| Student 输出 | 同结构相关性热图（不同架构：CNN vs DPT）|
| Loss target | 空间稠密预测（dense prediction），不是分类 |
| 架构差异 | teacher = ViT+DPT，student = 纯 CNN → **跨架构蒸馏** |
| 特征分辨率 | 有多尺度可用（levels=[0,2]，对应 corr_map 尺寸 [B,H,W]） |
| 现有 backbone 特征差异 | teacher 是 DINOv2 冻结特征，channel 数多；student 是 CCVPE 自己的 CNN encoder |

**关键结论**：
- 分类 KD 方法（KD/DKD/MLKD/DIST）需要**适配到空间维度**（把 heatmap flatten 当 logits，或按位置做 per-pixel KD）。
- **稠密预测 KD**（CWD/MGD/FGD/ReviewKD）天然更贴合。
- **跨架构 KD**（DIST、ReviewKD、MGD）对我们特别关键，因为架构差异大。

---

## 2. 已有基线（本项目已跑）

| 名字 | 类别 | 状态 | 备注 |
|---|---|---|---|
| `ce` (KD/DistillKL) | classical logit KD | 已实现，旧运行 OOM 未完 | 等价 RepDistiller 的 `DistillKL`，温度化 softmax + KL/CE |
| `peak_hard` | 变体（自研）| 已实现，旧运行 OOM 未完 | 强化 peak 位置的 hard negative |
| `freq` (SDKD) | 频域对齐 | ✅ 完成 10 epoch，**mode collapse** | 频域一致性 loss |
| `peak_hard + boundary` (GaitKD 风格) | attention hinge | ✅ 完成 10 epoch，**mode collapse** | channel-attention + margin hinge |

---

## 3. 候选清单（按优先级排序）

### ★★★ 高优先级 —— 稠密预测 SOTA / 跨架构友好

#### 3.1 CWD (Channel-wise Distillation, ICCV 2021)
- **论文**：[Channel-wise Knowledge Distillation for Dense Prediction](https://arxiv.org/abs/2011.13256)
- **思路**：把每个通道的空间 feature map 用 softmax 归一化成概率分布，再算 teacher/student 逐通道 KL。
- **为什么适合**：
  - 专为 **dense prediction**（语义分割）设计，与 CCVPE heatmap 输出天然对齐
  - 只需通道对齐（1×1 conv 适配 channel 数），跨架构没障碍
  - 论文里"KL 会自然关注高显著性区域"的特性，正好符合定位任务的 peak 重要性
- **适配方案**：直接对 corr heatmap 逐通道 softmax + KL；或作用在中间 feature map（sat_feat / g2s_feat）。
- **风险**：低（成熟方法，代码简洁）。

#### 3.2 MGD (Masked Generative Distillation, ECCV 2022)
- **论文**：[Masked Generative Distillation](https://arxiv.org/abs/2205.01529)
- **思路**：随机遮盖 student 特征的一部分位置，用一个小 generator 从可见部分重建 teacher 的完整特征。
- **为什么适合**：
  - 检测/分割 SOTA，对 dense prediction 效果好
  - **跨架构友好**：不要求 student 直接匹配 teacher 特征，而是通过 generator 桥接，对 CNN 学 ViT 特征特别合适
  - 天然抗 mode collapse（重建目标提供了强正则）
- **风险**：中（要加一个小 generator 模块，增加参数和调参）。

#### 3.3 DIST (NeurIPS 2022, Knowledge Distillation from A Stronger Teacher)
- **论文**：[Knowledge Distillation from A Stronger Teacher](https://arxiv.org/abs/2205.10536)
- **思路**：把 KL 换成 **relaxed Pearson correlation**（inter-class 和 intra-class 两个层次），避免强 teacher 分布过窄时 student 无法拟合的问题。
- **为什么适合**：
  - **teacher-student 容量差距大**时 SOTA（我们 teacher DPT + ViT-L 特征 vs student CNN，差距明显）
  - 数学简洁，只是把 KL 换成 corr，几乎零成本
  - 对温度不敏感，减少调参
- **风险**：低。

---

### ★★ 中优先级 —— 通用 SOTA / 需要少量适配

#### 3.4 DKD (Decoupled KD, CVPR 2022)
- **论文**：[Decoupled Knowledge Distillation](https://arxiv.org/abs/2203.08679)
- **思路**：把 KL 分解成 **target class KL (TCKD)** 和 **non-target KL (NCKD)** 两部分，独立加权。
- **为什么适合**：可以适配到我们的 peak_hard 思路——把 corr heatmap 分成"gt 位置 vs 非 gt 位置"两部分做解耦 KD，比自研 peak_hard 更有理论依据。
- **风险**：中（需重新组织 heatmap 为 target/non-target 两组）。

#### 3.5 CRD (Contrastive Representation Distillation, ICLR 2020)
- **论文**：[Contrastive Representation Distillation](https://arxiv.org/abs/1910.10699)
- **思路**：InfoNCE 风格，把 teacher/student 对应样本当正对，batch 内其他当负对。
- **为什么适合**：
  - RepDistiller 里综合性能最强
  - 与 GeoDistill stage1 用的 `multi_scale_contrastive_loss` 思想同源，工程上易衔接
- **风险**：中偏高（需要 memory bank，特征投影层，训练开销 +30~50%）。

#### 3.6 ReviewKD (CVPR 2021)
- **论文**：[Distilling Knowledge via Knowledge Review](https://arxiv.org/abs/2104.09044)
- **思路**：**跨层**特征蒸馏——student 每层特征与 teacher 所有更浅层特征聚合后对齐（不是同层对齐）。
- **为什么适合**：
  - 明确设计用来处理**深层 teacher vs 浅层 student** 的层数不匹配（我们 ViT-L teacher vs CCVPE CNN 就是这种情况）
  - 对分割/检测都有 gain
- **风险**：中偏高（要改前向 hook 拿多层特征，加融合模块）。

---

### ★ 低优先级 —— 经典对照组 / 简单基线

| 方法 | 类别 | 一句话总结 | 相对已有的增量 |
|---|---|---|---|
| FitNet | hint-based | student 单层特征直接 MSE 拟合 teacher 特征 | 最简单基线，跨架构差 |
| AT (Attention Transfer) | attention map | 空间 attention（$\sum_c |F|^2$）用 L2 对齐 | 与我们 boundary 类似，但更简单 |
| RKD | relational | 用样本间的距离/角度关系做 loss | 检索类任务友好 |
| PKT (Probabilistic KT) | 分布 | 用 KDE 估计特征分布再对齐 | 计算重 |
| SP (Similarity Preserving) | 相似度矩阵 | Gram matrix 对齐 | 与 CRD 同源但更轻 |
| VID | 互信息 | 用变分下界估计特征互信息 | 收敛慢 |
| NST | 神经元选择性 | MMD 对齐特征分布 | 适合同架构 |

---

## 4. 建议的下一步实验计划

### Phase 0 —— 修 mode collapse（阻塞项）
先诊断 pipeline bug，检查：
1. **学习率**：CCVPE 原本用 1e-4，蒸馏时是否合适？（DKD/CWD 论文里 dense KD 常用 1e-3～1e-4）
2. **Loss scale**：teacher heatmap 值域 [0,4]（`multi_scale_contrastive_loss` 里就是这个范围），KD loss 是否被淹没在其他项里？
3. **输入归一化对齐**：CCVPE 原本的 sat/pano 归一化统计量是否与 geokd 数据流一致（BGR/RGB、[0,1] 还是 [-1,1]）
4. **一个 sanity check**：teacher forward 出来的 corr map peak 位置是否与 gt 位置对齐（可视化前 20 batch）

### Phase 1 —— 最小改动候选（先跑）
1. **DIST** —— 把 `cross_entropy` 的 KL 换成 relaxed Pearson correlation，代码改动 <10 行，跑一遍看是否直接超越 ce baseline。
2. **CWD** —— 加一个 `channel_wise_kd_loss`，对 corr map 逐通道 softmax+KL。约 30 行。

### Phase 2 —— 中等改动候选（Phase 1 有正向信号后再上）
3. **DKD** —— 拆 corr map 为 gt 位置 vs 非 gt 位置两部分做加权 KL。
4. **MGD** —— 加 mask + 小 generator，对 sat_feat / g2s_feat 中间特征做重建蒸馏。

### Phase 3 —— 可选深挖
5. **CRD** —— 如果前几种都不够，用对比学习框架蒸馏。
6. **ReviewKD** —— 跨层特征聚合，处理 ViT-L teacher vs CNN student 的深度不匹配。

---

## 5. 参考代码库

| 库 | 内容 | 适合参考的实现 |
|---|---|---|
| [HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller) | 12 种经典 KD（KD, FitNet, AT, SP, CC, VID, RKD, PKT, AB, FT, FSP, NST, CRD） | ce/AT/SP/CRD |
| [megvii-research/mdistiller](https://github.com/megvii-research/mdistiller) | 支持 DKD, ReviewKD, KD, FitNet, AT, NST, PKT, KDSVD, OFD, RKD, VID, SP, CRD，覆盖分类/检测 | DKD, ReviewKD |
| [pppppM/mmrazor](https://github.com/open-mmlab/mmrazor) 或 mmseg KD 分支 | CWD, MGD 官方或社区实现 | CWD, MGD |
| DIST 原仓库 [hunto/DIST_KD](https://github.com/hunto/DIST_KD) | 官方 PyTorch 实现 | DIST |

---

## 6. 备忘：为什么先诊断 collapse 再跑 loss 对比

如果不先修 collapse：
- 所有 candidate 都会得到 collapse 结果，无法区分是"loss 无效"还是"pipeline bug"
- Loss ablation 的价值来自于 candidates 之间的相对差异，collapse 会让所有实验 mean 集中在 44–49m，差异被噪声淹没
- 白花 GPU 时间

**先花 1–2 天做 Phase 0 诊断，再进入 Phase 1**，性价比最高。
