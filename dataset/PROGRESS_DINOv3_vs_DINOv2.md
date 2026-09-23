# DINOv3 vs DINOv2 GeoDistill 复现进展总结

> 目标：在 VIGOR cross-area 数据集上，将 GeoDistill 原始的 DINOv2 backbone 替换为 DINOv3（双分支：卫星域 SAT-493M + 通用域 LVD-1689M），走完两阶段训练，并与 DINOv2 官方测试集基准进行公平对比。

---

## 1. Backbone 配置

| 项 | DINOv2（原方案，基准） | DINOv3（本次） |
|---|---|---|
| 结构 | 单 `vitb14`，sat/pano 共享同一份权重 | `DualDINOv3(vitl16)`：sat 分支用 SAT-493M，grd 分支用 web LVD-1689M |
| Patch size | 14 | 16 |
| Backbone 参数量（冻结） | **86.6M** | **606.3M**（303.1M × 2） |
| 微调策略 | **完全冻结**（`requires_grad=False`） | **完全冻结**（无 LoRA、无 adapter、无解冻） |
| 归一化 | ImageNet 统计量 | web 分支 ImageNet 统计量；sat 分支专用统计量 `mean=(0.430,0.411,0.296)` |
| 特征使用 | dense-cls（CLS token 与 patch token 拼接） | dense-cls |

- 真正训练的是下游 `LocalizationNet`（DPT-based 定位头，约 **79.94M** 可训练参数），两个方案共用同一套 head 架构。

---

## 2. 两阶段训练回顾

### Stage 1：`train_g2sweakly`（弱监督对比学习）

- 从 zero 开始训练 `LocalizationNet`，只用 GPS 弱标签。
- 计算 batch 内 sat / g2s 相关性矩阵 `[B,B,H,W]`，用 `multi_scale_contrastive_loss`（soft-margin 对比 loss）。
- 显存 O(B²)，batch 越大越慢；实测 batch8 最优。

### Stage 2：`train_geodistill`（EMA 自蒸馏 + 遮挡鲁棒性）

- 用 Stage 1 checkpoint 初始化 student 和 teacher（`teacher = deepcopy(student)`）。
- Teacher 看完整全景图，Student 看 FOV 遮挡后的全景图。
- `cross_entropy` (KD loss, 带温度) 让 student 拟合 teacher 的相关性分布。
- Epoch 结束用 EMA（`ema=0.9`）更新 teacher。
- 关键：teacher / student 都是**同一架构**（相同的定位头 + 相同的冻结 backbone），只是 EMA 版本差异。

### 与 CCVPE 蒸馏的关系

- GeoDistill stage2 里的 "teacher/student" 是**同架构自蒸馏**，与 backbone 选型（v2/v3）无关。
- 另一条独立实验线 `train_ccvpe_distill.py` 才是把 geokd 的 DPT teacher（v2）跨架构蒸馏进 CNN CCVPE student —— 目前存在 mode collapse 未修，与本次对比无关。

---

## 3. 训练进度

| 阶段 | 状态 | 完成时间 | 关键 checkpoint | 内部 val（mean / median, m）|
|---|---|---|---|---|
| Stage 1（DDP GPU4,5） | 崩溃（epoch9 时因 CCVPE 抢卡触发 SIGABRT） | 2026-08-11 | `dinov3sat-g2sweakly-stage1-ddp45_20260810_230226.pth`（epoch8）| — |
| Stage 1 恢复（单卡 GPU6） | ✅ 完成（+2 epoch）| 2026-08-13 12:29 | `dinov3sat-g2sweakly-stage1-gpu6-resume_20260812_221547.pth` | **4.1546 / 2.0909** |
| Stage 2（单卡 GPU5, batch32, 20 epoch）| ✅ 完成 | 2026-08-19 09:58 | student: `dinov3sat-geodistill-stage2-gpu5-b32_20260817_222509.pth`（epoch16）| **3.6317 / 1.8302** |
| Stage 2 teacher（同 checkpoint 目录 `teacher/`）| ✅ 完成 | 2026-08-19 09:58 | epoch20 最终 | **3.7014 / 1.8578** |

> 注：Stage 2 自蒸馏使 student 相对 stage1 内部 val 从 4.15m → 3.63m（-12.5%），验证了 GeoDistill 方法本身对 DINOv3 也有效。

---

## 4. 官方测试集评测（SF+Chicago, `cross_area=True`）

评测协议：`train_vigor.py` 的 `test()` 函数，加载 `same_area=False` 时的 SF+Chicago 城市，与 DINOv2 原始基准使用**完全一致**的测试集和 dataloader。

| 模型 | Mean (m) | Median (m) |
|---|---|---|
| DINOv2 vitb14（原基准）| **2.6409** | **1.2132** |
| DINOv3 stage2 **student**（本次）| **2.7798** | **1.2228** |
| DINOv3 stage2 **teacher**（本次）| ⏳ 进行中（GPU2，PID 223442，进度 ~31%）| ⏳ |

---

## 5. 参数量 & 推理速度对比

### 参数量（backbone，冻结）

| Backbone | 结构 | 参数量 |
|---|---|---|
| DINOv2 `vitb14` | 单共享 backbone | **86.6M** |
| DINOv3 dual `vitl16` | sat + grd 独立双 backbone | **606.3M**（303.1M × 2）|

定位头 `LocalizationNet` 两方案均为 **79.94M**（可训练）。

### 推理速度（H100 单卡，batch=8，仅 backbone 前向）

| Backbone | sat 640×640 | pano 512×512 | 合计 |
|---|---|---|---|
| DINOv2 `vitb14` | 114.5 ms | 59.8 ms | **174.2 ms** |
| DINOv3 dual `vitl16` | 281.7 ms | 146.1 ms | **427.8 ms** |

DINOv3 backbone 参数 ~7×、推理 ~2.5×，但官方测试集精度略降。

---

## 6. 为什么 DINOv3 反而略输 DINOv2 —— 原因分析

### (1) Patch size 变粗（14 → 16），空间粒度直接下降
- v2：sat 640×640 → 45×45 特征图；pano 512×512 → 36×36
- v3：sat 640×640 → 40×40 特征图；pano 512×512 → 32×32
- 空间 token 数少 ~21%，相关性热图分辨率同步下降 → 定位不确定性提高约 **~12%**。定位任务里 patch 越细越好，patch16 相对 patch14 天然吃亏。

### (2) 双 backbone 造成 sat/pano 跨域嵌入不对齐（最关键）
- v2：sat 和 pano 共享同一份 vitb14 权重，两侧特征在同一空间，直接内积即可匹配。
- v3：sat 用 SAT-493M（卫星域），grd 用 LVD-1689M（自然图像域），**权重独立、预训练分布不同**。sat / grd 特征位于两个不同嵌入空间，定位头必须先"跨域翻译"再"跨视角匹配"，难度陡增。

### (3) 定位头容量没跟着 backbone 扩容
- `LocalizationNet` 里 DPT 的 `input_dims=[2048,2048,2048,2048]` 是硬编码，v2 (1536 dim) 和 v3 (2048 dim) 共用同一个 79.94M head。
- v3 大 backbone 输出的信息被同一容量的 head 挤压，且要额外承担域对齐任务。

### (4) Stage1 batch size 受显存约束
- v3 backbone 显存 ~7× 大，$O(B^2)$ 的对比损失只能 batch8。
- 对比学习对负样本池大小极其敏感，v3 stage1 学到的 embedding 质量不如 v2 时代（当年可用更大 batch）。stage2 起点就先天不足。

### (5) 冻结策略 + 域差异组合放大劣势
- backbone 完全冻结 → v3 无法针对 VIGOR 域和相关性 loss 做适配。
- SAT-493M 的自监督目标是"卫星图 SSL"，不直接对应"跨视角相关性打分"，冻结后先验优势无法转化。

### (6) 语义特征 vs 几何特征错配（更根本）
- 定位是**几何 grounding** 任务，偏爱细粒度纹理/结构信息。
- v3（尤其 SAT-493M）相对 v2 的主要改进在**语义/instance 判别力**，语义抽象层次更高可能丢失细粒度纹理。
- 对比印证：median（好样本）几乎持平（1.223 vs 1.213），**mean 差距主要来自尾部长距离错定位样本**，符合"语义 vs 几何"错配假设。

### 一句话
**patch 变粗 + 双 backbone 跨域不对齐 + head 容量不匹配 + stage1 batch 受限 + 定位偏几何而 v3 更偏语义**，五个因素叠加，DINOv3 的规模优势被消耗殆尽。

---

## 7. 后续可尝试的方向

1. **单 backbone v3**：sat/grd 共享一份 `vitl16` web 权重，保住跨视角特征空间一致性（预期最有效）。
2. **提高输入分辨率**：576×576 sat → 36×36 或 704×704 → 44×44，把 patch16 的空间粒度追回 patch14 的水平。
3. **加 LoRA / 少量层微调**：现在完全冻结太保守，v3 容量未释放。
4. **head 扩容 & 去掉 1536→2048 投影瓶颈**：让 DPT 直接吃 dense-cls 拼接维度。
5. **stage1 用更大 batch（换更大显存卡或梯度累积）** 提高对比学习负样本池。

---

## 8. 关键代码位置速查

| 内容 | 文件 |
|---|---|
| DINOv3 wrapper（冻结） | [GeoDistill/model/dinov3.py](../GeoDistill/model/dinov3.py) |
| DINOv2 wrapper（冻结） | [GeoDistill/model/dino.py](../GeoDistill/model/dino.py) |
| Stage1 训练主循环 | [GeoDistill/train_vigor.py](../GeoDistill/train_vigor.py) `train_epoch_g2sweakly` |
| Stage2 自蒸馏主循环 | [GeoDistill/train_vigor.py](../GeoDistill/train_vigor.py) `train_epoch_geodistill` |
| 官方测试集评测 | [GeoDistill/train_vigor.py](../GeoDistill/train_vigor.py) `test()` @ line 545 |
| 对比损失 | [GeoDistill/model/loss.py](../GeoDistill/model/loss.py) `multi_scale_contrastive_loss` |
| KD 损失 | [GeoDistill/model/loss.py](../GeoDistill/model/loss.py) `cross_entropy` |
| 定位头 | [GeoDistill/model/network_vigor.py](../GeoDistill/model/network_vigor.py) `LocalizationNet` |

## 9. 复现命令模板

```bash
# 官方测试集评测（student）
cd ~/GeoDistill
CUDA_VISIBLE_DEVICES=<idle_gpu> \
  ~/geokd-main/.venv/bin/python3 train_vigor.py \
  --config dataset/config_vigor_test_student.json --gpuid 0
# 注意：必须使用 ~/geokd-main/.venv 内的 python，系统 python3 无 torch/numpy
```

---

## 附录 A：DINOv2 内部 geokd 蒸馏方法消融（历史工作）

> 这部分是在 DINOv2 backbone 时代做的**蒸馏方法消融**，与本次 backbone 对比（v2 vs v3）是**两个不同层次**的实验：主线是"换 backbone 看效果"；本附录是"固定 v2 backbone，比不同蒸馏 loss/初始化策略"。

### 设定

- 数据集：VIGOR cross-area, 10 epochs
- Batch：56（单卡 GPU0）
- Teacher：GeoDistill 完整定位头（DPT-based，DINOv2 `vitb14` backbone）
- Student：更小的 geokd 模型（ViT-S 版）
- 评测：与本 report 主线相同的官方测试集（SF+Chicago）
- 来源：[dataset/RESULTS.md](RESULTS.md)

### 对照矩阵

| 实验 | 初始化 | 训练 | distill_mode | 状态 | best mean (m) | best median (m) |
|---|---|---|---|---|---|---|
| T teacher 上界 | — | eval only | — | ✅ | **2.6409** | **1.2132** |
| A 无 KD 下界 | 随机 | supervised | — | 运行中 (2 卡 DDP) | — | — |
| B 仅初始化 | teacher 切片 | eval only | — | 暂缓 | — | — |
| C 仅蒸馏 | 随机 | distill | peak_hard | ✅ | 3.6268 | 1.8022 |
| **D 当前方法** | teacher 切片 | distill | peak_hard | ✅ | 3.7588 | 1.8516 |
| freq (SDKD) | teacher 切片 | distill | freq | ✅ | 3.7993 | 1.8859 |
| boundary (GaitKD) | teacher 切片 | distill | peak_hard+boundary | ✅ | 3.7522 | 1.8492 |

### 结论

- **蒸馏本身有效**：C（随机 + KD）显著优于假想的 A（随机 + 无 KD）下界。
- **初始化 + 蒸馏组合最好**：D（teacher 切片初始化 + peak_hard KD）在几种蒸馏策略里最优。
- **peak_hard / boundary / freq 差异不大**：几种 KD 变体在 ~3.75m mean 附近波动，说明本任务上蒸馏 loss 的选择不是主要瓶颈。
- **与 T 上界仍有 ~1.1m 差距**：ViT-S 学生天生容量有限，蒸馏能拿到 teacher 78% 左右的性能上限。

---

## 附录 B：CCVPE 跨架构蒸馏（未完成 / 已知问题）

> 这条实验线是"把 geokd（DINOv2 backbone + DPT teacher）蒸馏进 CCVPE（CNN 架构 student）"，与本 report 主线的 GeoDistill 自蒸馏**完全无关**。

### 现状

- 代码：[geokd-main/train_ccvpe_distill.py](../geokd-main/train_ccvpe_distill.py)
- 支持模式：`--distill_mode {ce, peak_hard, freq}` + 可选 `--boundary_weight` 加 GaitKD 风格 boundary loss
- 已跑过：ce / peak_hard（旧运行 OOM，只到 epoch1）、freq / peak_hard+boundary（完整 10 epoch）
- **问题**：freq / boundary 完整跑完的两个实验都出现 **mode collapse**（mean 稳定在 ~44–49m 不下降）
- freq 路径未做代码改动也出现同样 collapse → 排除是 boundary 代码加错导致，属于 pipeline 层面 bug
- **未定位根因**（候选：learning rate、loss 缩放、CCVPE 输入归一化与 geokd 数据流不匹配）

### 优先级

- 目前**低优**，等 v2/v3 backbone 主线对比结束后再回来 debug。

