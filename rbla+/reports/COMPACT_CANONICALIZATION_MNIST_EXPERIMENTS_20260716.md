# Compact Global Canonicalization：MNIST 新增实验配置、结果与分析

> 文档日期：2026-07-16  
> 项目：RBLA heterogeneous-rank LoRA  
> 数据集：MNIST  
> 运行设备：CPU  
> 随机种子：单 seed，入口 `entries.lora` 调用 `TrainingUtils.set_seed_all(42)`  
> 目的：为后续网页 GPT 的独立分析保留完整、可追溯的实验上下文。

## 1. 文档范围与结论使用规则

本文覆盖本次 Compact Global Canonicalization 实现之后新增或重新运行的全部 MNIST 实验：

1. 原始参考结果 `mnist_rbla_-train-20260707_230243-f956a718.csv`；
2. 第一批 3 个 low-rank、SGD、1 local epoch 的 10-round canonicalization 试跑；
3. 第二批 3 个按原始参考超参重新对齐的 10-round canonicalization 实验；
4. 第三批 9 个 5-round SP、RBLA、canonicalization 与 SARA+canonicalization 对照。

总计纳入 1 个历史参考和 15 个新增结果 CSV。没有运行多 seed，没有置信区间，也没有显著性检验。因此本文中的“更高”“更低”只描述当前单次运行，不能解释为稳定的总体优势。

### 1.1 数据权威顺序

分析时按以下优先级判断配置：

1. 每个结果 CSV 第一行保存的 `Config`/`effective_yaml`；
2. 对应批处理成功日志；
3. 当前工作区中的 YAML；
4. 文件名。

早期 3 个 10-round 配置后来被原路径复用并修改过，所以不能用当前 YAML 反推早期运行。本文对早期实验一律以 CSV 内嵌 `effective_yaml` 为准。

### 1.2 Round 语义

RBLA、SP 和 SARA runner 都使用包含上界的循环：

```text
range(training_rounds + 1)
```

所以：

- `training_rounds: 9` 实际产生 round 0--9，共 10 行结果；
- `training_rounds: 4` 实际产生 round 0--4，共 5 行结果；
- round 0 已经过一次客户端本地训练、服务器聚合和全局评估，不是未经训练的初始化模型。

历史参考 CSV 的元数据写 `training_rounds: 50`，但文件中只有 round 0--9 十行。本文只把它当作“已记录的前 10 个 round”，不声称它完成了 50 round。

## 2. 方法定义

### 2.1 Vanilla RBLA

服务器对 heterogeneous-rank LoRA 的同名 rank slots 条件聚合，广播时按客户端本地 rank 截取全局因子 prefix。未开启 canonicalization 时，slot 位置继续沿用聚合前的客户端 slot 语义。

### 2.2 SP

使用项目已有的 SP runner/client/server strategy 与 `aggregation_sp.yaml`。这些实验使用 standard trainer，不启用 Compact Global Canonicalization。

### 2.3 RBLA + Compact Global Canonicalization（“新方法”）

对 RBLA 聚合后的每层因子

```text
A_tilde in R^(R x d_in)
B_tilde in R^(d_out x R)
```

执行 thin QR 和小型 `R x R` 核心矩阵 SVD：

```text
B_tilde = Q_B R_B
A_tilde^T = Q_A R_A
K = R_B R_A^T
K = U diag(sigma) Vh

B_canonical = Q_B U diag(sqrt(sigma))
A_canonical = diag(sqrt(sigma)) Vh Q_A^T
```

生产路径不显式构造 dense `B_tilde @ A_tilde`。canonicalization 位于 RBLA 聚合完成之后、全局状态保存和客户端 prefix 广播之前。开启后，slot 0 表示当前全局更新的最大奇异方向，低秩客户端接收全局最重要的 prefix。

本批实验的开关为：

```yaml
canonicalization:
  enabled: true
  start_round: 0
  interval: 1
  deterministic_sign: true
  svd_fallback: true
  log_diagnostics: true
```

确定性符号修复只消除单个奇异向量的正负号自由度；重复奇异值子空间仍可能存在正交旋转自由度。

### 2.4 SARA + Compact Global Canonicalization

SARA 客户端继续使用项目已有 alignment trainer，服务器仍使用 RBLA 聚合，再执行 Compact Global Canonicalization。SARA 参数未修改：

```yaml
sara:
  lambda_slot_0: 0.1
  lambda_slot_min: 0.01
  slot_weight_type: "1/s"
  lambda_sub_0: 0.1
  lambda_sub_min: 0.01
  beta: 0.01
  warmup_rounds: 5
  enable_rank_expansion: false
```

注意：5-round SARA 实验的 round 0--4 全部落在 `warmup_rounds: 5` 覆盖范围内，不能用它判断 warmup 结束后的行为。

### 2.5 Optimizer state 与 slot 重排

当前 RBLA 和 SARA 客户端每轮加载服务器状态后都会调用 optimizer builder 重建本地 optimizer，并在训练结束后清理。因此 canonical slot 重排不会继承上一轮旧 slot 的 Adam momentum/state。本批运行没有发现 optimizer state 与 canonical slot 错配。

## 3. 公共实验配置

### 3.1 模型与 LoRA scaling

所有实验使用 `simple_lora_mlp`：

```text
784 -> 200 -> 200 -> 10
LoRA base ranks: 160, 160, 100
```

该历史模型的 `lora_alpha` 来自默认 `lora_rank=4` 和 `lora_scaling=0.5`，即实际为常数 2，而不是随本地 rank 成比例。因此 `alpha/r` 会随 rank 改变。这个设计会严重混淆“rank 容量”和“LoRA 更新缩放”：低 rank 客户端的 `alpha/r` 更大，不能把 low-rank 与 full-rank 的性能差异单纯解释为容量差异。

### 3.2 Rank 组

#### Low-rank 组

```yaml
[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
```

对三个 LoRA 层的有效 ranks 为：

| 层 | 有效 rank 序列 |
|---|---|
| `fc1`、`fc2` | `[1,3,4,6,8,9,11,12,14,16]` |
| `fc3` | `[1,2,3,4,5,6,7,8,9,10]` |

这些 rank 都满足 `R <= min(d_in, d_out)`，所以 rank cap 是否开启不会改变 low-rank 组的实际 shape。

#### Full-ratio 组

```yaml
[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

历史 `simple_lora_mlp` 原始 shape：

| 层 | 原始有效 rank 序列 |
|---|---|
| `fc1`、`fc2` | `[16,32,48,64,80,96,112,128,144,160]` |
| `fc3` | `[10,20,30,40,50,60,70,80,90,100]` |

但是 `fc3` 只有 10 个输出维度，Compact Global Canonicalization 的 thin-QR 前提要求 `R <= 10`。为保证新方法合法，并让 SP/SARA 对照使用相同 shape，新配置设置：

```yaml
cap_lora_rank_to_matrix_dim: true
```

因此本次 full-ratio canonical/SP/SARA 实验实际为：

| 层 | 本次有效 rank 序列 |
|---|---|
| `fc1`、`fc2` | `[16,32,48,64,80,96,112,128,144,160]` |
| `fc3` | `[10,10,10,10,10,10,10,10,10,10]` |

这个 cap 默认关闭，不改变旧实验行为。历史参考 B0 未开启 cap，其输出层仍是过完备 rank 10--100。因此 B0 与新 full-ratio 实验不是仅差一个 canonicalization 开关的严格消融。

### 3.3 数据分布

| 分布 | 客户端样本数 | 总样本数 | 特征 |
|---|---|---:|---|
| Double imbalance | `[600,1266,2016,2873,3873,5073,6573,8573,11573,17580]` | 60000 | label support 和客户端数据量同时递增 |
| Dirichlet `alpha=0.1` | `[2092,4048,4574,409,3142,16400,9303,10544,4806,4682]` | 60000 | 极强 label/volume heterogeneity |
| Two-label balanced | 每客户端 5000 | 50000 | 每客户端两个相邻标签，volume 完全平衡 |

所有实验每轮选择全部 10 个客户端；虽然 client selector 配置为 random/seed 42，但 `number=10`，不存在客户端子采样差异。

### 3.4 训练超参分组

| 分组 | Optimizer | LR | Weight decay | Momentum/Nesterov | Local epochs | Batch | Rank ratios |
|---|---|---:|---:|---|---:|---:|---|
| B0 历史参考 | Adam | 0.001 | 0.0001 | 配置中保留 momentum 0.9，但 Adam 不使用 | 5 | 64 | 0.1--1.0，未 cap |
| A1--A3 初始试跑 | SGD | 0.005 | 0.01 | 0.9 / true | 1 | 64 | 0.01--0.10 |
| B1--B3 对齐重跑 | Adam | 0.001 | 0.0001 | Adam | 5 | 64 | 0.1--1.0，输出层 cap 10 |
| C1--C3 full 对照 | Adam | 0.001 | 0.0001 | Adam | 5 | 64 | 0.1--1.0，输出层 cap 10 |
| C4--C9 low-rank 对照 | Adam | 0.001 | 0.0001 | Adam | 5 | 64 | 0.01--0.10 |

## 4. 实验索引与文件映射

| ID | 方法 | 分布 | 实际 rounds | 配置文件 | 结果 CSV |
|---|---|---|---:|---|---|
| B0 | Vanilla RBLA 历史参考 | Double imbalance（由源配置追溯；旧 CSV 未内嵌分布） | 10 recorded | `src/test/rblasa_experiment/idea_verification/mnist_rbla.yaml` | `mnist_rbla_-train-20260707_230243-f956a718.csv` |
| A1 | RBLA + canonical，low-rank，SGD | Double imbalance | 10 | `debug/mnist_rbla_canonical_double_imbalance_10round.yaml`（后被修改） | `...223145-b836c77af1797c0c.csv` |
| A2 | RBLA + canonical，low-rank，SGD | Dirichlet 0.1 | 10 | `debug/mnist_rbla_canonical_dirichlet_01_10round.yaml`（后被修改） | `...223246-7548c22345a97748.csv` |
| A3 | RBLA + canonical，low-rank，SGD | Two-label balanced | 10 | `debug/mnist_rbla_canonical_two_label_balanced_10round.yaml`（后被修改） | `...223350-54f2ab823767e469.csv` |
| B1 | RBLA + canonical，full-ratio cap | Double imbalance | 10 | `debug/mnist_rbla_canonical_double_imbalance_10round.yaml` | `...224213-d4bdc0a6289bb572.csv` |
| B2 | RBLA + canonical，full-ratio cap | Dirichlet 0.1 | 10 | `debug/mnist_rbla_canonical_dirichlet_01_10round.yaml` | `...224513-773ab9a5aee5a0f2.csv` |
| B3 | RBLA + canonical，full-ratio cap | Two-label balanced | 10 | `debug/mnist_rbla_canonical_two_label_balanced_10round.yaml` | `...224805-e0f25e3a588601b7.csv` |
| C1 | SARA + canonical，full-ratio cap | Double imbalance | 5 | `debug/mnist_sara_canonical_double_imbalance_5round.yaml` | `...225938-5c61d06d9b23118a.csv` |
| C2 | SP，full-ratio cap | Double imbalance | 5 | `debug/mnist_sp_full_double_imbalance_5round.yaml` | `...230159-ba4ed75ee9ce3344.csv` |
| C3 | SP，full-ratio cap | Dirichlet 0.1 | 5 | `debug/mnist_sp_full_dirichlet_01_5round.yaml` | `...230331-adc354e0d8f197aa.csv` |
| C4 | Vanilla RBLA，low-rank | Double imbalance | 5 | `debug/mnist_rbla_lowrank_double_imbalance_5round.yaml` | `...230502-b1d22afce84c2001.csv` |
| C5 | Vanilla RBLA，low-rank | Dirichlet 0.1 | 5 | `debug/mnist_rbla_lowrank_dirichlet_01_5round.yaml` | `...230620-a080fd0ff95b3487.csv` |
| C6 | SP，low-rank | Double imbalance | 5 | `debug/mnist_sp_lowrank_double_imbalance_5round.yaml` | `...230734-740cec97eeb0f7d8.csv` |
| C7 | SP，low-rank | Dirichlet 0.1 | 5 | `debug/mnist_sp_lowrank_dirichlet_01_5round.yaml` | `...230851-79bc85124fa12f90.csv` |
| C8 | RBLA + canonical，low-rank | Double imbalance | 5 | `debug/mnist_rbla_canonical_lowrank_double_imbalance_5round.yaml` | `...231008-8994788906d53f7c.csv` |
| C9 | RBLA + canonical，low-rank | Dirichlet 0.1 | 5 | `debug/mnist_rbla_canonical_lowrank_dirichlet_01_5round.yaml` | `...231122-a3e9529f52b42896.csv` |

上表中的省略 CSV 文件名均位于 `src/test/experiment_results/`，完整名称见本文第 11 节。

## 5. 历史参考 B0

旧 CSV 未保存完整 `effective_yaml`。根据同名源配置追溯，它使用 vanilla RBLA、Double imbalance、Adam 1e-3、5 local epochs、full ratios 0.1--1.0，且没有输出层 rank cap。它记录了十个 round：

| Round | Accuracy | Loss | F1 | MCC |
|---:|---:|---:|---:|---:|
| 0 | 0.7087 | 0.9863 | 0.6618 | 0.6887 |
| 1 | 0.8205 | 0.5382 | 0.7864 | 0.8081 |
| 2 | 0.8538 | 0.4020 | 0.8398 | 0.8426 |
| 3 | 0.9041 | 0.3010 | 0.9021 | 0.8941 |
| 4 | 0.9203 | 0.2701 | 0.9190 | 0.9116 |
| 5 | 0.9228 | 0.2583 | 0.9216 | 0.9143 |
| 6 | 0.9249 | 0.2522 | 0.9239 | 0.9166 |
| 7 | 0.9305 | 0.2354 | 0.9294 | 0.9228 |
| 8 | 0.9247 | 0.2501 | 0.9238 | 0.9165 |
| 9 | 0.9309 | 0.2293 | 0.9299 | 0.9234 |

这个结果只能作为“旧代码路径性能参考”，不能作为 canonicalization 的严格关闭开关对照，因为新 full-ratio 实验额外启用了输出层 rank cap，运行设备也从配置中的 CUDA 改为当前可用 CPU。

## 6. 第一批 A1--A3：low-rank、SGD、1 local epoch 的 10-round 试跑

### 6.1 汇总

| ID | 分布 | 最终 Accuracy | 最佳 Accuracy（round） | 最终 Loss | 最终 F1 | 最终 MCC |
|---|---|---:|---:|---:|---:|---:|
| A1 | Double imbalance | 0.2159 | 0.2159（9） | 2.2684 | 0.1358 | 0.1598 |
| A2 | Dirichlet 0.1 | 0.3051 | 0.3051（9） | 2.0544 | 0.2435 | 0.2582 |
| A3 | Two-label balanced | 0.2315 | 0.2392（8） | 2.3004 | 0.1511 | 0.1694 |

这些实验训练非常不足，且超参与后续 Adam/5-local-epoch 实验不同。它们主要用于验证端到端代码可运行和 diagnostics 有限，不应作为新方法效果的正式性能结论。

### 6.2 A1：Double imbalance

| R | Acc. | Loss | F1 | MCC | Mean eff. rank | Max SV | Min SV | Core err. | Balance err. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .1232 | 2.2918 | .0580 | .0559 | 8.557 | .0204 | 4.15e-10 | 6.53e-7 | 7.50e-7 |
| 1 | .1489 | 2.2919 | .0737 | .0801 | 8.272 | .0188 | 3.23e-10 | 6.48e-7 | 8.31e-7 |
| 2 | .1650 | 2.2905 | .0837 | .0957 | 7.930 | .0174 | 2.52e-10 | 5.85e-7 | 8.05e-7 |
| 3 | .1754 | 2.2878 | .0953 | .1082 | 7.535 | .0163 | 1.96e-10 | 6.01e-7 | 6.27e-7 |
| 4 | .1839 | 2.2846 | .1041 | .1185 | 7.102 | .0177 | 1.53e-10 | 6.27e-7 | 4.29e-7 |
| 5 | .1879 | 2.2813 | .1090 | .1245 | 6.646 | .0207 | 1.19e-10 | 5.40e-7 | 9.43e-7 |
| 6 | .1921 | 2.2784 | .1138 | .1306 | 6.187 | .0254 | 9.27e-11 | 5.75e-7 | 1.07e-6 |
| 7 | .1997 | 2.2752 | .1210 | .1401 | 5.741 | .0353 | 7.22e-11 | 6.93e-7 | 8.79e-7 |
| 8 | .2064 | 2.2718 | .1266 | .1479 | 5.314 | .0508 | 5.63e-11 | 4.12e-7 | 7.03e-7 |
| 9 | .2159 | 2.2684 | .1358 | .1598 | 4.912 | .0742 | 4.38e-11 | 3.40e-7 | 6.86e-7 |

### 6.3 A2：Dirichlet 0.1

| R | Acc. | Loss | F1 | MCC | Mean eff. rank | Max SV | Min SV | Core err. | Balance err. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .1182 | 2.2924 | .0490 | .0376 | 8.712 | .0265 | 1.11e-10 | 6.04e-7 | 9.63e-7 |
| 1 | .1340 | 2.2931 | .0499 | .0512 | 8.388 | .0267 | 1.04e-10 | 9.14e-7 | 1.19e-6 |
| 2 | .1389 | 2.2924 | .0523 | .0571 | 7.556 | .0377 | 9.71e-11 | 9.80e-7 | 1.09e-6 |
| 3 | .1440 | 2.2894 | .0569 | .0643 | 6.341 | .0693 | 9.09e-11 | 4.76e-7 | 1.05e-6 |
| 4 | .1541 | 2.2839 | .0675 | .0785 | 5.249 | .1525 | 8.51e-11 | 3.78e-7 | 8.52e-7 |
| 5 | .1734 | 2.2737 | .0861 | .1062 | 4.464 | .3353 | 7.96e-11 | 4.17e-7 | 7.15e-7 |
| 6 | .2022 | 2.2502 | .1195 | .1496 | 3.882 | .7042 | 7.45e-11 | 5.97e-7 | 4.98e-7 |
| 7 | .2383 | 2.1998 | .1703 | .2031 | 3.409 | 1.3075 | 6.97e-11 | 2.69e-7 | 1.64e-6 |
| 8 | .2679 | 2.1278 | .2140 | .2309 | 3.044 | 2.0919 | 6.53e-11 | 2.35e-7 | 2.51e-7 |
| 9 | .3051 | 2.0544 | .2435 | .2582 | 2.932 | 2.8373 | 6.11e-11 | 3.47e-7 | 6.86e-7 |

### 6.4 A3：Two-label balanced

| R | Acc. | Loss | F1 | MCC | Mean eff. rank | Max SV | Min SV | Core err. | Balance err. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .1068 | 2.2890 | .0574 | .0174 | 6.823 | .0587 | 2.05e-10 | 5.69e-7 | 6.23e-7 |
| 1 | .1236 | 2.2851 | .0874 | .0371 | 5.396 | .1289 | 1.91e-10 | 5.11e-7 | 6.51e-7 |
| 2 | .1455 | 2.2828 | .1168 | .0625 | 3.803 | .2877 | 1.78e-10 | 5.39e-7 | 3.18e-7 |
| 3 | .1647 | 2.2814 | .1313 | .0884 | 2.802 | .4798 | 1.66e-10 | 4.20e-7 | 5.52e-7 |
| 4 | .1678 | 2.2813 | .1311 | .0954 | 2.290 | .6491 | 1.55e-10 | 1.90e-7 | 9.03e-7 |
| 5 | .1691 | 2.2820 | .1238 | .0961 | 2.044 | .9453 | 1.45e-10 | 4.53e-7 | 2.33e-7 |
| 6 | .1895 | 2.2811 | .1299 | .1145 | 2.008 | 1.3257 | 1.35e-10 | 1.81e-7 | 3.33e-7 |
| 7 | .2223 | 2.2905 | .1478 | .1498 | 2.113 | 1.6445 | 1.26e-10 | 2.77e-7 | 4.35e-7 |
| 8 | .2392 | 2.2992 | .1600 | .1698 | 2.125 | 2.0402 | 1.18e-10 | 2.62e-7 | 5.69e-7 |
| 9 | .2315 | 2.3004 | .1511 | .1694 | 2.258 | 2.3286 | 1.10e-10 | 3.14e-7 | 1.04e-6 |

## 7. 第二批 B1--B3：按参考超参对齐的 10-round canonicalization

### 7.1 汇总

| ID | 分布 | 最终 Accuracy | 最佳 Accuracy（round） | 最终 Loss | 最终 F1 | 最终 MCC |
|---|---|---:|---:|---:|---:|---:|
| B1 | Double imbalance | 0.8482 | 0.8523（8） | 0.4242 | 0.8193 | 0.8375 |
| B2 | Dirichlet 0.1 | 0.7740 | 0.7740（9） | 0.8714 | 0.7426 | 0.7618 |
| B3 | Two-label balanced | 0.4692 | 0.4692（9） | 1.6442 | 0.3724 | 0.4336 |

与 A1--A3 相比，最终 Accuracy 分别提高 63.23、46.89 和 23.77 个百分点。但 B 组同时改变了 optimizer、local epochs 和 rank 区间，因此这个差值不能归因于 canonicalization；它主要说明第一批低训练预算不适合作为性能基准。

### 7.2 B1：Double imbalance

| R | Acc. | Loss | F1 | MCC | Mean eff. rank | Max SV | Min SV | Core err. | Balance err. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .6598 | 1.1230 | .6112 | .6377 | 5.983 | 119.204 | 8.96e-5 | 1.56e-6 | 6.79e-7 |
| 1 | .7881 | .7230 | .7475 | .7733 | 7.261 | 125.852 | 5.64e-6 | 2.45e-6 | 8.73e-7 |
| 2 | .8054 | .6427 | .7630 | .7914 | 7.697 | 124.117 | 7.17e-7 | 1.96e-6 | 1.26e-6 |
| 3 | .8148 | .5771 | .7725 | .8014 | 7.951 | 120.697 | 7.62e-7 | 1.68e-6 | 1.22e-6 |
| 4 | .8244 | .5209 | .7845 | .8119 | 8.159 | 117.714 | 9.23e-7 | 1.45e-6 | 1.32e-6 |
| 5 | .8275 | .4907 | .7903 | .8151 | 8.330 | 114.872 | 6.60e-7 | 1.62e-6 | 1.15e-6 |
| 6 | .8375 | .4582 | .8043 | .8256 | 8.490 | 111.716 | 5.40e-7 | 1.26e-6 | 1.21e-6 |
| 7 | .8399 | .4497 | .8092 | .8283 | 8.615 | 108.448 | 6.90e-7 | 2.02e-6 | 1.54e-6 |
| 8 | .8523 | .4153 | .8271 | .8410 | 8.712 | 105.152 | 6.34e-7 | 1.59e-6 | 1.33e-6 |
| 9 | .8482 | .4242 | .8193 | .8375 | 8.784 | 101.984 | 4.95e-7 | 1.87e-6 | 1.54e-6 |

Accuracy 在 round 8 达到峰值，round 9 回落 0.41 个百分点；loss 同步从 0.4153 上升到 0.4242。单次小幅回落不足以判定发散。

### 7.3 B2：Dirichlet 0.1

| R | Acc. | Loss | F1 | MCC | Mean eff. rank | Max SV | Min SV | Core err. | Balance err. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .1785 | 2.0060 | .1139 | .1426 | 6.614 | 57.636 | 1.31e-4 | 1.47e-6 | 7.60e-7 |
| 1 | .5011 | 1.6345 | .4897 | .4873 | 7.953 | 67.179 | 2.00e-5 | 2.12e-6 | 8.86e-7 |
| 2 | .6479 | 1.3548 | .6347 | .6334 | 8.270 | 72.434 | 4.46e-6 | 2.74e-6 | 9.61e-7 |
| 3 | .7191 | 1.1594 | .6973 | .7046 | 8.475 | 75.561 | 1.23e-6 | 1.81e-6 | 1.12e-6 |
| 4 | .7420 | 1.0439 | .7160 | .7279 | 8.698 | 76.416 | 5.23e-7 | 1.82e-6 | 8.66e-7 |
| 5 | .7547 | .9818 | .7287 | .7422 | 8.853 | 76.806 | 3.96e-7 | 2.11e-6 | 1.56e-6 |
| 6 | .7639 | .9396 | .7357 | .7516 | 8.983 | 76.163 | 5.17e-7 | 1.57e-6 | 1.35e-6 |
| 7 | .7656 | .9218 | .7368 | .7533 | 9.068 | 75.380 | 4.58e-7 | 1.54e-6 | 1.47e-6 |
| 8 | .7723 | .8902 | .7421 | .7604 | 9.143 | 75.024 | 5.95e-7 | 1.63e-6 | 1.15e-6 |
| 9 | .7740 | .8714 | .7426 | .7618 | 9.182 | 74.029 | 3.08e-7 | 1.63e-6 | 1.52e-6 |

该曲线在十个 round 内 Accuracy 单调不降、loss 单调下降。round 4 之后仍在缓慢改善，没有显示已经完全收敛。

### 7.4 B3：Two-label balanced

| R | Acc. | Loss | F1 | MCC | Mean eff. rank | Max SV | Min SV | Core err. | Balance err. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .1548 | 2.2493 | .0620 | .0864 | 3.438 | 56.667 | 7.51e-5 | 1.91e-6 | 5.77e-7 |
| 1 | .2479 | 2.1181 | .1399 | .2023 | 4.129 | 62.745 | 7.55e-6 | 2.54e-6 | 8.80e-7 |
| 2 | .2993 | 2.0069 | .1845 | .2571 | 4.395 | 67.427 | 1.84e-6 | 2.50e-6 | 1.51e-6 |
| 3 | .3476 | 1.9219 | .2494 | .3041 | 4.535 | 72.296 | 1.19e-6 | 3.56e-6 | 7.70e-7 |
| 4 | .3733 | 1.8534 | .2809 | .3324 | 4.587 | 76.488 | 8.55e-7 | 2.60e-6 | 1.65e-6 |
| 5 | .3987 | 1.8000 | .3094 | .3598 | 4.592 | 80.749 | 5.03e-7 | 2.37e-6 | 1.38e-6 |
| 6 | .4295 | 1.7495 | .3407 | .3924 | 4.587 | 82.578 | 5.37e-7 | 2.72e-6 | 1.32e-6 |
| 7 | .4602 | 1.7090 | .3673 | .4250 | 4.579 | 84.625 | 6.39e-7 | 2.08e-6 | 6.85e-7 |
| 8 | .4667 | 1.6740 | .3720 | .4311 | 4.568 | 86.430 | 5.42e-7 | 1.84e-6 | 8.62e-7 |
| 9 | .4692 | 1.6442 | .3724 | .4336 | 4.551 | 87.987 | 5.85e-7 | 2.16e-6 | 1.17e-6 |

Two-label balanced 虽然每个客户端 volume 相同，但每个客户端只见两个标签，客户端目标冲突很强。在当前设置下，它比 Double imbalance 和 Dirichlet 0.1 更难，不能把“volume balanced”误解为“更接近 IID”。

## 8. 第三批 C1--C9：5-round 方法对照

### 8.1 总结果

| ID | 方法 / Rank | 分布 | 最终 Acc. | 最佳 Acc. | Loss | F1 | MCC |
|---|---|---|---:|---:|---:|---:|---:|
| C1 | SARA + canonical / full-cap | Double | .8264 | .8264 | .4875 | .7947 | .8132 |
| C2 | SP / full-cap | Double | .8148 | .8148 | .6292 | .7732 | .8018 |
| C3 | SP / full-cap | Dirichlet | .6749 | .6749 | 1.1725 | .6083 | .6585 |
| C4 | Vanilla RBLA / low | Double | .9049 | .9049 | .2913 | .9025 | .8950 |
| C5 | Vanilla RBLA / low | Dirichlet | .3898 | .4322（R2） | 3.1534 | .2760 | .3632 |
| C6 | SP / low | Double | .8503 | .8503 | .4442 | .8200 | .8405 |
| C7 | SP / low | Dirichlet | .7683 | .7683 | 1.1935 | .7305 | .7526 |
| C8 | RBLA + canonical / low | Double | .8692 | .8692 | .3609 | .8496 | .8593 |
| C9 | RBLA + canonical / low | Dirichlet | .7944 | .7944 | .7314 | .7809 | .7814 |

### 8.2 C1：SARA + canonical，full-cap，Double imbalance

| R | Acc. | Loss | F1 | MCC | Mean eff. rank | Max SV | Min SV | Core err. | Balance err. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .7397 | .9837 | .7048 | .7208 | 7.613 | 132.231 | 8.44e-4 | 1.17e-6 | 9.93e-7 |
| 1 | .7728 | .7186 | .7229 | .7565 | 8.043 | 138.629 | 3.98e-4 | 1.95e-6 | 1.18e-6 |
| 2 | .7980 | .6193 | .7552 | .7833 | 7.737 | 135.671 | 3.46e-4 | 2.95e-6 | 1.73e-6 |
| 3 | .8134 | .5431 | .7749 | .7998 | 7.800 | 130.922 | 3.23e-4 | 2.52e-6 | 1.25e-6 |
| 4 | .8264 | .4875 | .7947 | .8132 | 7.865 | 125.876 | 3.37e-4 | 2.41e-6 | 1.14e-6 |

### 8.3 C2--C3：SP full-cap

#### Double imbalance

| R | Acc. | Loss | F1 | MCC |
|---:|---:|---:|---:|---:|
| 0 | .6231 | 1.9980 | .5483 | .5989 |
| 1 | .7277 | .9432 | .6522 | .7097 |
| 2 | .7773 | .7509 | .7266 | .7615 |
| 3 | .8081 | .6684 | .7654 | .7944 |
| 4 | .8148 | .6292 | .7732 | .8018 |

#### Dirichlet 0.1

| R | Acc. | Loss | F1 | MCC |
|---:|---:|---:|---:|---:|
| 0 | .0914 | 2.2518 | .0206 | .0345 |
| 1 | .4373 | 1.8235 | .3684 | .4127 |
| 2 | .5705 | 1.4956 | .5145 | .5492 |
| 3 | .6636 | 1.2958 | .6000 | .6467 |
| 4 | .6749 | 1.1725 | .6083 | .6585 |

### 8.4 C4--C5：Vanilla RBLA low-rank

#### Double imbalance

| R | Acc. | Loss | F1 | MCC |
|---:|---:|---:|---:|---:|
| 0 | .6530 | 1.1905 | .5887 | .6307 |
| 1 | .8023 | .7390 | .7675 | .7896 |
| 2 | .8355 | .4928 | .8140 | .8243 |
| 3 | .8928 | .3263 | .8896 | .8823 |
| 4 | .9049 | .2913 | .9025 | .8950 |

#### Dirichlet 0.1

| R | Acc. | Loss | F1 | MCC |
|---:|---:|---:|---:|---:|
| 0 | .1838 | 1.9909 | .1268 | .1494 |
| 1 | .3998 | 1.9126 | .3058 | .3756 |
| 2 | .4322 | 2.1910 | .3314 | .4064 |
| 3 | .3970 | 2.7031 | .2924 | .3723 |
| 4 | .3898 | 3.1534 | .2760 | .3632 |

Dirichlet 下 vanilla RBLA 从 round 2 开始明显退化：Accuracy 从 .4322 降至 .3898，loss 从 2.1910 升至 3.1534。该现象是本批最显著的不稳定曲线。

### 8.5 C6--C7：SP low-rank

#### Double imbalance

| R | Acc. | Loss | F1 | MCC |
|---:|---:|---:|---:|---:|
| 0 | .5013 | 2.0197 | .4457 | .4804 |
| 1 | .7647 | .8077 | .7126 | .7490 |
| 2 | .8193 | .5958 | .7812 | .8078 |
| 3 | .8360 | .5047 | .7984 | .8254 |
| 4 | .8503 | .4442 | .8200 | .8405 |

#### Dirichlet 0.1

| R | Acc. | Loss | F1 | MCC |
|---:|---:|---:|---:|---:|
| 0 | .0892 | 2.2567 | .0164 | .0000 |
| 1 | .4766 | 1.8235 | .4249 | .4585 |
| 2 | .6849 | 1.4598 | .6398 | .6651 |
| 3 | .7333 | 1.2948 | .6936 | .7166 |
| 4 | .7683 | 1.1935 | .7305 | .7526 |

### 8.6 C8--C9：RBLA + canonical low-rank

#### Double imbalance

| R | Acc. | Loss | F1 | MCC | Mean eff. rank | Max SV | Min SV | Core err. | Balance err. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .6530 | 1.1905 | .5887 | .6307 | 5.553 | 15.366 | 5.46e-4 | 6.40e-7 | 5.44e-7 |
| 1 | .8111 | .5871 | .7715 | .7980 | 7.975 | 17.666 | .0291 | 5.01e-7 | 1.02e-6 |
| 2 | .8389 | .4630 | .8056 | .8282 | 9.330 | 19.009 | .0705 | 5.27e-7 | 9.39e-7 |
| 3 | .8578 | .3999 | .8321 | .8476 | 10.332 | 19.568 | .1224 | 7.30e-7 | 8.11e-7 |
| 4 | .8692 | .3609 | .8496 | .8593 | 10.955 | 19.635 | .1922 | 6.87e-7 | 1.14e-6 |

#### Dirichlet 0.1

| R | Acc. | Loss | F1 | MCC | Mean eff. rank | Max SV | Min SV | Core err. | Balance err. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .1838 | 1.9909 | .1268 | .1494 | 5.941 | 6.571 | 9.23e-4 | 6.12e-7 | 8.91e-7 |
| 1 | .6624 | 1.3289 | .6304 | .6455 | 7.964 | 8.882 | .0120 | 6.39e-7 | 8.45e-7 |
| 2 | .7375 | .9922 | .7125 | .7234 | 8.855 | 10.303 | .0388 | 8.19e-7 | 1.26e-6 |
| 3 | .7731 | .8308 | .7514 | .7615 | 9.473 | 11.219 | .0636 | 7.79e-7 | 1.04e-6 |
| 4 | .7944 | .7314 | .7809 | .7814 | 9.919 | 11.808 | .0654 | 6.55e-7 | 9.71e-7 |

## 9. 跨方法分析

### 9.1 最直接的 function-preservation 运行证据

C4/C8 和 C5/C9 是当前最干净的 canonicalization 开关对照：相同 seed、数据、optimizer、local epochs、rank shapes 和 RBLA 聚合，唯一区别是是否在聚合后 canonicalize。

在 round 0，vanilla 与 canonical 的全局测试结果几乎逐位一致：

| 分布 | Vanilla Acc./Loss | Canonical Acc./Loss | Loss 绝对差 |
|---|---|---|---:|
| Double | `.6530 / 1.1904876362` | `.6530 / 1.1904876227` | `1.34e-8` |
| Dirichlet | `.1838 / 1.9909420914` | `.1838 / 1.9909421532` | `6.18e-8` |

F1 和 MCC 也完全相同。这与设计目标一致：服务器完整 global `B @ A` 的函数在 canonicalization 前后保持不变。round 1 起曲线开始分离，是因为客户端收到的低秩 prefix 已从“旧 slot identity prefix”变成“按全局奇异值排序的最重要 prefix”。

这不是严格数学证明；数学正确性仍由单元测试中的 dense product 对比验证。但它是端到端训练路径上非常有价值的 smoke evidence。

### 9.2 Low-rank 的精确对照：Vanilla RBLA vs SP vs Canonical RBLA

#### Round 4 最终指标

| 分布 | 方法 | Accuracy | Loss | F1 | MCC |
|---|---|---:|---:|---:|---:|
| Double | Vanilla RBLA | **.9049** | **.2913** | **.9025** | **.8950** |
| Double | SP | .8503 | .4442 | .8200 | .8405 |
| Double | Canonical RBLA | .8692 | .3609 | .8496 | .8593 |
| Dirichlet | Vanilla RBLA | .3898 | 3.1534 | .2760 | .3632 |
| Dirichlet | SP | .7683 | 1.1935 | .7305 | .7526 |
| Dirichlet | Canonical RBLA | **.7944** | **.7314** | **.7809** | **.7814** |

#### Canonical 相对 Vanilla RBLA 的逐轮 Accuracy 差

| Round | Double | Dirichlet 0.1 |
|---:|---:|---:|
| 0 | .0000 | .0000 |
| 1 | +.0088 | +.2626 |
| 2 | +.0034 | +.3053 |
| 3 | -.0350 | +.3761 |
| 4 | -.0357 | +.4046 |

解释：

- 在 Double imbalance 下，canonicalization 前两轮略高，round 3 后低于 vanilla；最终低 3.57 个百分点。这里旧 slot prefix 并没有表现出明显失败，甚至最终最好。
- 在 Dirichlet 0.1 下，vanilla RBLA 在 round 2 后 loss 快速上升，而 canonical 曲线持续改善；最终 Accuracy 高 40.46 个百分点，loss 低 2.4220。
- 因此当前单 seed 证据不支持“canonicalization 总是提高性能”；更合理的描述是：它在当前 severe Dirichlet partition 中显著稳定了低秩客户端 prefix，但在 Double imbalance 中存在负收益。

#### Canonical 相对 SP 的最终差值

| 分布 | Accuracy | Loss | F1 | MCC |
|---|---:|---:|---:|---:|
| Double | +.0189 | -.0833 | +.0296 | +.0188 |
| Dirichlet | +.0261 | -.4620 | +.0503 | +.0288 |

在本次两个 low-rank 分布中，canonical RBLA 的最终指标都高于 SP，但差距远小于 Dirichlet 下 canonical 对 vanilla RBLA 的差距。仍需多 seed 确认 1.89--2.61 个百分点是否稳定。

### 9.3 Full-cap 的 5-round 方法对照

B1/B2 的前五个 round 可以与 C1--C3 比较，因为训练超参、模型 cap、rank ratios 和数据分布相同；区别是算法/训练器。

| 分布 | 方法 | Round-4 Acc. | Loss | F1 | MCC |
|---|---|---:|---:|---:|---:|
| Double | Canonical RBLA（B1） | .8244 | .5209 | .7845 | .8119 |
| Double | SARA + Canonical（C1） | **.8264** | **.4875** | **.7947** | **.8132** |
| Double | SP（C2） | .8148 | .6292 | .7732 | .8018 |
| Dirichlet | Canonical RBLA（B2） | **.7420** | **1.0439** | **.7160** | **.7279** |
| Dirichlet | SP（C3） | .6749 | 1.1725 | .6083 | .6585 |

观察：

- Double 下 SARA+canonical 与 standard canonical 非常接近，Accuracy 只高 0.20 个百分点；单 seed 不能说明 SARA 有稳定收益。
- SARA 的 loss 低 0.0334、F1 高 0.0101，但其五个 round 全在 warmup 内，不能外推到 post-warmup。
- Dirichlet 下 standard canonical 比 SP 高 6.71 个百分点，差距比 low-rank 对照更大。
- 没有运行“vanilla RBLA + full-cap + 相同 5 rounds”，因此 full-cap 表中缺少一个严格 vanilla 对照。

### 9.4 Low-rank 看起来优于 full-ratio，但不能解释为“小 rank 更好”

| 方法 | 分布 | Full-cap Round-4 | Low-rank Round-4 | Low - Full |
|---|---|---:|---:|---:|
| SP | Double | .8148 | .8503 | +.0355 |
| SP | Dirichlet | .6749 | .7683 | +.0934 |
| Canonical RBLA | Double | .8244 | .8692 | +.0448 |
| Canonical RBLA | Dirichlet | .7420 | .7944 | +.0524 |

这个结果反直觉，但当前模型存在强混杂：`lora_alpha=2` 基本固定，`alpha/r` 随 rank 减小而增大。low-rank 客户端不仅参数更少，其 LoRA 更新缩放也更大。此外 full-cap 组输出层所有客户端均为 rank 10，而 low-rank 输出层为 1--10，异构结构也不同。

要回答“rank 容量本身的效果”，应改用 rank-independent scaling 模型，例如令每层 `lora_alpha = local_rank * constant_scale`，并保持其他变量不变。

### 9.5 历史 B0 与 B1 的差距不能当作纯 canonicalization effect

在已记录十轮中：

| 指标 | B0 Vanilla 历史参考 | B1 Canonical full-cap | B1 - B0 |
|---|---:|---:|---:|
| Round-4 Accuracy | .9203 | .8244 | -.0959 |
| Round-9 Accuracy | .9309 | .8482 | -.0827 |
| Round-9 Loss | .2293 | .4242 | +.1949 |

B0 更高，但至少存在以下差异：

1. B0 的 `fc3` ranks 是 10--100，B1 被 cap 为统一 10；
2. B0 配置设备为 CUDA，B1 实际运行在 CPU；
3. 旧 CSV 未保存完整 effective YAML，分布信息由同名源配置追溯；
4. B0 没有 canonical diagnostics，无法验证其谱结构；
5. 两者不是在同一个预先冻结的初始 checkpoint 上做 paired run。

因此不能用 8.27 个百分点的最终差距声称 canonicalization 导致退化。需要补充 vanilla RBLA full-cap paired run 才能隔离开关效应。

## 10. Canonicalization 数值诊断分析

### 10.1 正确性与稳定性

所有 canonical 实验每轮均处理 3 个 LoRA layer，没有 NaN/Inf 或 SVD failure 被报告。全体新增运行中：

| 指标 | 最大观测值 | 出现位置 |
|---|---:|---|
| `core_reconstruction_error` | `3.564e-6` | B3，round 3 |
| `factor_balance_error` | `1.730e-6` | C1，round 2 |

这些误差与 FP32 QR/SVD 的数值尺度一致，支持以下实现性质：

- 小核心矩阵 SVD 重构正确；
- `B_canonical^T B_canonical` 与 `A_canonical A_canonical^T` 保持平衡；
- 训练中没有观察到由 canonicalization 数值失败导致的中断。

这两个诊断不是 dense `B @ A` function-preservation error。生产代码没有构造 dense update；dense product 只在单元测试中用于验证。

### 10.2 Effective rank 的不同轨迹

初始 SGD/1-epoch 试跑中，平均 effective rank 明显下降：

| 实验 | Round 0 | Round 9 | 方向 |
|---|---:|---:|---|
| A1 Double | 8.557 | 4.912 | 下降 |
| A2 Dirichlet | 8.712 | 2.932 | 大幅下降 |
| A3 Two-label | 6.823 | 2.258 | 大幅下降后略回升 |

同时 A2/A3 最大奇异值从约 0.03/0.06 增长到约 2.84/2.33，最小奇异值保持在 `1e-10` 量级。这表示能量逐步集中到少数方向，接近 rank-deficient，但 reconstruction/balance error 仍很小，所以不是 QR/SVD 数值崩溃。

Adam/5-epoch 对齐实验中，effective rank 大多上升或保持：

| 实验 | Round 0 | 最后 round |
|---|---:|---:|
| B1 Double full-cap | 5.983 | 8.784 |
| B2 Dirichlet full-cap | 6.614 | 9.182 |
| B3 Two-label full-cap | 3.438 | 4.551 |
| C1 SARA+canonical | 7.613 | 7.865 |
| C8 Double low | 5.553 | 10.955 |
| C9 Dirichlet low | 5.941 | 9.919 |

这说明谱轨迹强烈依赖 optimizer、训练预算、数据分布和本地更新，而不是 canonicalization 固有地令 rank 上升或下降。

注意：CSV 中的 mean effective rank 是三个层的汇总均值，maximum/minimum singular value 也是跨层汇总极值；不能从这些汇总值反推出某一具体层的完整谱。

### 10.3 为什么 canonical prefix 可能在 Dirichlet 下更重要

一个与数据一致、但仍需验证的解释是：severe Dirichlet partition 使客户端本地更新方向高度不一致。Vanilla RBLA 的 slot prefix 来自历史 slot identity，并不保证低 rank prefix 承载聚合后最大能量；canonicalization 每轮把全局奇异方向按重要性重排，使所有低秩客户端优先接收高能方向。

支持这个解释的现象包括：

1. round 0 全局函数一致，排除了 canonicalization 立即改变完整 global update；
2. round 1 广播后 Dirichlet 曲线迅速分离；
3. canonical 的 loss 持续下降，vanilla loss 在 round 2 后持续上升；
4. canonical low-rank 的 mean effective rank 从 5.94 上升到 9.92，前缀可用方向增加。

但 Double imbalance 下 vanilla 最终更高，说明上述机制不是唯一因素。可能还涉及 slot support、样本量权重、rank 与客户端数据量相关，以及 canonical 重排打破了某些对旧 slot identity 有利的局部连续性。

## 11. 关键混杂因素与未回答问题

### 11.1 单 seed

所有实验只运行 seed 42。2--5 个百分点的差异可能落在初始化/数据顺序波动内；即使 40 个百分点的 Dirichlet 差异很大，也需要至少 3--5 seeds 确认不是特定轨迹。

### 11.2 Rank 与客户端数据属性相关

rank ratios 按 client 1--10 单调增加。在 Double imbalance 中，客户端样本数也从 600 单调增加到 17580，且可见标签数同步增加。因此高 rank 与高 volume/高 label diversity 强相关。

这会同时影响：

- 数据量加权聚合；
- 高 rank tail slots 的参与客户端集合；
- canonical prefix 对不同客户端的价值；
- 方法对 tail support 的敏感性。

必须加入 rank permutation seeds，或至少打乱 rank-client mapping，才能把 rank 方法效应与客户端数据属性分离。

### 11.3 Rank-dependent LoRA scaling

`simple_lora_mlp` 的固定 alpha 使不同 rank 的 `alpha/r` 不一致。full 与 low 比较同时改变容量和更新缩放，应使用 rank-independent scaling 做确认实验。

### 11.4 输出层 rank cap

full-ratio 新实验的输出层被统一 cap 到 10；历史 B0 没有 cap。需要运行 vanilla RBLA full-cap，才能与 B1/B2/B3 做严格 paired 对照。

### 11.5 SARA 实验不足

C1 只有 Double imbalance、单 seed、5 rounds，并且全程处于 warmup。还缺少：

- 相同 full-cap shape 的 SARA without canonical；
- Dirichlet 0.1；
- 超过 warmup 的 10--20 rounds；
- 多 seed；
- repeated/near-repeated singular values 下 alignment reference 的稳定性诊断。

运行成功只能证明当前生命周期不存在直接 shape/optimizer 冲突，不能证明 SARA 与 canonicalization 理论上协同。

### 11.6 训练时长不足

5-round 曲线多数仍在上升，特别是 SP 和 canonical Dirichlet。最终 round 排名可能随训练延长改变。C5 已出现退化，因此延长时还应保存最佳 round，而不是只报告最后一行。

### 11.7 缺少 layer-wise diagnostics

当前 CSV 保存汇总 diagnostics。若要解释是 `fc1`、`fc2` 还是 `fc3` 驱动 effective rank/condition 变化，应另存每层 singular values、prefix energy 和 condition indicator；不要从跨层 max/min 过度推断。

## 12. 推荐的后续严格实验矩阵

按优先级建议：

1. **严格 canonical 开关消融**：Vanilla RBLA vs Canonical RBLA，固定 low-rank、相同初始化 checkpoint、Double/Dirichlet，各 5 seeds。
2. **full-cap vanilla 补齐**：为 B1/B2/B3 增加相同 cap 的 vanilla RBLA。
3. **rank permutation**：固定每个分布，至少 3 个 rank-client permutation；特别关注 Double imbalance。
4. **rank-independent scaling**：换用 `lora_alpha/local_rank` 恒定的模型，重复 full vs low 比较。
5. **SARA paired study**：SARA vs SARA+canonical，至少 15 rounds，使 warmup 后有 10 个有效 round。
6. **更长曲线**：对 C4--C9 延长到 10/20 rounds，报告 final、best 和 area-under-learning-curve。
7. **层级谱日志**：保存每层 singular values 和 `prefix_energy[k]`，直接检查客户端 rank `r_i` 捕获的能量。

## 13. 原始文件与运行日志

### 13.1 结果 CSV

全部位于 `src/test/experiment_results/`：

```text
mnist_rbla_-train-20260707_230243-f956a718.csv

mnist_rbla_canonical_double_imbalance_10round_-train-20260716_223145-b836c77af1797c0c.csv
mnist_rbla_canonical_dirichlet_01_10round_-train-20260716_223246-7548c22345a97748.csv
mnist_rbla_canonical_two_label_balanced_10round_-train-20260716_223350-54f2ab823767e469.csv

mnist_rbla_canonical_double_imbalance_10round_-train-20260716_224213-d4bdc0a6289bb572.csv
mnist_rbla_canonical_dirichlet_01_10round_-train-20260716_224513-773ab9a5aee5a0f2.csv
mnist_rbla_canonical_two_label_balanced_10round_-train-20260716_224805-e0f25e3a588601b7.csv

mnist_sara_canonical_double_imbalance_5round_-train-20260716_225938-5c61d06d9b23118a.csv
mnist_sp_full_double_imbalance_5round_-train-20260716_230159-ba4ed75ee9ce3344.csv
mnist_sp_full_dirichlet_01_5round_-train-20260716_230331-adc354e0d8f197aa.csv
mnist_rbla_lowrank_double_imbalance_5round_-train-20260716_230502-b1d22afce84c2001.csv
mnist_rbla_lowrank_dirichlet_01_5round_-train-20260716_230620-a080fd0ff95b3487.csv
mnist_sp_lowrank_double_imbalance_5round_-train-20260716_230734-740cec97eeb0f7d8.csv
mnist_sp_lowrank_dirichlet_01_5round_-train-20260716_230851-79bc85124fa12f90.csv
mnist_rbla_canonical_lowrank_double_imbalance_5round_-train-20260716_231008-8994788906d53f7c.csv
mnist_rbla_canonical_lowrank_dirichlet_01_5round_-train-20260716_231122-a3e9529f52b42896.csv
```

### 13.2 当前实验 YAML

10-round 配置：

```text
src/test/fedss_experiment/debug/mnist_rbla_canonical_double_imbalance_10round.yaml
src/test/fedss_experiment/debug/mnist_rbla_canonical_dirichlet_01_10round.yaml
src/test/fedss_experiment/debug/mnist_rbla_canonical_two_label_balanced_10round.yaml
```

5-round 配置：

```text
src/test/fedss_experiment/debug/mnist_sara_canonical_double_imbalance_5round.yaml
src/test/fedss_experiment/debug/mnist_sp_full_double_imbalance_5round.yaml
src/test/fedss_experiment/debug/mnist_sp_full_dirichlet_01_5round.yaml
src/test/fedss_experiment/debug/mnist_rbla_lowrank_double_imbalance_5round.yaml
src/test/fedss_experiment/debug/mnist_rbla_lowrank_dirichlet_01_5round.yaml
src/test/fedss_experiment/debug/mnist_sp_lowrank_double_imbalance_5round.yaml
src/test/fedss_experiment/debug/mnist_sp_lowrank_dirichlet_01_5round.yaml
src/test/fedss_experiment/debug/mnist_rbla_canonical_lowrank_double_imbalance_5round.yaml
src/test/fedss_experiment/debug/mnist_rbla_canonical_lowrank_dirichlet_01_5round.yaml
```

共享片段：

```text
src/yamls/aggregation/canonicalization_compact_enabled.yaml
src/yamls/general/general_rbla_10_communication_rounds.yaml
src/yamls/general/general_5_communication_rounds.yaml
src/yamls/nn_model/nn_model_mnist_mlp_lora_canonical.yaml
src/yamls/rank_distribution/het_rank_canonical_valid.yaml
src/yamls/rank_distribution/het_rank_full.yaml
src/yamls/optimizer/adam_1e3.yaml
src/yamls/training/training_epoch5.yaml
```

### 13.3 批处理日志

```text
src/test/batch_summary-20260716_223136-ee010ec5d4f6e719.log  # A1--A3: 3/3 success
src/test/batch_summary-20260716_224130-4d6158b7f9794fd7.log  # Windows GBK 日志编码失败，未进入训练
src/test/batch_summary-20260716_224203-4d6158b7f9794fd7.log  # B1--B3 UTF-8 重试: 3/3 success
src/test/batch_summary-20260716_225929-850c8f2f58baaae1.log  # C1--C9: 9/9 success
```

`224130` 的失败是父批处理进程打印 UTF-8 子进程输出时触发 `UnicodeEncodeError`，不是模型、数据或 canonicalization 失败。设置 `PYTHONIOENCODING=utf-8` 后原配置完整重跑成功。

## 14. 已执行验证

1. Canonicalization 单元测试：`21 passed`；
2. 九个新增 5-round 配置逐一做组合解析，确认 runner/client/server、aggregation、rank、trainer 和 canonicalization 开关；
3. 5-round 批处理：`9 success / 0 fail`；
4. 自动审计全部 9 个 CSV：每个恰有 `[0,1,2,3,4]` 五个 round；
5. 自动审计公共超参：Adam、lr、weight decay、local epochs、batch、客户端数量全部匹配；
6. 所有启用 canonicalization 的 5-round 结果每轮 `canonicalization_layer_count=3`；
7. `git diff --check` 通过，仅有工作区既有的 CRLF 提示。

没有运行完整测试套件，没有运行 MNIST 之外的数据集，也没有多 seed。

## 15. 给网页 GPT 的建议分析问题

将本文交给网页 GPT 后，建议明确要求它分别回答：

1. C4/C5 与 C8/C9 的 round-0 一致性是否支持 function preservation？
2. 为什么 canonical prefix 在 Dirichlet 0.1 下稳定 vanilla RBLA，却在 Double imbalance 下最终落后？
3. rank-data correlation、slot support 和 data-volume weighting 中哪一个最值得优先消融？
4. 固定 `lora_alpha=2` 导致的 rank-dependent scaling 能否解释 low-rank 优于 full-cap？
5. SARA+canonical 的 0.20 个百分点差异在单 seed/warmup 内是否有解释价值？
6. 应如何设计最小成本的 3--5 seed paired experiment，区分 canonicalization、rank cap、rank scaling 和 rank permutation 的影响？

网页 GPT 不应仅依据最终 Accuracy 排名得出“新方法全面优于基线”的结论。当前最强的、相对可信的现象是：**在这一次 low-rank Dirichlet 0.1 运行中，canonicalization 阻止了 vanilla RBLA round 2 后的明显退化；但同样设置在 Double imbalance 中没有超过 vanilla RBLA。**
