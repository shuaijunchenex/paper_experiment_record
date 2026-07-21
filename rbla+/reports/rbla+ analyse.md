# RBLA+ Analyse：完整验证矩阵、峰值准确率与稳定性分析

> 文档日期：2026-07-21  
> 最近更新：2026-07-21，新增 8 个 SGD optimizer-ablation 实验  
> 项目：RBLA heterogeneous-rank LoRA / Compact Global Canonicalization  
> 数据集：MNIST、KMNIST  
> 结果范围：第一阶段 48 个 Adam verification 任务；第二阶段 8 个 SGD 消融任务；全部成功  
> 核心问题：RBLA+ 是否能通过每轮全局 canonicalization 改善 heterogeneous-rank LoRA 聚合，以及这种改善能否稳定保持到训练结束。

## 1. 文档范围与结论

本报告分析 `usyd-learning-src/src/test/experiment_results/` 中 2026-07-20 至 2026-07-21 生成的完整 RBLA+ verification matrix：

```text
2 datasets × 3 non-IID distributions × 2 rank assignments × 4 methods
= 48 experiments
```

每个实验都保存 round 0--100，共 101 个服务器测试集评估点。批处理日志显示 48 个任务全部成功，没有失败或提前退出。

2026-07-21 追加 optimizer ablation：

```text
MNIST double imbalance
× 2 methods (RBLA, RBLA+)
× 2 rank assignments (correct, mismatch)
× 2 SGD momentum settings (0.0, 0.9)
= 8 experiments
```

第二阶段每个实验执行 30 个实际通信轮次，即 round 0--29。8 个任务全部成功。SGD 使用 `lr=0.01`，而第一阶段 Adam 使用 `lr=0.001`；因此第二阶段可以检测 optimizer recipe 是否影响退化，但不能把差异严格归因于 optimizer family 单一变量。

当前结果的主要结论是：

1. **固定训练 100 轮并使用最终模型时，RBLA+ 没有优于普通 RBLA。** RBLA+ 的平均最终准确率为 30.87%，RBLA 为 38.08%，平均落后 7.21 个百分点。
2. **RBLA+ 的前期峰值能力很强。** 在 12 个实验条件中，RBLA+ 有 8 个取得四种方法中最高的单轮准确率；其平均峰值为 44.71%，高于 RBLA 的 39.19%。
3. **RBLA+ 的核心问题是峰值无法保持。** RBLA+ 从平均峰值到最终轮平均回落 13.84 个百分点，普通 RBLA 只回落 1.12 个百分点。
4. **rank assignment mismatch 会显著放大退化。** mismatch 条件下 RBLA+ 最终准确率只有 14.89%，比 RBLA 低 10.28 个百分点；其中两个 double-imbalance mismatch 实验分别回落 36.94 和 35.95 个百分点。
5. **canonicalization 没有可测量的整体运行时间开销。** RBLA+ 平均耗时 1361.79 秒，RBLA 平均耗时 1363.73 秒，差异仅 -0.14%，属于运行噪声。
6. **SGD momentum 0.9 消除了 mismatch 条件下 RBLA+ 的早期峰值后崩塌。** round 29 达到 67.61%，比同长度 Adam 的 41.32% 高 26.29 个百分点，并略高于相同 SGD 设置下 RBLA 的 66.37%。但在 correct 条件下，SGD RBLA+ 为 83.36%，低于 Adam RBLA+ 的 89.62%，也低于 SGD RBLA 的 90.77%。这表明问题更可能是 Adam、canonicalization 和 rank mismatch 的交互，而不是 Adam 在所有条件下都更差。

因此，第一阶段的“RBLA+ 长期不稳定”结论需要细化：不稳定性不是所有优化器和 rank 条件下都必然出现。当前最强证据指向 **optimizer recipe × canonicalization × rank assignment** 的交互效应。完整 SGD 结果和更新后的判断见第 13--14 节。

## 2. 数据权威顺序与评价规则

### 2.1 数据权威顺序

本报告按以下顺序判断实验配置和结果：

1. 每个结果 CSV 第一行保存的 `Config` 和 `effective_yaml`；
2. 批处理成功日志；
3. verification YAML 与生成脚本；
4. 结果文件名。

主要结果目录：

```text
/Users/chensj/Documents/GitHub/usyd-learning-src/src/test/experiment_results/
```

实验矩阵生成脚本：

```text
/Users/chensj/Documents/GitHub/usyd-learning-src/src/test/rbla+/generate_verification_configs.py
```

批处理完成记录：

```text
/Users/chensj/Documents/GitHub/usyd-learning-src/src/test/batch_summary-20260720_130408-402cf2e2f9f362e5.log
```

### 2.2 指标定义

- **最终准确率**：round 100 的服务器测试集 accuracy。
- **最高准确率**：round 0--100 中测试集 accuracy 的最大值。
- **峰值轮次**：最高准确率第一次出现的 round。
- **最后 10 轮准确率**：round 91--100 的 accuracy 均值。
- **全程平均准确率**：101 个测试点 accuracy 的算术平均，用作归一化曲线面积的近似。
- **峰值回落**：最高准确率减去最终准确率。

表格中的紧凑格式为：

```text
最终准确率 / 最高准确率@峰值轮次
```

### 2.3 关于最高准确率的使用限制

最高准确率来自测试集，不是独立验证集。因此它可以用于诊断训练轨迹和稳定性，但不能直接作为正规的 checkpoint 选择结果。若希望报告“best checkpoint”，应增加独立 validation set，通过验证集选择轮次，再只在测试集评估一次。

### 2.4 单次运行限制

每个矩阵单元当前只有一次运行，没有多 seed、置信区间或显著性检验。配置包含客户端选择随机种子 42，框架也会重置部分随机数生成器，但单次运行仍不足以估计方法方差。因此本文中的“领先”“落后”和“退化”均描述当前结果矩阵，不应解释为总体统计显著性。

## 3. 方法定义

### 3.1 RBLA+

RBLA+ 使用 RBLA runner、client strategy、server strategy 和 RBLA aggregation，并在服务器每轮聚合后执行 Compact Global Canonicalization。

canonicalization 配置为：

```yaml
canonicalization:
  enabled: true
  start_round: 0
  interval: 1
  deterministic_sign: true
  svd_fallback: true
  log_diagnostics: true
  ordering: singular_value
  activation_chunk_size: 4096
  activation_fallback: true
```

对聚合后的 LoRA 因子执行 thin QR 和小型核心矩阵 SVD：

```text
B_tilde = Q_B R_B
A_tilde^T = Q_A R_A
K = R_B R_A^T
K = U diag(sigma) Vh

B_canonical = Q_B U diag(sqrt(sigma))
A_canonical = diag(sqrt(sigma)) Vh Q_A^T
```

奇异值按降序排列，使全局 slot 0 对应当前最强奇异方向。广播时，不同 rank 的客户端接收全局 canonical factors 的 leading prefix。

### 3.2 RBLA

普通 RBLA 使用与 RBLA+ 相同的 runner、client、server 和 RBLA aggregation，但关闭 canonicalization。它是判断 canonicalization 净效果的最重要直接基线。

### 3.3 SP

SP 使用项目已有的 SP runner、client、server 和 `aggregation_sp.yaml`，关闭 canonicalization。

### 3.4 ZeroPadding

ZeroPadding 使用 RBLA runner/client/server，但聚合方法改为 `aggregation_zeropadding.yaml`，关闭 canonicalization。

## 4. 实验设计

### 4.1 公共训练配置

| 项目 | 配置 |
|---|---|
| 数据集 | MNIST、KMNIST |
| 模型 | `simple_lora_mlp` / `nn_model_mnist_mlp_lora_canonical.yaml` |
| 客户端数 | 10 |
| 每轮参与客户端 | 10 |
| 服务器训练轮数配置 | 100 |
| 实际记录 | round 0--100，共 101 个训练后评估点 |
| 本地训练 | 每轮 5 epochs |
| 优化器 | Adam |
| 学习率 | 0.001 |
| weight decay | 0.0001 |
| batch size | 64 |
| shuffle | true |
| 客户端选择 | random，seed 42；10/10 客户端参与 |
| LoRA inference scaling | 0.5 |

### 4.2 数据分布

每个数据集测试三种 non-IID 分布：

1. `dirichlet`：Dirichlet alpha 0.1；
2. `double_imbalance`：标签分布和客户端样本量同时不均衡；
3. `extreme`：每个客户端主要或完全只包含单一类别，是最强 label skew 条件。

### 4.3 Rank 分配

对 `double_imbalance` 和 `extreme`：

```text
correct  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mismatch = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
```

对 Dirichlet 分布，根据客户端数据量构造专门的 correct/mismatch 映射：

```text
correct  = [0.12, 0.24, 0.27, 0.02, 0.19, 1.00, 0.56, 0.59, 0.29, 0.28]
mismatch = [0.59, 0.29, 0.28, 1.00, 0.56, 0.02, 0.19, 0.12, 0.24, 0.27]
```

## 5. 完整实验结果

以下所有数值均来自对应 CSV 的 101 个 accuracy 记录。粗体表示该实验条件下四种方法中最高的峰值准确率。

### 5.1 MNIST

| 数据分布 | Rank 分配 | RBLA+ 最终 / 最高@轮 | RBLA 最终 / 最高@轮 | SP 最终 / 最高@轮 | ZeroPadding 最终 / 最高@轮 |
|---|---|---:|---:|---:|---:|
| Dirichlet | correct | **59.43% / 68.13%@11** | 66.36% / 66.98%@89 | 53.06% / 65.15%@11 | 53.04% / 53.62%@15 |
| Dirichlet | mismatch | **17.46% / 34.47%@4** | 24.51% / 26.36%@66 | 10.28% / 12.98%@1 | 20.38% / 23.36%@36 |
| Double imbalance | correct | 88.53% / 90.43%@37 | **93.00% / 93.32%@93** | 88.70% / 88.79%@88 | 87.08% / 87.39%@91 |
| Double imbalance | mismatch | 26.50% / 63.44%@8 | 57.65% / 58.29%@90 | 39.25% / 63.83%@4 | **65.13% / 66.90%@44** |
| Extreme | correct | **11.68% / 21.87%@7** | 10.09% / 11.78%@0 | 11.64% / 13.58%@2 | 13.20% / 16.06%@17 |
| Extreme | mismatch | **9.08% / 19.91%@3** | 9.80% / 10.03%@0 | 10.91% / 13.27%@84 | 8.69% / 9.82%@8 |

### 5.2 KMNIST

| 数据分布 | Rank 分配 | RBLA+ 最终 / 最高@轮 | RBLA 最终 / 最高@轮 | SP 最终 / 最高@轮 | ZeroPadding 最终 / 最高@轮 |
|---|---|---:|---:|---:|---:|
| Dirichlet | correct | 37.25% / 49.55%@7 | **54.07% / 54.31%@95** | 33.66% / 45.24%@4 | 36.40% / 37.03%@69 |
| Dirichlet | mismatch | **12.81% / 32.48%@53** | 19.40% / 20.13%@69 | 10.00% / 18.40%@11 | 14.31% / 14.31%@100 |
| Double imbalance | correct | 72.65% / 74.38%@47 | **72.35% / 76.89%@55** | 71.79% / 72.78%@93 | 68.28% / 68.97%@97 |
| Double imbalance | mismatch | **12.25% / 48.20%@7** | 29.68% / 32.22%@80 | 10.37% / 38.75%@6 | 37.26% / 42.60%@32 |
| Extreme | correct | **11.53% / 13.54%@6** | 10.00% / 10.00%@0 | 9.70% / 12.81%@51 | 10.06% / 10.78%@1 |
| Extreme | mismatch | **11.24% / 20.14%@6** | 10.00% / 10.00%@1 | 10.35% / 11.59%@98 | 9.97% / 11.10%@1 |

### 5.3 每个条件的最终与峰值赢家

| 数据集 | 分布 | Rank | 最终准确率赢家 | 最高准确率赢家 |
|---|---|---|---|---|
| MNIST | Dirichlet | correct | RBLA，66.36% | RBLA+，68.13%@11 |
| MNIST | Dirichlet | mismatch | RBLA，24.51% | RBLA+，34.47%@4 |
| MNIST | Double imbalance | correct | RBLA，93.00% | RBLA，93.32%@93 |
| MNIST | Double imbalance | mismatch | ZeroPadding，65.13% | ZeroPadding，66.90%@44 |
| MNIST | Extreme | correct | ZeroPadding，13.20% | RBLA+，21.87%@7 |
| MNIST | Extreme | mismatch | SP，10.91% | RBLA+，19.91%@3 |
| KMNIST | Dirichlet | correct | RBLA，54.07% | RBLA，54.31%@95 |
| KMNIST | Dirichlet | mismatch | RBLA，19.40% | RBLA+，32.48%@53 |
| KMNIST | Double imbalance | correct | RBLA+，72.65% | RBLA，76.89%@55 |
| KMNIST | Double imbalance | mismatch | ZeroPadding，37.26% | RBLA+，48.20%@7 |
| KMNIST | Extreme | correct | RBLA+，11.53% | RBLA+，13.54%@6 |
| KMNIST | Extreme | mismatch | RBLA+，11.24% | RBLA+，20.14%@6 |

赢家计数：

| 方法 | 最终准确率第一 | 最高准确率第一 |
|---|---:|---:|
| RBLA+ | 3/12 | **8/12** |
| RBLA | **5/12** | 3/12 |
| SP | 1/12 | 0/12 |
| ZeroPadding | 3/12 | 1/12 |

### 5.4 各方法的绝对最高准确率

| 方法 | 绝对最高准确率 | 实验条件 | 峰值轮次 |
|---|---:|---|---:|
| RBLA | **93.32%** | MNIST / Double imbalance / correct | 93 |
| RBLA+ | 90.43% | MNIST / Double imbalance / correct | 37 |
| SP | 88.79% | MNIST / Double imbalance / correct | 88 |
| ZeroPadding | 87.39% | MNIST / Double imbalance / correct | 91 |

## 6. 聚合表现

### 6.1 全部 12 个实验条件

| 方法 | 平均最终准确率 | 平均最后10轮 | 全程平均准确率 | 平均最终 F1 | 平均最终 loss | 最终平均排名 |
|---|---:|---:|---:|---:|---:|---:|
| RBLA | **38.08%** | **38.07%** | **36.69%** | **30.79%** | 2.9881 | **1.92** |
| ZeroPadding | 35.32% | 34.97% | 33.75% | 28.41% | **2.0929** | 2.67 |
| RBLA+ | 30.87% | 30.68% | 33.54% | 24.43% | 2.3936 | 2.33 |
| SP | 29.98% | 30.06% | 31.93% | 23.60% | 2.6496 | 3.08 |

RBLA+ 按最终准确率排第三，落后 RBLA 和 ZeroPadding，仅略高于 SP。按全程平均准确率，RBLA+ 仍排第三，但与 ZeroPadding 接近，说明 RBLA+ 的前期高值对曲线平均产生了明显贡献。

### 6.2 按数据集

| 数据集 | 方法 | 最终准确率 | 最后10轮 | 全程平均准确率 |
|---|---|---:|---:|---:|
| MNIST | RBLA+ | 35.45% | 35.27% | 37.92% |
| MNIST | RBLA | **43.57%** | **43.37%** | **41.62%** |
| MNIST | SP | 35.64% | 35.62% | 37.76% |
| MNIST | ZeroPadding | 41.25% | 40.97% | 39.96% |
| KMNIST | RBLA+ | 26.29% | 26.08% | 29.17% |
| KMNIST | RBLA | **32.58%** | **32.77%** | **31.77%** |
| KMNIST | SP | 24.31% | 24.50% | 26.09% |
| KMNIST | ZeroPadding | 29.38% | 28.96% | 27.54% |

相对 RBLA：

- MNIST：RBLA+ 最终准确率低 8.12 个百分点；
- KMNIST：RBLA+ 最终准确率低 6.30 个百分点。

两个数据集方向一致，因此总体落后不是某个数据集单独造成的。

### 6.3 配对比较

每个方法都在完全相同的 12 个数据集/分布/rank 条件下进行配对。

| 对照 | RBLA+ 平均最终差值 | 中位差值 | 最差差值 | 最好差值 | 胜/负 |
|---|---:|---:|---:|---:|---:|
| RBLA | **-7.21pp** | -5.53pp | -31.15pp | +1.59pp | 4/8 |
| SP | +0.89pp | +1.36pp | -12.75pp | +7.18pp | 9/3 |
| ZeroPadding | -4.45pp | +0.62pp | -38.63pp | +6.39pp | 7/5 |

RBLA+ 对 ZeroPadding 虽然胜场较多，但在 double-imbalance mismatch 上出现巨大失败，导致平均结果显著落后。这说明只报告胜场会掩盖尾部风险。

## 7. 按实验类型分析

### 7.1 Correct 与 mismatch

| Rank 分配 | RBLA+ | RBLA | SP | ZeroPadding |
|---|---:|---:|---:|---:|
| correct 最终准确率 | 46.85% | **50.98%** | 44.76% | 44.68% |
| mismatch 最终准确率 | 14.89% | 25.17% | 15.19% | **25.96%** |
| mismatch - correct | **-31.96pp** | -25.81pp | -29.57pp | -18.72pp |

RBLA+ 在 correct 条件下比 RBLA 低 4.13 个百分点，在 mismatch 条件下差距扩大到 10.28 个百分点。其 correct-to-mismatch 降幅也是四种方法中最大，表明当前 canonical prefix 机制没有提升 rank/data mismatch 鲁棒性。

### 7.2 按数据分布

| 分布 | RBLA+ | RBLA | SP | ZeroPadding | RBLA+ - RBLA |
|---|---:|---:|---:|---:|---:|
| Dirichlet | 31.74% | **41.08%** | 26.75% | 31.03% | -9.35pp |
| Double imbalance | 49.98% | 63.17% | 52.53% | **64.44%** | -13.19pp |
| Extreme | **10.88%** | 9.97% | 10.65% | 10.48% | +0.91pp |

Extreme 条件下所有方法都约为 10%，接近十分类随机猜测。RBLA+ 的 0.91 个百分点优势不能视为有实际意义的学习改进。主要有意义的结果来自 Dirichlet 和 Double imbalance，而 RBLA+ 在这两类条件下都明显落后 RBLA。

## 8. 峰值与长期稳定性

### 8.1 方法级峰值保持能力

| 方法 | 平均最高准确率 | 平均最终准确率 | 平均峰值回落 | 峰值轮次中位数 |
|---|---:|---:|---:|---:|
| RBLA+ | **44.71%** | 30.87% | **13.84pp** | 7.0 |
| RBLA | 39.19% | **38.08%** | **1.12pp** | 67.5 |
| SP | 38.10% | 29.98% | 8.12pp | 11.0 |
| ZeroPadding | 36.83% | 35.32% | 1.51pp | 34.0 |

RBLA+ 的平均峰值是四种方法中最高的，但峰值通常很早出现，并在后续训练中丢失。RBLA 和 ZeroPadding 的峰值较晚，且最终模型基本保持峰值性能。

### 8.2 RBLA+ 每个条件的峰值回落

| 数据集 | 分布 | Rank | 最高准确率@轮 | 最终准确率 | 回落 |
|---|---|---|---:|---:|---:|
| MNIST | Dirichlet | correct | 68.13%@11 | 59.43% | 8.70pp |
| MNIST | Dirichlet | mismatch | 34.47%@4 | 17.46% | 17.01pp |
| MNIST | Double imbalance | correct | 90.43%@37 | 88.53% | 1.90pp |
| MNIST | Double imbalance | mismatch | 63.44%@8 | 26.50% | **36.94pp** |
| MNIST | Extreme | correct | 21.87%@7 | 11.68% | 10.19pp |
| MNIST | Extreme | mismatch | 19.91%@3 | 9.08% | 10.83pp |
| KMNIST | Dirichlet | correct | 49.55%@7 | 37.25% | 12.30pp |
| KMNIST | Dirichlet | mismatch | 32.48%@53 | 12.81% | 19.67pp |
| KMNIST | Double imbalance | correct | 74.38%@47 | 72.65% | 1.73pp |
| KMNIST | Double imbalance | mismatch | 48.20%@7 | 12.25% | **35.95pp** |
| KMNIST | Extreme | correct | 13.54%@6 | 11.53% | 2.01pp |
| KMNIST | Extreme | mismatch | 20.14%@6 | 11.24% | 8.90pp |

两个 `double_imbalance + mismatch` 条件的轨迹最值得优先诊断：

- MNIST：63.44%@round 8 → 26.50%@round 100；
- KMNIST：48.20%@round 7 → 12.25%@round 100。

这不是“收敛速度慢”，而是达到较好解之后发生持续性退化。

## 9. 运行时间与执行完整性

批处理日志记录 48/48 成功。每种方法 12 个任务的平均耗时为：

| 方法 | 平均耗时 | 标准差 | 最短 | 最长 |
|---|---:|---:|---:|---:|
| RBLA+ | 1361.79s | 8.15s | 1345.71s | 1372.39s |
| RBLA | 1363.73s | 11.24s | 1346.66s | 1390.46s |
| SP | 1362.22s | 6.95s | 1344.59s | 1371.76s |
| ZeroPadding | 1362.13s | 7.07s | 1347.03s | 1372.51s |

RBLA+ 比 RBLA 平均少 1.94 秒，即 -0.14%。在约 22.7 分钟的完整任务尺度上，这个差异属于噪声。当前 Compact Global Canonicalization 的计算成本相对本地训练可以忽略。

## 10. 机制解释与归因边界

### 10.1 与结果一致的可能机制

RBLA+ 每轮对聚合后的全局 LoRA 更新重新进行 SVD，并按当前奇异值降序定义 slot。低 rank 客户端只接收 leading prefix。以下机制与当前“高早期峰值、后期退化、mismatch 更严重”的现象一致：

1. **跨轮方向交换。** 相邻轮的奇异值接近或交叉时，slot 顺序可能改变，使相同本地参数位置在不同轮对应不同全局方向。
2. **近重复奇异值子空间旋转。** 确定性 sign fixing 只消除正负号自由度，不能唯一确定重复或近重复奇异值子空间内部的旋转。
3. **低 rank prefix 语义漂移。** 高 rank 客户端可以接收更多方向，低 rank 客户端只接收前几个 slot，因此更容易受 slot 排序与方向漂移影响。
4. **mismatch 条件放大反馈。** 数据量、标签覆盖和本地 rank 不匹配时，强方向可能由不同客户端群体轮流主导；每轮独立 canonicalization 可能进一步放大这种变化。
5. **每轮执行过于激进。** `start_round: 0`、`interval: 1` 意味着从第一轮开始不断重定义因子坐标系，没有 warm-up 或稳定窗口。

### 10.2 当前不能直接证明的内容

以上是与实验轨迹和实现机制一致的假设，不是已被日志直接证明的因果结论。虽然配置启用了 `log_diagnostics: true`，当前结果 CSV 只保存了标准评估指标，没有保存逐轮：

- effective rank；
- singular values / singular gaps；
- prefix energy；
- 跨轮方向相似度或 permutation；
- canonicalization reconstruction error；
- factor balance error。

原始远程 console diagnostics 也没有随本批结果一起保存，因此目前无法把某次 accuracy collapse 与某层具体的谱变化直接对齐。

### 10.3 可以暂时排除或降低优先级的解释

RBLA client 每轮重新构建并清空本地优化器，因此 canonical slot reorder 不会继承上一轮对应 slot 的旧 Adam momentum。由陈旧 optimizer state 单独导致退化的解释优先级较低。

同时，RBLA+ 和 RBLA 的总体运行时间几乎相同，性能差异不是因为 canonicalization 造成显著训练时长变化或任务失败。

## 11. 后续实验建议

### 11.1 第一优先级：canonicalization schedule 消融

保持其他条件不变，至少测试：

| 变体 | start_round | interval | 目的 |
|---|---:|---:|---|
| 当前设置 | 0 | 1 | 基线 |
| warm-up | 5 或 10 | 1 | 避免早期不稳定谱反复重排 |
| 稀疏 canonicalization | 0 | 5 | 减少坐标系变化频率 |
| warm-up + 稀疏 | 10 | 5 或 10 | 同时降低早期和长期方向切换 |
| one-shot | 5/10 | 大于总轮数 | 判断一次统一参考系是否已足够 |

优先在以下两个高信息量条件运行：

1. MNIST / Double imbalance / mismatch；
2. KMNIST / Double imbalance / mismatch。

### 11.2 第二优先级：跨轮方向连续性

不要只按当前轮奇异值独立排序。可测试：

1. 使用上一轮 canonical directions 与当前方向的绝对余弦相似度矩阵；
2. 通过 Hungarian matching 确定跨轮 permutation；
3. 匹配后再做 deterministic sign alignment；
4. 仅在新方向能量明显超过旧方向时允许 slot 替换；
5. 对近重复奇异值子空间使用 Procrustes alignment，而不是独立选择 SVD basis。

### 11.3 第三优先级：ordering 对照

测试现有 `activation_aware` ordering，并确保 calibration activations 来自全局一致、固定且与测试集隔离的数据。比较：

- singular-value ordering；
- activation-aware ordering；
- cross-round continuity ordering；
- continuity + activation-aware hybrid ordering。

### 11.4 必须增加的诊断记录

建议为每层、每轮输出独立 JSONL/CSV：

```text
round
layer
effective_rank
singular_values
singular_gap_min
prefix_energy_at_each_client_rank
reconstruction_error
factor_balance_error
direction_similarity_to_previous_round
matched_permutation
sign_flips
```

并将这些诊断与服务器 accuracy/loss 使用相同 round 编号保存，直接分析 collapse 前后 3--5 轮的变化。

### 11.5 多 seed 与正规 checkpoint 评价

在确定较有希望的 schedule/ordering 后：

1. 每个关键实验至少运行 3 个 seed，正式报告建议 5 个；
2. 固定并记录模型初始化、DataLoader generator、客户端选择和 CUDA 相关随机性；
3. 报告 mean ± standard deviation 或置信区间；
4. 增加独立 validation split；
5. 用 validation accuracy 选择 checkpoint，再报告对应 test accuracy；
6. 不应直接用本文的测试集最高准确率作为最终方法结果。

## 12. 第一阶段判断：Adam verification matrix

现有完整矩阵同时揭示了 RBLA+ 的潜力与缺陷：

- 潜力：12 个条件中 8 个取得最高峰值，平均峰值也高于所有基线；
- 缺陷：最终准确率平均比 RBLA 低 7.21 个百分点，峰值平均回落 13.84 个百分点；
- 风险集中点：rank/data mismatch，尤其 double imbalance mismatch；
- 工程成本：计算开销可以忽略，不是当前瓶颈；
- 研究重点：应从“是否做 canonicalization”转向“如何保证 canonical directions 的跨轮连续性，以及何时做 canonicalization”。

因此，下一阶段不应直接扩大数据集或模型规模，而应先在现有两个 mismatch 失败案例上完成 schedule、ordering 和跨轮 matching 消融。只有当峰值能够稳定保持到后期训练，RBLA+ 才能被视为对普通 RBLA 的有效改进。

## 13. SGD optimizer ablation：配置、结果与分析

### 13.1 研究问题与实验控制

这一阶段针对第一阶段最严重的 MNIST Double-imbalance mismatch 退化，回答：

> RBLA+ 在 Adam 下出现的峰值后崩塌，是否与本地优化器有关？

实验矩阵为：

| 变量 | 水平 |
|---|---|
| 方法 | RBLA、RBLA+ |
| 数据集 | MNIST |
| 数据分布 | Double imbalance |
| Rank 分配 | correct、mismatch |
| 优化器 | SGD |
| 学习率 | 0.01 |
| Momentum | 0.0、0.9 |
| Nesterov | false |
| Weight decay | 0.0001 |
| 本地训练 | 5 epochs，batch size 64 |
| 通信轮次 | 30 个实际轮次，round 0--29 |
| 其他配置 | 与第一阶段对应 Adam 实验一致 |

对应结果文件位于：

```text
/Users/chensj/Documents/GitHub/usyd-learning-src/src/test/experiment_results/
```

文件名前缀为：

```text
mnist_double_imbalance_{correct|mismatch}_{rbla|rbla_plus}_sgd_momentum{0|09}_
```

批处理日志：

```text
/Users/chensj/Documents/GitHub/usyd-learning-src/src/test/batch_summary-20260721_192508-ea44ebca5610fa77.log
```

8/8 任务成功，每个 CSV 都包含 30 行 round 记录。配置内嵌元数据确认 SGD `lr=0.01`、对应 momentum、`training_rounds=29` 和正确的 canonicalization 开关均生效。

需要强调：第一阶段 Adam 的学习率是 0.001，第二阶段 SGD 的学习率是 0.01。二者相差 10 倍，因此这是 optimizer recipe 消融，而不是只替换优化器名称的严格单变量实验。

### 13.2 完整 SGD 结果

| Rank | 方法 | Momentum | 最终准确率 | 最高准确率@轮 | 峰值回落 | 全程平均 | 最后10轮 | 最后10轮标准差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correct | RBLA+ | 0.0 | 54.36% | 54.36%@29 | 0.00pp | 28.75% | 42.76% | 6.65pp |
| correct | RBLA | 0.0 | 58.50% | 58.50%@29 | 0.00pp | 30.59% | 47.20% | 8.21pp |
| correct | RBLA+ | 0.9 | 83.36% | 83.84%@28 | 0.48pp | 75.58% | 82.67% | 0.64pp |
| correct | RBLA | 0.9 | **90.77%** | **90.83%@28** | 0.06pp | **79.78%** | **89.81%** | 0.81pp |
| mismatch | RBLA+ | 0.0 | **47.63%** | **47.63%@29** | 0.00pp | 29.63% | **44.80%** | 3.96pp |
| mismatch | RBLA | 0.0 | 45.53% | 45.71%@22 | 0.18pp | **29.77%** | 44.40% | 2.05pp |
| mismatch | RBLA+ | 0.9 | **67.61%** | **67.61%@29** | 0.00pp | **60.84%** | **66.29%** | 0.84pp |
| mismatch | RBLA | 0.9 | 66.37% | 66.37%@29 | 0.00pp | 59.49% | 65.36% | 0.63pp |

最后一轮的 loss 与 F1：

| Rank | 方法 | Momentum | 最终 loss | 最终 F1 |
|---|---|---:|---:|---:|
| correct | RBLA+ | 0.0 | 1.4753 | 49.11% |
| correct | RBLA | 0.0 | 1.3818 | 51.77% |
| correct | RBLA+ | 0.9 | 0.4890 | 81.11% |
| correct | RBLA | 0.9 | **0.2980** | **90.59%** |
| mismatch | RBLA+ | 0.0 | **1.7864** | **42.24%** |
| mismatch | RBLA | 0.0 | 1.8278 | 39.21% |
| mismatch | RBLA+ | 0.9 | **1.4064** | **60.81%** |
| mismatch | RBLA | 0.9 | 1.4253 | 59.41% |

### 13.3 训练轨迹

| Rank / 方法 / Momentum | r0 | r1 | r2 | r3 | r5 | r10 | r15 | r20 | r25 | r29 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| correct / RBLA+ / 0.0 | 12.42% | 15.16% | 16.72% | 17.46% | 19.17% | 22.94% | 24.70% | 32.55% | 42.83% | 54.36% |
| correct / RBLA / 0.0 | 12.42% | 15.21% | 16.81% | 17.54% | 19.38% | 23.26% | 25.02% | 34.80% | 49.11% | 58.50% |
| correct / RBLA+ / 0.9 | 23.16% | 41.39% | 54.97% | 62.17% | 72.93% | 79.50% | 80.90% | 82.22% | 83.00% | 83.36% |
| correct / RBLA / 0.9 | 23.16% | 40.58% | 53.91% | 61.80% | 73.17% | 82.83% | 86.21% | 88.61% | 90.41% | 90.77% |
| mismatch / RBLA+ / 0.0 | 12.42% | 15.16% | 16.73% | 17.43% | 19.12% | 22.16% | 23.40% | 35.44% | 47.10% | 47.63% |
| mismatch / RBLA / 0.0 | 12.42% | 15.20% | 16.76% | 17.47% | 19.24% | 22.22% | 24.20% | 38.50% | 44.73% | 45.53% |
| mismatch / RBLA+ / 0.9 | 26.05% | 38.81% | 43.41% | 52.84% | 60.58% | 62.36% | 63.01% | 65.32% | 66.41% | 67.61% |
| mismatch / RBLA / 0.9 | 26.05% | 38.31% | 40.64% | 48.83% | 58.35% | 60.50% | 62.30% | 64.64% | 65.50% | 66.37% |

SGD 的四条 RBLA+ 曲线均没有复现 Adam mismatch 中“早期达到高峰后持续下跌”的形态。Momentum 0.0 的主要问题是收敛慢：到 round 29 仍明显上升，不能把当前最终值解释为已收敛性能。Momentum 0.9 的曲线在约 round 10 后进入缓慢、稳定提升阶段。

### 13.4 Momentum 影响

下表为 momentum 0.9 减去 momentum 0.0：

| Rank | 方法 | 最终差值 | 峰值差值 | 全程平均差值 | 最后10轮差值 |
|---|---|---:|---:|---:|---:|
| correct | RBLA+ | +29.00pp | +29.48pp | +46.83pp | +39.91pp |
| correct | RBLA | +32.27pp | +32.33pp | +49.19pp | +42.61pp |
| mismatch | RBLA+ | +19.98pp | +19.98pp | +31.22pp | +21.49pp |
| mismatch | RBLA | +20.84pp | +20.66pp | +29.71pp | +20.96pp |

在固定 30 轮预算下，momentum 0.9 对收敛速度至关重要。由于客户端优化器每轮都会重建，这里的 momentum 作用发生在每轮 5 个 local epochs 内，不是跨通信轮次累计的服务器 momentum。

### 13.5 RBLA+ 相对 RBLA

下表为 RBLA+ 减去同 rank、同 momentum 的 RBLA：

| Rank | Momentum | 最终差值 | 峰值差值 | 全程平均差值 | 最后10轮差值 | RBLA+ 单轮胜出 |
|---|---:|---:|---:|---:|---:|---:|
| correct | 0.0 | -4.14pp | -4.14pp | -1.84pp | -4.45pp | 0/30 |
| correct | 0.9 | -7.41pp | -6.99pp | -4.21pp | -7.14pp | 3/30 |
| mismatch | 0.0 | +2.10pp | +1.92pp | -0.14pp | +0.40pp | 8/30 |
| mismatch | 0.9 | **+1.24pp** | **+1.24pp** | **+1.36pp** | **+0.93pp** | **29/30** |

这一结果呈现明确的 rank 条件交互：

- correct：canonicalization 持续损害性能，momentum 0.9 下最终低 7.41 个百分点；
- mismatch：canonicalization 略有收益，momentum 0.9 下除 round 0 完全相同外，其余 29 轮 RBLA+ 全部高于 RBLA。

因此不能把 SGD 下的结果概括为“RBLA+ 全面优于 RBLA”；更准确的表述是，SGD 使 canonicalization 在 mismatch 条件下从显著负效应转为小幅正效应，但 correct 条件仍有明显代价。

### 13.6 与 Adam 的同长度对照

为避免将 SGD 的 round 29 与 Adam 的 round 100 直接比较，本节只使用第一阶段对应 Adam CSV 的前 30 个评估点，即 round 0--29。

| Rank | 方法 | 优化器 | r29 准确率 | 前30轮峰值@轮 | 峰值回落 | 前30轮平均 | 最后10轮 |
|---|---|---|---:|---:|---:|---:|---:|
| correct | RBLA+ | Adam lr=0.001 | 89.62% | 89.62%@29 | 0.00pp | 85.01% | 88.25% |
| correct | RBLA | Adam lr=0.001 | 89.65% | 92.98%@11 | 3.33pp | 89.28% | 90.16% |
| mismatch | RBLA+ | Adam lr=0.001 | 41.32% | 63.44%@8 | **22.12pp** | 50.61% | 42.43% |
| mismatch | RBLA | Adam lr=0.001 | 49.63% | 49.93%@27 | 0.30pp | 43.71% | 48.56% |

SGD momentum 0.9 减去 Adam：

| Rank | 方法 | r29 差值 | 峰值差值 | 全程平均差值 | 最后10轮差值 |
|---|---|---:|---:|---:|---:|
| correct | RBLA+ | -6.26pp | -5.78pp | -9.43pp | -5.58pp |
| correct | RBLA | +1.12pp | -2.15pp | -9.49pp | -0.35pp |
| mismatch | RBLA+ | **+26.29pp** | +4.17pp | +10.24pp | **+23.86pp** |
| mismatch | RBLA | +16.74pp | +16.44pp | +15.78pp | +16.81pp |

最关键的 difference-in-differences 是 mismatch 条件下 RBLA+ 相对 RBLA 的变化：

```text
Adam:            RBLA+ - RBLA = 41.32% - 49.63% = -8.31pp
SGD momentum0.9: RBLA+ - RBLA = 67.61% - 66.37% = +1.24pp
相对效应变化：                              +9.55pp
```

这说明更换 optimizer recipe 不仅整体提高了 mismatch 性能，还专门逆转了 canonicalization 的相对效应。它强烈支持“Adam 与 mismatch canonicalization 存在不利交互”的假设。

但 correct 条件方向相反：

```text
Adam:            RBLA+ - RBLA = -0.03pp
SGD momentum0.9: RBLA+ - RBLA = -7.41pp
相对效应变化：                  -7.38pp
```

因此 SGD 不是无条件更适合 RBLA+；它解决了 mismatch 稳定性，同时增加了 correct 条件下 canonicalization 的性能代价。

### 13.7 Canonicalization 数值诊断

新 RBLA+ CSV 已保存每轮 canonicalization summary：

| Rank | Momentum | Mean effective rank：r0→r29 | 最大奇异值 r29 | 最大 core reconstruction error | 最大 factor balance error |
|---|---:|---:|---:|---:|---:|
| correct | 0.0 | 18.79 → 9.83 | 28.48 | 1.06e-6 | 1.51e-6 |
| correct | 0.9 | 18.46 → 8.04 | 53.46 | 2.40e-6 | 1.48e-6 |
| mismatch | 0.0 | 16.15 → 3.26 | 18.07 | 2.47e-6 | 1.55e-6 |
| mismatch | 0.9 | 9.91 → 6.79 | 37.62 | 2.31e-6 | 1.46e-6 |

core reconstruction error 和 factor balance error 始终约为 `1e-6`，没有出现与性能崩塌相符的数值爆炸。这说明 canonicalization 的 QR/SVD 重构本身在数值上稳定，第一阶段的退化不太可能由 SVD 计算误差直接造成。

所有设置的 mean effective rank 都下降，但准确率同时上升；尤其 mismatch momentum 0.0 的 effective rank 降到约 3.26，accuracy 仍升到 47.63%。因此 effective rank 下降本身不能作为失败判据。仍需记录逐层 singular gap 和跨轮 direction similarity，才能判断 slot 交换或子空间旋转。

### 13.8 运行时间

| 方法 | 4 个任务平均耗时 |
|---|---:|
| RBLA+ | 452.14s |
| RBLA | 441.49s |

RBLA+ 平均多 10.65 秒，约 2.41%。单任务差异方向并不完全一致，且实验在 Mac 上串行运行，因此只能认为 canonicalization 开销很小，不能把 2.41% 当作稳定性能估计。

## 14. 更新后的最终判断

### 14.1 是否是 Adam 优化器的问题？

当前证据支持以下结论：

> **Adam optimizer recipe 是 mismatch 条件下 RBLA+ 崩塌的重要促成因素，但现有实验还不能证明 Adam optimizer family 是唯一原因。**

支持依据：

1. Adam mismatch RBLA+ 在 round 8 达到 63.44%，到 round 29 降至 41.32%，回落 22.12 个百分点；
2. SGD momentum 0.9 mismatch RBLA+ 从 26.05% 基本单调升至 67.61%，没有峰值后崩塌；
3. RBLA+ 相对 RBLA 的效应从 Adam 下的 -8.31pp 变为 SGD momentum 0.9 下的 +1.24pp，净变化 +9.55pp；
4. canonicalization 数值误差保持在 `1e-6`，没有数值分解失败证据。

不能作唯一归因的原因：

1. Adam 使用 `lr=0.001`，SGD 使用 `lr=0.01`，同时改变了 optimizer family 和学习率；
2. 当前每个条件仍只有一个 seed；
3. SGD 只运行 30 轮，尚未证明继续到 100 轮仍不会退化；
4. 当前 summary diagnostics 没有跨轮方向连续性指标。

### 14.2 当前最合理的机制判断

第一阶段将问题主要归因于 canonical direction 的跨轮变化；第二阶段表明这个解释需要加入优化器交互：

- canonicalization 重定义低秩 prefix 的方向和顺序；
- 本地 optimizer 决定客户端围绕这些新 prefix 进行更新的步长和轨迹；
- Adam 的逐参数自适应缩放可能在 rank/data mismatch 下放大某些 canonical slots 的反馈；
- SGD momentum 0.9 在本批实验中产生更平滑、单调的 mismatch 轨迹；
- 但 correct 条件下 SGD RBLA+ 明显落后 RBLA，说明 canonicalization 仍可能丢失有用的客户端 slot 连续性。

“Adam 自适应缩放放大反馈”目前仍是机制假设，需要通过更新幅度、per-slot gradient/update norm 和 Adam second-moment 诊断验证。

### 14.3 下一步最小实验集

为严格判断是不是 Adam 本身，应优先运行：

1. **学习率交叉网格**：Adam 与 SGD 都测试 `0.001 / 0.003 / 0.01`；
2. **固定 momentum 对照**：SGD 重点保留 momentum 0.9，momentum 0.0 可延长轮次作为慢收敛控制；
3. **延长 SGD**：把 momentum 0.9 的 correct/mismatch RBLA/RBLA+ 延长到 100 轮，检查是否只是推迟崩塌；
4. **多 seed**：至少 3 个 seed，正式报告建议 5 个；
5. **优化器诊断**：逐层/逐 slot 保存 gradient norm、update norm；Adam 额外保存 first/second moment 范数；
6. **方向连续性诊断**：保存相邻轮 canonical direction cosine similarity、permutation 和 singular gap。

在这些实验完成前，论文中建议使用以下谨慎表述：

> 在 MNIST double-imbalance rank-mismatch 条件下，RBLA+ 的训练稳定性对本地 optimizer recipe 高度敏感。将 Adam `lr=0.001` 替换为 SGD `lr=0.01, momentum=0.9` 后，30 轮内的峰值后退化消失，且 RBLA+ 从落后 RBLA 8.31 个百分点变为领先 1.24 个百分点；该结果表明 optimizer 与 canonicalization 存在显著交互，但由于学习率同时变化且目前为单 seed，尚不能将改进唯一归因于 optimizer family。
