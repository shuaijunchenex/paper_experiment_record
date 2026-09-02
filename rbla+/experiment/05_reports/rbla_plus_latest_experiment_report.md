# RBLA+ 最新实验结果报告

## 1. 报告范围

本报告汇总 `src/test/experiment_results` 中 2026-07-28 至 2026-07-29 生成的最新一批实验结果。数据直接来自以下命名模式的 48 个 CSV 文件：

```text
{dataset}_dirichlet_0_4_rank_{assignment}_{method}_-train-202607*.csv
```

实验矩阵包括：

- 数据集：MNIST、FMNIST、KMNIST、QMNIST；
- 数据划分：Dirichlet non-IID，`alpha=0.4`；
- rank assignment：`correct`、`medium`、`mismatch`；
- 聚合方法：RBLA、RBLA+、SP、ZeroPadding；
- 实验总数：`4 × 3 × 4 = 48`。

文件名中的 `rbla_plus` 在运行配置内部对应 `sp_plus` strategy/aggregation，并启用了 canonicalization。本文继续使用实验文件中的名称“RBLA+”。

## 2. 实验设置

各组实验的主要公共设置如下：

| 配置项 | 取值 |
|---|---|
| 模型 | `simple_lora_mlp` |
| 训练轮数 | 100 rounds |
| 客户端数 | 10 |
| 每轮参与客户端 | 10 |
| 本地训练 | 5 epochs |
| 优化器 | SGD |
| 学习率 | 0.01 |
| Momentum | 0.0 |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Client-selection seed | 42 |

每个 CSV 包含 round 0 的初始评估以及 round 1–100 的训练评估，因此共有 101 条评估记录。除特别说明外，本文的“最终准确率”均指 round 100 的 accuracy。

## 3. 全部实验的最终准确率

| Dataset | Rank assignment | RBLA | RBLA+ | SP | ZeroPadding | RBLA+ SPMI_abs | RBLA+ SPMI_rel | 最优方法 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MNIST | correct | 81.32% | **82.42%** | 29.23% | 81.06% | 0.114736192 | 0.999297209 | RBLA+ |
| MNIST | medium | 81.19% | **82.65%** | 43.20% | 80.55% | 0.111449809 | 0.998803902 | RBLA+ |
| MNIST | mismatch | 80.38% | **82.18%** | 31.58% | 79.55% | 0.049542267 | 0.997962407 | RBLA+ |
| FMNIST | correct | **63.96%** | 63.92% | 39.13% | 62.25% | 0.228221034 | 0.999942352 | RBLA |
| FMNIST | medium | 51.86% | **52.08%** | 24.71% | 51.47% | 0.072003658 | 0.999772209 | RBLA+ |
| FMNIST | mismatch | 61.68% | **62.32%** | 41.80% | 60.93% | 0.026416421 | 0.999572815 | RBLA+ |
| KMNIST | correct | 42.99% | **44.15%** | 21.34% | 38.72% | 0.258226828 | 0.999862804 | RBLA+ |
| KMNIST | medium | 37.74% | **42.58%** | 18.59% | 36.07% | 0.121502485 | 0.999491749 | RBLA+ |
| KMNIST | mismatch | 36.03% | **42.71%** | 19.99% | 35.80% | 0.050863975 | 0.998856958 | RBLA+ |
| QMNIST | correct | 76.54% | **81.59%** | 48.39% | 77.24% | 0.233234657 | 0.999623018 | RBLA+ |
| QMNIST | medium | 77.23% | **81.01%** | 39.18% | 75.27% | 0.128456002 | 0.999035684 | RBLA+ |
| QMNIST | mismatch | 77.23% | **82.24%** | 47.97% | 77.11% | 0.021445695 | 0.997058677 | RBLA+ |

表中的 accuracy 是各完整训练在 round 100 的最终测试准确率；`SPMI_abs` 和 `SPMI_rel` 则是在相同 dataset/rank assignment 下，对 RBLA+ 第一轮聚合矩阵进行只读计算得到的结果，二者不处于同一训练时点。SPMI 原始数值及能量分解见 [`experiment_results/spmi_rbla_plus_matrix/summary.csv`](../../../experiment_results/spmi_rbla_plus_matrix/summary.csv)。

RBLA+ 在 12 个 dataset/rank 场景中的 11 个取得最高最终准确率。唯一例外是 FMNIST-correct：RBLA 为 63.96%，RBLA+ 为 63.92%，差距仅为 0.04 个百分点。

## 4. 聚合方法的整体表现

下表对每种方法的 12 个场景取平均：

| 方法 | 平均最终准确率 | 平均最佳准确率 | 全程平均准确率 | 平均最终 F1 | 平均最终 loss |
|---|---:|---:|---:|---:|---:|
| **RBLA+** | **66.65%** | **66.67%** | **48.25%** | **65.46%** | **0.9830** |
| RBLA | 64.01% | 64.03% | 46.75% | 61.84% | 1.0314 |
| ZeroPadding | 63.00% | 63.02% | 46.22% | 61.24% | 1.0469 |
| SP | 33.76% | 34.15% | 16.90% | 26.12% | 1.8962 |

RBLA+ 的平均最终准确率相对：

- RBLA 提高 2.64 个百分点，在 12 个场景中胜出 11 次；
- ZeroPadding 提高 3.65 个百分点，在 12 个场景中全部胜出；
- SP 提高 32.90 个百分点，在 12 个场景中全部胜出。

RBLA+ 相对 RBLA 提升最明显的场景是：

| Dataset | Rank assignment | RBLA+ − RBLA |
|---|---|---:|
| KMNIST | mismatch | +6.68 pp |
| QMNIST | correct | +5.05 pp |
| QMNIST | mismatch | +5.02 pp |
| KMNIST | medium | +4.84 pp |
| QMNIST | medium | +3.78 pp |

## 5. RBLA+ 在不同 rank assignment 下的表现

| Dataset | correct | medium | mismatch | 最大波动范围 |
|---|---:|---:|---:|---:|
| MNIST | 82.42% | **82.65%** | 82.18% | 0.47 pp |
| FMNIST | **63.92%** | 52.08% | 62.32% | 11.84 pp |
| KMNIST | **44.15%** | 42.58% | 42.71% | 1.57 pp |
| QMNIST | 81.59% | 81.01% | **82.24%** | 1.24 pp |

MNIST、KMNIST 和 QMNIST 上，RBLA+ 对三种 rank assignment 较为稳定。FMNIST-medium 是主要异常点，其最终准确率比 FMNIST-correct 低 11.84 个百分点，且该配置的最佳准确率为 52.18%，出现在 round 35，round 100 为 52.08%，说明它较早进入了低性能平台，而不是最后几轮偶然下降。

“correct”并未在所有数据集上取得最高准确率：

- MNIST 的 medium 略高于 correct；
- QMNIST 的 mismatch 高于 correct；
- 只有 FMNIST 和 KMNIST 明确表现为 correct 最优。

因此，本批结果不支持“volume-rank 越正相关，最终准确率必然越高”这一强结论。

## 6. Rank budget 混杂因素

每个数据集内部，correct 和 medium 使用相同的 rank-ratio 多重集，仅客户端分配顺序不同；mismatch 则使用了另一组 rank ratio，并具有更高的平均 rank budget：

| Dataset | correct/medium 平均 ratio | mismatch 平均 ratio | mismatch 增量 |
|---|---:|---:|---:|
| MNIST | 0.550 | 0.700 | +0.150 |
| FMNIST | 0.499 | 0.751 | +0.253 |
| KMNIST | 0.548 | 0.702 | +0.154 |
| QMNIST | 0.470 | 0.780 | +0.310 |

这意味着 correct 与 mismatch 的比较同时改变了：

1. 数据量与 rank 的相关关系；
2. 客户端总体 rank capacity；
3. LoRA 的 rank-dependent scaling；
4. 本地训练轨迹和最终聚合模型。

因此，本批实验适合比较同一配置下不同聚合方法的效果，但不能将 correct/mismatch 的准确率差异完全归因于 volume-rank alignment。

## 7. Canonicalization 数值稳定性

RBLA+ 的日志记录了 canonicalization diagnostics。全部 12 个 RBLA+ 实验、101 个评估点中的最大误差为：

| 诊断项 | 最大值 |
|---|---:|
| Core reconstruction error | `3.95e-6` |
| Factor balance error | `2.47e-6` |

误差量级较小，没有观察到明显的 canonicalization 数值失稳。因而 RBLA+ 的性能差异不能简单归因于 SVD/canonicalization 重构误差。

## 8. 主要结论

1. **RBLA+ 是本批实验中整体表现最好的聚合方法。** 它取得最高的平均最终准确率、平均 F1 和全程平均准确率，同时具有最低的平均最终 loss。
2. **RBLA+ 相对普通 RBLA 的优势主要出现在 KMNIST 和 QMNIST。** MNIST 上提升稳定但较小；FMNIST 上两者接近。
3. **RBLA+ 明显优于 ZeroPadding。** 12 个场景全部胜出，平均提高 3.65 个百分点。
4. **当前 SP 基线表现显著偏低。** 其平均最终准确率只有 33.76%，需要结合实现和 scaling 设置判断是否属于公平基线。
5. **RBLA+ 对 rank assignment 通常较鲁棒。** 除 FMNIST-medium 外，其余三个数据集的最大波动均不超过 1.57 个百分点。
6. **这批实验不能单独验证 rank-allocation 因果关系。** mismatch 的总体 rank budget 高于 correct/medium，是必须明确披露的实验混杂因素。

## 9. 局限与建议

- 当前每个配置只有 seed 42，无法报告方差或统计显著性；
- 应至少增加 3–5 个独立随机种子，并报告 mean ± std；
- 应使用完全相同的整数 rank-profile 多重集构造 correct、medium 和 mismatch；
- 建议优先复查 FMNIST-medium，确认数据划分、client-rank 映射及训练平台期；
- 建议检查 SP 与 RBLA+ 的 LoRA scaling 和广播语义是否完全可比；
- 后续报告应同时给出最终准确率、最佳准确率、全程平均准确率和通信/参数开销。

综合而言，本批结果为“RBLA+ 优于现有聚合基线”提供了较强的单 seed 实验证据；但关于 rank-volume alignment 的结论仍需要相同 rank budget 和多随机种子的受控实验支持。

## 10. 2026-08-13：CNN/NLP Dirichlet 0.4 rank-mismatch 实验

### 10.1 实验范围与完整性

本节补充提交 `1c26244` 中的最新一批结果。该批实验固定采用 Dirichlet `alpha=0.4` 和 `rank_mismatch`，覆盖 10 个数据集与 4 种聚合方法：

- CNN 数据集：CIFAR-10、CINIC-10、SVHN；
- NLP 数据集：AGNews、IMDB、MNLI、MRPC、QQP、RTE、SST-2；
- 聚合方法：RBLA、RBLA+、SP（FlexLoRA）、ZeroPadding（HetLoRA）；
- 实验规模：`10 × 4 = 40`；
- 执行状态：40 个任务全部成功，无失败任务；
- 所有 CSV 均包含 round 0–100，共 101 个有效评估点。

以下表格同时报告峰值准确率及第 100 轮准确率。单元格格式为“峰值准确率@峰值轮次 / 最终准确率”，单位为 `%`；加粗表示该数据集的最高峰值准确率。

### 10.2 完整结果

| Dataset | ZeroPadding / HetLoRA | RBLA | SP / FlexLoRA | RBLA+ |
|---|---:|---:|---:|---:|
| AGNews | 85.36@77 / 84.16 | 90.28@12 / 78.41 | 89.32@45 / 88.64 | **90.50@28 / 89.09** |
| CIFAR-10 | 78.22@92 / 77.76 | 57.63@4 / 35.38 | 78.53@95 / 78.19 | **79.21@67 / 78.74** |
| CINIC-10 | 68.46@74 / 67.09 | 42.10@5 / 16.53 | 68.33@93 / 68.06 | **69.25@91 / 68.95** |
| IMDB | 81.84@15 / 77.76 | **83.10@11 / 80.37** | 80.97@64 / 78.76 | 81.41@88 / 81.19 |
| MNLI | 43.53@79 / 43.34 | 57.80@7 / 37.74 | 46.32@79 / 45.85 | **60.02@46 / 58.30** |
| MRPC | 70.38@75 / 69.57 | 69.39@25 / 68.46 | **70.67@87 / 70.03** | 69.04@96 / 68.81 |
| QQP | 63.40@76 / 62.07 | 73.78@7 / 68.79 | 72.57@88 / 71.55 | **79.81@75 / 74.74** |
| RTE | 54.87@37 / 50.90 | 53.43@20 / 45.49 | 54.51@53 / 52.35 | **55.96@97 / 54.87** |
| SST-2 | 79.36@20 / 77.18 | **81.08@4** / 70.99 | 80.16@21 / **78.90** | **81.08@16** / 76.15 |
| SVHN | 90.44@74 / 89.51 | 90.35@99 / 89.21 | 91.99@80 / 91.59 | **93.45@35 / 92.34** |

### 10.3 方法级汇总

| 方法 | 平均峰值准确率 | 平均最终准确率 | 平均峰值至最终降幅 | 峰值获胜情况 | 最终轮获胜情况 |
|---|---:|---:|---:|---:|---:|
| ZeroPadding / HetLoRA | 71.59% | 69.93% | 1.65 pp | 0 | 0 |
| RBLA | 69.89% | 59.14% | 10.76 pp | 1 胜 + 1 并列 | 0 |
| SP / FlexLoRA | 73.34% | 72.39% | 0.94 pp | 1 胜 | 2 胜 |
| **RBLA+** | **75.97%** | **74.32%** | 1.65 pp | **7 胜 + 1 并列** | **8 胜** |

RBLA+ 相比 RBLA 的逐数据集峰值结果为 7 胜、1 平、2 负，平均峰值准确率提高 6.08 个百分点。第 100 轮比较则为 10 胜、0 负，平均最终准确率提高 15.18 个百分点。

按任务类型拆分后：

| 任务类型 | RBLA+ 平均峰值 | RBLA 平均峰值 | RBLA+ − RBLA | 峰值胜/平/负 |
|---|---:|---:|---:|---:|
| CNN（3 个数据集） | 80.64% | 63.36% | +17.28 pp | 3 / 0 / 0 |
| NLP（7 个数据集） | 73.97% | 72.69% | +1.28 pp | 4 / 1 / 2 |
| 全部（10 个数据集） | 75.97% | 69.89% | +6.08 pp | 7 / 1 / 2 |

### 10.4 结果解读

1. **RBLA+ 在该批 rank-mismatch 实验中整体最优。** 它同时取得最高的平均峰值准确率和平均最终准确率，并在 10 个数据集中的 7 个取得独立峰值第一、在 SST-2 与 RBLA 并列第一。
2. **RBLA+ 的优势主要集中在 CNN。** CIFAR-10、CINIC-10 和 SVHN 均由 RBLA+ 获胜；相对 RBLA 的平均峰值提升达到 17.28 个百分点。
3. **NLP 上的提升较温和且不是全面胜出。** RBLA+ 在 AGNews、MNLI、QQP 和 RTE 上取得最高峰值；IMDB 由 RBLA 获胜，MRPC 由 SP 获胜，SST-2 与 RBLA 峰值持平。
4. **普通 RBLA 存在明显的峰值后退化。** 其平均峰值至最终轮下降 10.76 个百分点，而 RBLA+ 的对应降幅只有 1.65 个百分点。这解释了为什么最终轮口径下 RBLA+ 相比 RBLA 的平均优势明显大于峰值口径。
5. **RBLA+ 并非在所有 mismatch 场景都严格最优。** IMDB 峰值落后 RBLA 1.68 个百分点，MRPC 峰值落后 SP 1.62 个百分点，表明论文中更适合表述为“总体性能和训练稳定性更好”，而不是“在每个任务上都最优”。

### 10.5 局限

- 本批每个配置只有一次运行，不能据此报告方差或统计显著性；
- 峰值准确率依赖 checkpoint/early-stopping 选择规则，论文中应同时披露最终准确率；
- 普通 RBLA 在多个任务上的后期退化会显著影响最终轮平均值，需要结合训练曲线说明；
- 最新 HEAD 中新增的 FLoRA 配置尚无对应结果，因此本节没有把 FLoRA 纳入比较。

原始 CSV 位于仓库的 `src/test/experiment_results`，批处理执行记录为 `src/test/batch_summary-20260810_110607-7dd05bce4cbc018e.log`。
