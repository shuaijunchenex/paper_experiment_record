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
