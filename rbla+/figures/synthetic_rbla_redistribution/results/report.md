# 合成 RBLA 聚合流程实验

## 实验口径

预先固定 7 个设置，每个 30 个种子，合计 210 次配对聚合、7200 条 seed-rank 记录。主设置固定为 `shared_R32_eta0.3`，未按结果选择或丢弃种子。种子为 2026091100–2026091129。先通过小尺寸独立验证，再运行主设置与补充设置。

生成 → 按逐索引参与集合分别平均 B/A → 得到同一对全局因子 → 原始前缀和 SP+ → 相对于 X=BᵍAᵍ 测量。X 不是客户端稠密更新的加权平均。原始索引顺序保留；聚合后没有额外 gauge、旋转、置换或排序。SP+ 内部 SVD 排序属于方法本身。有效 LoRA scaling=1，所有误差使用未平方 Frobenius 范数。

参考 B₀/A₀ 元素分别为 N(0,1/m)、N(0,1/n)。每个参考块 F 的噪声为 N=Z‖F‖F/√|F|，Z 为独立标准高斯，因此 E‖ηN‖F²=η²‖F‖F²。η 表示合成因子差异程度，不表示真实 non-IID 程度。同一种子在不同 η 下复用参考和噪声流；不同种子独立。补充独立生成器直接生成相同逐元素尺度的独立客户端 B/A。

## 实际客户端预算下的主要结果

先在每个种子内按真实客户端 rᵢ、pᵢ 计算加权误差，再跨种子统计。下表为中位数；IQR 为第 25–75 百分位，不是置信区间。“总误差降低比例”是每个种子 (ΣpᵢE_pre−ΣpᵢE_SP+)/ΣpᵢE_pre 的跨种子中位数，不是所有扫描 rank 的平均，也不是中位数之差的比例。

| 设置 | 前缀相对误差 | SP+ 相对误差 | 前缀额外误差 | 配对 gain [IQR] | 总误差降低 |
|---|---:|---:|---:|---:|---:|
| shared_R32_eta0 | 0.61759 | 0.52739 | 0.09235 | 0.09235 [0.08805, 0.09425] | 14.96% |
| shared_R32_eta0.1 | 0.61919 | 0.52709 | 0.09385 | 0.09385 [0.08938, 0.09594] | 15.18% |
| shared_R32_eta0.3 | 0.62915 | 0.52556 | 0.10439 | 0.10439 [0.09982, 0.10706] | 16.56% |
| shared_R32_eta1 | 0.70922 | 0.47709 | 0.23213 | 0.23213 [0.22684, 0.23724] | 32.74% |
| independent_R32 | 0.86115 | 0.20682 | 0.65466 | 0.65466 [0.65092, 0.65687] | 75.98% |
| shared_R16_eta0.3 | 0.62917 | 0.55420 | 0.07544 | 0.07544 [0.07044, 0.08124] | 11.92% |
| shared_R64_eta0.3 | 0.63047 | 0.48639 | 0.14435 | 0.14435 [0.14205, 0.14608] | 22.87% |

主设置 30/30 个种子的客户端加权 gain 为正；范围 0.08893–0.11281。前缀的客户端加权归一化额外误差中位数为 0.10439，SP+ 的对应绝对值最大量级见数值验证。实际 rank 条件下的逐客户端 reduction 在前缀误差数值非零时才定义；r=R 的 reduction 为 NA，而 gain 在容差内为零。另存 conditional mean reduction 及其有效客户端权重，避免把 NA 当作零。

## 哪些 rank 改善较大

诊断扫描中，主设置的中位数 gain 最大位于 r=22 (r/R=0.6875)，gain=0.13992；实际客户端 rank 中最大位于 r=20，gain=0.13916。以下包括最小扫描 rank 及全部实际客户端 rank；完整扫描单独保存。

| r | 客户端数 | 前缀相对误差 | SP+ 相对误差 | gain 中位数 | reduction 中位数 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.98528 | 0.97140 | 0.01372 | 1.39% |
| 8 | 2 | 0.87030 | 0.78857 | 0.08171 | 9.39% |
| 12 | 2 | 0.79819 | 0.68981 | 0.10987 | 13.74% |
| 16 | 2 | 0.71762 | 0.59014 | 0.12894 | 18.00% |
| 20 | 1 | 0.62717 | 0.48841 | 0.13916 | 22.23% |
| 24 | 1 | 0.51545 | 0.37907 | 0.13708 | 26.53% |
| 28 | 1 | 0.37364 | 0.25316 | 0.12073 | 32.24% |
| 32 | 1 | 0.00000 | 2.013e-15 | -2.013e-15 | NA |

低 rank、较高 rank 以及满 rank 的改善应区分绝对归一化 gain 与相对降低比例；全 rank 时两种方法都保留完整更新，实际浮点差异完整保留。“前缀本来最优”以及重复奇异值控制例同样保留零收益。

## 稳定性与解释

- `shared_R32_eta0`: 30/30 个种子的客户端加权 gain > 0；gain 范围 [0.07888, 0.09982]，总误差降低范围 [13.00%, 15.94%]。
- `shared_R32_eta0.1`: 30/30 个种子的客户端加权 gain > 0；gain 范围 [0.08006, 0.10131]，总误差降低范围 [13.18%, 16.15%]。
- `shared_R32_eta0.3`: 30/30 个种子的客户端加权 gain > 0；gain 范围 [0.08893, 0.11281]，总误差降低范围 [14.36%, 17.73%]。
- `shared_R32_eta1`: 30/30 个种子的客户端加权 gain > 0；gain 范围 [0.20879, 0.24446]，总误差降低范围 [29.89%, 34.09%]。
- `independent_R32`: 30/30 个种子的客户端加权 gain > 0；gain 范围 [0.64482, 0.66339]，总误差降低范围 [75.11%, 76.80%]。
- `shared_R16_eta0.3`: 30/30 个种子的客户端加权 gain > 0；gain 范围 [0.05883, 0.10112]，总误差降低范围 [9.45%, 15.81%]。
- `shared_R64_eta0.3`: 30/30 个种子的客户端加权 gain > 0；gain 范围 [0.13655, 0.14874]，总误差降低范围 [21.83%, 23.46%]。

普通未排序高斯参考因子在 η=0 时也不保证其索引前缀为最佳低秩近似，因此 η=0 不应预设为零收益。独立客户端因子属于另一种生成机制，其收益大小不能直接外推到共享参考或训练检查点。参与客户端数随索引减少，后部因子平均的扰动方差可与前部不同；这只是当前生成机制的解释线索，本实验没有对 η 的单调关系做普遍性推断。

敏感性 rank 映射按基准比例四舍五入（half-up），R=16 为 [4,4,6,6,8,8,10,12,14,16]，R=64 为 [16,16,24,24,32,32,40,48,56,64]；其他尺寸、权重与 30 次重复保持一致。

## 数值验证

Float64；10 个小尺寸控制例、60 个 rank 比较全部通过。紧凑误差、实际稠密残差及独立 dense SVD 最优误差的最大绝对差为 4.720e-14，按 ‖B‖F‖A‖F 归一化后最大为 4.618e-16。逐索引聚合与独立掩码实现最大逐元素差 5.551e-17。

覆盖满 rank、相关列与零列的秩不足、已经最优的前缀、重复奇异值、完整更新为零但第一项前缀非零、全零因子、不可可靠归一化的近零更新及极小但可归一化更新。重复奇异值只验证误差最优，不要求截断矩阵逐元素一致。刻意损坏 SP+ 输出的测试成功被测量与验收逻辑检出。

主流程仅在 R×R 核心做一次 SVD。前缀残差以原始因子的 QR 尾部乘积计算；SP+ 误差以 [B,B⁺]、[Aᵀ,(A⁺)ᵀ] 的联合 QR 基下不超过 2R×2R 的实际残差计算，包含输出因子中可能出现的原空间外分量。所有目标 rank 共享同一次 SP+ 输出。E_opt 来自核心奇异值尾部范数，与实际 SP+ 残差分开计算。

令 S=‖B‖F‖A‖F、u=float64 eps。仅当 ‖X‖F>64uS 且大于最小正规浮点数时归一化；否则相对指标为 NA、保留绝对误差。验收绝对容差为 128uS+10⁻¹²‖X‖F。reduction 仅在 E_pre 大于此容差时报告。原始 CSV 不裁剪；仅绘图时容差内微小负额外误差显示为零。

全部生产数据通过数值验收。max |E_SP+−E_opt|/‖X‖F = 3.765e-15；r=R 的最大 SP+ 相对残差为 3.765e-15。原始 SP+ excess 的最小值为 -7.403e-16；其原始正负浮点值均保留。

## 代码来源与结论边界

见 [source_audit.md](../source_audit.md)。当前仓库没有 AGENTS.md、训练侧聚合、rank assignment 或 LoRA forward 实现。RBLA 按用户指定公式新实现并独立验证，与归档 YAML 的 mask-normalized padding 描述一致；无法声称已与缺失的训练聚合逐元素对照。compact SP+ 函数从现有图代码逐字冻结复用，未复用其旋转数据生成器或加 epsilon 的相对误差函数。

本实验支持：在这些固定生成设置下，对 RBLA 原始因子聚合产生的同一个完整更新，SP+ 在相同 rank 预算下减少前缀额外重建误差，并在数值精度内达到最优低秩重建误差。它不能证明解决了聚合偏置、rank–data mismatch、任务精度、收敛或公平性问题。这些仍需真实训练检查点的同更新配对测量及完整训练评估。

## 图与复现

主图：[PDF](figures/main_R32_eta0.3.pdf)、[SVG](figures/main_R32_eta0.3.svg)、[600 dpi PNG](figures/main_R32_eta0.3.png)。全部设置各有英文双子图；[完整图注](figures/captions.md)包含生成方式、配置、种子数和统计口径。图线是逐 rank 跨种子中位数和 IQR，markers 标识实际客户端 rank，纵轴按数据范围自动确定。

从项目根目录执行：

```powershell
python "rbla+/figures/synthetic_rbla_redistribution/run.py"
python "rbla+/figures/synthetic_rbla_redistribution/plot_figure.py"
```

完整文件与依赖说明见 [README.md](../README.md)。全局 Bᵍ、Aᵍ 与奇异值按设置存于 `factors/*.npz`；客户端因子由固定种子重建；每行含同一全局因子的 SHA-256。

## Paper-ready English summary

In a synthetic RBLA aggregation experiment, we generated heterogeneous-rank client LoRA factors from shared Gaussian reference factors with independent RMS-scaled perturbations (eta=0.3). We used 10 equally weighted clients, 512 x 512 updates, R=32, client ranks [8, 8, 12, 12, 16, 16, 20, 24, 28, 32], and 30 fixed independent random seeds. For each seed, RBLA prefix redistribution and SP+ were evaluated on the same single participation-normalized RBLA factor aggregation, without any additional post-aggregation gauge transformation and with effective LoRA scaling set to one. Using the unsquared Frobenius norm, the median client-weighted relative reconstruction error decreased from 0.6292 to 0.5256; the median paired normalized decrease was 0.1044 (IQR 0.0998-0.1071), corresponding to a 16.56% median reduction in client-weighted total reconstruction error. Across all 7 prespecified settings and 7,200 rank comparisons, the measured SP+ error differed from the optimal rank-r reference by at most 3.76e-15 after normalization by the full update norm. These results support improved rank-constrained reconstruction of synthetic RBLA aggregate updates; they do not establish improvements in task accuracy, convergence, fairness, aggregation bias, or rank-data mismatch.
