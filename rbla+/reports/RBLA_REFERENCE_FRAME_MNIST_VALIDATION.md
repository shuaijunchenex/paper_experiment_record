# RBLA 共享参考基：MNIST P0–P3 验证计划

## 1. 目标与范围

本阶段只回答一个核心问题：

> RBLA 的 rank-wise factor aggregation 是否因为客户端 LoRA latent rank reference frame 不一致而产生显著误差，并影响联邦训练稳定性？

实验暂时只使用 MNIST。实现采用全新的策略、聚合方法、训练器、模型和 YAML 名称，不改变现有 `rbla`、`sara`、`rblasa`、`ffalora` 的运行行为。

新注册入口如下：

| 变体 | Runner/Client/Server strategy | Aggregation method | Trainer |
|---|---|---|---|
| P1 Vanilla + diagnostics | `rbla_refdiag` | `rbla_refdiag` | `standard` |
| P2 Freeze-A | `rbla_freeze_a` | `rbla_freeze_a` | `standard` |
| P3 Strong-A | `rbla_strong_a` | `rbla_strong_a` | `rbla_strong_a` |

P0 是不需要数据训练的独立压力测试，因此注册为独立实验工具，而不是伪造一个联邦训练策略。

## 2. 公共 MNIST 实验控制

### 2.1 模型

使用新模型 `mnist_lora_reference_mlp`：

- 结构为 `784 → 200 → 200 → 10`；
- 最大有效 factor ranks 为 `160/160/100`；
- 客户端 rank ratio 为 `0.1–1.0`；
- 每层设置 `lora_alpha = local_rank × lora_scale`；
- 默认 `lora_scale=1.0`，所以所有客户端均有 `alpha/r=1`。

这样可以排除 heterogeneous rank 导致的 LoRA scaling 差异。原 `simple_lora_mlp` 不作修改。

### 2.2 数据分布

主实验使用 `mnist_noniid_two_label_balanced_10`：

- 10 个客户端；
- 每个客户端正好 5,000 个样本；
- 每个客户端包含两个相邻数字类别；
- 每个类别在两个客户端出现；
- 每个类别总计使用 5,000 个样本。

该分布保留明显 Non-IID，同时消除客户端数据量与 rank 的相关性。

额外使用 `mnist_noniid_double_imbalance_10` 检查方法在 label skew 和客户端数据量不平衡同时存在时的行为。该分布使用 MNIST 全部 60,000 个训练样本，但客户端样本数和可见类别数均逐步增加：client 1 只有 class 0 的 600 个样本，client 10 包含全部类别，共 17,580 个样本。因此 double-imbalance 结果不能只解释为 reference-frame 效应，必须与相同分布下的 Freeze-A/Strong-A 成对比较。

### 2.3 Rank 分配

异构秩主实验使用预打乱列表：

```yaml
[0.4, 0.9, 0.1, 0.7, 0.2, 1.0, 0.5, 0.8, 0.3, 0.6]
```

rank 不再随 client index 单调增加。确认实验应再增加至少两个 rank permutation seed；所有对照方法在同一 seed 下必须使用完全相同的 rank-client mapping。

新增两个同秩对照：

| Rank 设置 | ratio 列表 | ratio 总和 | 控制目的 |
|---|---|---:|---|
| heterogeneous | `[0.4,0.9,0.1,0.7,0.2,1.0,0.5,0.8,0.3,0.6]` | 5.5 | 待验证的 rank heterogeneity 条件 |
| uniform 0.55 | `[0.55] × 10` | 5.5 | 与 heterogeneous 保持总 rank-ratio budget 相同，只移除客户端间秩差异 |
| uniform 1.00 | `[1.0] × 10` | 10.0 | 所有客户端使用最大秩，作为无 rank shortage 的容量上界控制 |

uniform 0.55 是判断“性能变化来自异构性还是总容量”的关键对照；uniform 1.00 同时增加了总容量，不能单独用于归因 rank heterogeneity。

### 2.4 训练控制

- 所有 10 个客户端每轮全部参与；
- Adam，学习率 `1e-3`，weight decay `1e-4`；
- 每轮本地训练 5 epochs；
- 正式曲线 30 communication rounds；
- 所有方法使用 RBLA 的 NaN-mask conditional aggregation；
- P2/P3 不使用现有 FFA-LoRA aggregator，避免同时改变 tail slot weighting。

### 2.5 实验 YAML 与 runner 名称

文件采用 `dataset_method_distribution_rank[_lambda]` 命名，避免仅靠 P1/P2/P3 无法辨认控制变量。

| Runner 名称 | 实验 YAML | 方法 | 数据分布 | Rank |
|---|---|---|---|---|
| `refdiag-hetero` | `mnist_refdiag_two_label_balanced_rank_heterogeneous.yaml` | vanilla RBLA + diagnostics | two-label balanced | heterogeneous，sum=5.5 |
| `refdiag-uniform-055` | `mnist_refdiag_two_label_balanced_rank_uniform_0_55.yaml` | vanilla RBLA + diagnostics | two-label balanced | uniform 0.55，sum=5.5 |
| `refdiag-uniform-100` | `mnist_refdiag_two_label_balanced_rank_uniform_1_00.yaml` | vanilla RBLA + diagnostics | two-label balanced | uniform 1.00，sum=10 |
| `freeze-balanced` | `mnist_freeze_a_two_label_balanced_rank_heterogeneous.yaml` | Freeze-A | two-label balanced | heterogeneous |
| `strong-balanced` | `mnist_strong_a_two_label_balanced_rank_heterogeneous_lambda_0_10.yaml` | Strong-A，lambda=0.10 | two-label balanced | heterogeneous |
| `freeze-double` | `mnist_freeze_a_double_imbalance_rank_heterogeneous.yaml` | Freeze-A | double imbalance | heterogeneous |
| `strong-double` | `mnist_strong_a_double_imbalance_rank_heterogeneous_lambda_0_10.yaml` | Strong-A，lambda=0.10 | double imbalance | heterogeneous |

runner 分组：

- `rank-controls`：三个 RefDiag rank 对照；
- `balanced`：two-label balanced 下的五个实验；
- `double-imbalance`：Freeze-A 和 Strong-A 两个实验；
- `all`：按表中顺序运行全部七个实验；
- `p1/p2/p3`：分别兼容 `refdiag-hetero`、`freeze-balanced`、`strong-balanced` 的旧入口。

## 3. P0：等价重参数化压力测试

### 目的

直接验证 RBLA factor aggregation 对功能等价的客户端特定换基不具有不变性。P0 证明机制在数学和代码路径中确实存在，但不声称它一定是训练性能下降的主因。

### 步骤

1. 生成 ranks 为 `2/4/6` 的随机 LoRA factors；
2. 为每个客户端采样独立正交矩阵 `Q_i`；
3. 执行 `A_i' = Q_i^T A_i`、`B_i' = B_i Q_i`；
4. 比较每个客户端的 `B_i A_i`；
5. 比较直接 dense weighted mean；
6. 比较换基前后的 RBLA factor aggregation。

### 指标与通过条件

| 指标 | 预期 |
|---|---:|
| `per_client_function_error` | `< 1e-5` |
| `dense_update_invariance_error` | `< 1e-5` |
| `rbla_reparameterization_sensitivity` | `> 1e-3` |

### 执行

```bash
PYTHONPATH=src python src/test/rbla_reference_reparameterization.py
```

### 决策

- 前两个误差不够小：压力测试或换基实现错误，停止后续实验；
- RBLA sensitivity 接近零：当前测试没有构造出有效 gauge mismatch，应更换 seed/Q；
- 三项满足：进入 P1。

## 4. P1：Vanilla RBLA + reference diagnostics

### 目的

不改变本地目标和 RBLA 聚合，只观察真实 MNIST 联邦训练中：

1. A frame 是否逐轮漂移；
2. factor aggregation discrepancy 是否逐轮增大；
3. discrepancy 是否与后期准确率下降一致；
4. high-rank tail support 是否构成独立问题。

### 新增指标

| 指标 | 含义 |
|---|---|
| `ref_a_cos_drift` | A 对应行方向漂移 |
| `ref_a_norm_drift` | A 对应行尺度漂移 |
| `ref_a_prox_drift` | normalized A proximal distance |
| `ref_frame_transform_drift` | `A_i A_g^dagger - I` 的归一化范数 |
| `ref_frame_residual` | 客户端 A 离开 global prefix row-space 的程度 |
| `ref_agg_discrepancy` | RBLA factor aggregate 与同 slot 条件 dense aggregate 的相对误差 |
| `ref_tail_energy_ratio` | 聚合 LoRA 后半 rank slots 的能量占比 |
| `ref_slot_support_min/max` | slot 参与客户端数范围 |

### 4.1 诊断量的精确定义

记第 `l` 个 LoRA 层上客户端 `i` 的因子为

```text
A_i,l in R^(r_i,l × d_in),  B_i,l in R^(d_out × r_i,l),
DeltaW_i,l = B_i,l A_i,l.
```

服务器广播因子记为 `A_g,l`，归一化客户端权重为 `w_i`。比较 A 时只使用客户端实际拥有的 global prefix，即 slot `s < r_i,l`。以下指标先在 slot 内平均，再按客户端权重平均，最后对 LoRA 层等权平均。

#### A cosine drift

```text
D_cos = mean_layer sum_i w_i mean_s [1 - cos(a_i,l,s, a_g,l,s)]
```

它只测行向量方向变化，不直接响应纯尺度变化。0 表示方向完全一致；反向时单项最大为 2。

#### A norm drift

```text
D_norm = mean_layer sum_i w_i mean_s
         |log((||a_i,l,s|| + eps) / (||a_g,l,s|| + eps))|
```

对放大和缩小对称：客户端行范数变成 global 的 `c` 倍时贡献约为 `|log c|`。

#### A normalized proximal drift

```text
D_prox = mean_layer sum_i w_i mean_s
         ||a_i,l,s - a_g,l,s||^2 / (||a_g,l,s||^2 + eps)
```

这是训练完成后的 round-level 诊断量。它同时响应方向和尺度变化，并用 global slot 能量归一化，使不同层/slot 可比较。它与 Strong-A 本地 regularizer 使用相同的 slot 公式，但两者不要混淆：`ref_a_prox_drift` 是服务器对客户端更新的加权汇总；`strong_a_prox_loss` 是每个客户端训练 batch 中、尚未乘 `lambda_A` 的局部正则项。

#### Frame transform drift 与 residual

对每个客户端 prefix 求 ridge-stabilized 最小二乘变换：

```text
T_i,l = A_i,l A_g,l^T (A_g,l A_g,l^T + eps I)^(-1)
D_transform = ||T_i,l - I||_F / sqrt(r_i,l)
D_residual  = ||A_i,l - T_i,l A_g,l||_F / (||A_i,l||_F + eps)
```

`D_transform` 测量 global row frame 内部的旋转、混合和尺度变化；`D_residual` 测量无法由 global prefix row-space 线性解释的部分。transform 很大而 residual 很小通常表示主要是 frame 内换基，而不是离开原 row-space。

#### RBLA aggregation discrepancy

对 slot `s` 定义有该 slot 的客户端集合 `E_s={i | r_i,l>s}`，并在集合内重新归一化权重：

```text
alpha_i,s = w_i / sum_(j in E_s) w_j
a_bar_s = sum_(i in E_s) alpha_i,s a_i,s
b_bar_s = sum_(i in E_s) alpha_i,s b_i,s
DeltaW_factor = sum_s b_bar_s a_bar_s
DeltaW_direct = sum_s sum_(i in E_s) alpha_i,s b_i,s a_i,s
D_agg = ||DeltaW_factor - DeltaW_direct||_F
        / (||DeltaW_direct||_F + eps)
```

`ref_agg_discrepancy` 是各 LoRA 层 `D_agg` 的算术平均。`DeltaW_direct` 和 RBLA 使用完全相同的 slot eligibility/conditional weights，区别仅在于“先分别平均 A/B 再相乘”还是“先形成每个客户端的 outer product 再平均”。所以该指标隔离了 factor aggregation 引入的交叉客户端 cross terms；它不是普通、无条件 FedAvg dense update 的误差。

当所有 eligible 客户端共享相同 A slot 时，两种表达相等，discrepancy 应仅剩浮点误差。这也是 Freeze-A 实现检查的核心不变量。

#### Tail energy 与 slot support

每个聚合 slot 的能量 proxy 定义为：

```text
e_s = ||b_bar_s||_2 ||a_bar_s||_2
tail_ratio = sum_(s >= floor(r_max/2)) e_s / (sum_s e_s + eps)
support_s = |E_s|
```

`ref_tail_energy_ratio` 对层取平均；`ref_slot_support_min/max` 先取每层所有 slot 的最小/最大参与客户端数，再对层取平均。异构秩下 tail slots 通常 support 更低，因此该组指标用于区分 reference mismatch 与 rank-dependent participation bias。

### 分析

除逐轮 accuracy/loss 外，应报告：

- 最后 5 rounds 的平均 accuracy；
- best accuracy 到 final accuracy 的下降；
- `ref_agg_discrepancy` 的 round-wise 曲线；
- drift 与 discrepancy 的 Pearson/Spearman 相关；
- discrepancy 与下一轮 accuracy change 的相关；
- tail energy 与 accuracy change 的相关。

### 4.2 同秩控制实验

三个 RefDiag 实验使用相同模型、two-label balanced 数据、优化器和随机种子，只改变 rank ratios：

1. `refdiag-hetero` 与 `refdiag-uniform-055` 总 rank-ratio budget 都是 5.5。两者差异主要用于估计 rank heterogeneity、slot support 和 rank-dependent aggregation 的影响。
2. `refdiag-uniform-055` 与 `refdiag-uniform-100` 都没有客户端间秩差异，但后者总容量更高。两者差异用于检查 0.55 是否因容量不足而形成性能上限。
3. 同秩时所有 slot 的 support 应为 10，且 `ref_slot_support_min == ref_slot_support_max == 10`。若不成立，rank 注入或指标实现有误。
4. 同秩并不会自动让 `ref_agg_discrepancy` 为零：客户端 A 仍可在相同维数内使用不同 reference frames。它只消除了 heterogeneous slot eligibility，而 Freeze-A 才固定 A。

执行三个 rank 对照：

```bash
PYTHONPATH=src python src/test/rbla_reference_experiment/run_reference_experiments.py rank-controls
```

### 执行

```bash
PYTHONPATH=src python src/test/rbla_reference_experiment/run_reference_experiments.py refdiag-hetero
```

### 决策

- drift 和 discrepancy 均接近零：reference mismatch 在该 MNIST 设置中不明显，不应继续复杂 anchoring；
- drift 增长但 discrepancy 不增长：A/B covariance 可能抵消，frame drift 不是聚合误差主因；
- drift 与 discrepancy 同步增长：进入 P2；
- discrepancy 小但 tail energy/support 与性能下降明显相关：优先转向 participation-support bias，而不是 Strong-A。

## 5. P2：RBLA + Freeze-A

### 目的

通过 hard gauge fixing 进行最强的因果干预：所有客户端接收 global A prefix 后将 A 从 optimizer 中移除，只训练 B 和原本允许训练的非 A 参数。

聚合仍然使用 RBLA NaN-mask 规则，因此 P1 与 P2 的唯一关键差异是 A 是否允许本地漂移。

### 实现不变量

- optimizer 创建前设置所有 `lora_A.requires_grad=False`；
- 每轮训练后检查 `max(abs(A_after-A_anchor)) == 0`；
- 将结果记录为 `freeze_a_max_abs_change`；
- 不使用服务器端“取第一个 A”来假装冻结。

### 预期

| 指标 | 预期 |
|---|---|
| `ref_a_*_drift` | 约为 0 |
| `ref_agg_discrepancy` | 接近数值误差 |
| accuracy 稳定性 | 优于或不差于 P1 |
| 最终 accuracy 上限 | 可能低于可训练 Strong-A |

### 执行

```bash
PYTHONPATH=src python src/test/rbla_reference_experiment/run_reference_experiments.py freeze-balanced
```

### 因果判断

- Freeze-A 未使 discrepancy 接近零：实现或 metric 公式有错误；
- discrepancy 接近零且后期 accuracy 改善：reference mismatch 是性能瓶颈的强证据；
- discrepancy 接近零但 accuracy 不改善：mismatch 存在，但不是主要性能瓶颈；
- accuracy 下降但稳定性提高：固定随机 subspace 限制了表达能力，继续 P3；
- accuracy 和稳定性都下降：Strong-A 的研究价值显著降低，应先排查 scaling、base parameter 和数据分布。

## 6. P3：RBLA + Strong-A normalized proximal

### 目的

在保持 ordered reference frame 的同时允许 A 发生有限、受控的任务适应，测试它是否能超过 Freeze-A 的性能上限。

本地目标为：

```text
L = L_task + lambda_A * mean_layer(mean_slot(
      ||A_i[s]-A_g[s]||^2 / (||A_g[s]||^2 + eps)
    ))
```

先按 slot 平均，再按 layer 平均，使同一 `lambda_A` 不依赖模型层数。

### Lambda 筛选

已提供：

- `strong_a_003.yaml`：`lambda_A=0.03`
- `strong_a_010.yaml`：`lambda_A=0.10`，默认实验配置
- `strong_a_030.yaml`：`lambda_A=0.30`

推荐流程：

1. 单 seed、10 rounds 快速筛选三个 lambda；
2. 排除明显欠约束或过约束值；
3. 对最佳值运行 30 rounds；
4. 最终至少运行 3 个训练 seed × 3 个 rank assignment seed。

### 关键指标

- P1/P2 的全部 diagnostics；
- `strong_a_prox_loss`；
- A adaptation ratio；
- final accuracy；
- last-5-round mean accuracy；
- seed standard deviation；
- 与 Freeze-A 的性能差；
- 与 P1 的 discrepancy 降幅。

### 执行默认值

```bash
PYTHONPATH=src python src/test/rbla_reference_experiment/run_reference_experiments.py strong-balanced
```

若要测试 `0.03` 或 `0.30`，复制 P3 主配置并将 `strong_a_010.yaml/strong_a_010` 替换为对应 YAML 和 alias，避免覆盖默认结果。

### 成功条件

Strong-A 需要同时满足：

1. `ref_agg_discrepancy` 明显低于 P1；
2. A adaptation 非零，证明不是退化的 Freeze-A；
3. final 或 last-5 accuracy 高于 Freeze-A；
4. 不同 seed 下改善方向一致；
5. drift 降幅与性能改善具有一致机制关系。

若只满足 1 而不满足 3，说明 soft anchoring 解决了 factor mismatch，但当前任务性能主要受其他因素限制。

### 6.1 Double-imbalance robustness pair

新增的 `freeze-double` 和 `strong-double` 使用完全相同的 double-imbalance 数据矩阵、异构 rank-client mapping、优化器和训练轮数，唯一方法差异是 hard Freeze-A 与 soft Strong-A。

实验目的：

- 检查 Strong-A 在客户端样本权重高度不均衡时是否仍能控制 `D_prox` 和 `D_agg`；
- 判断允许 A 适应能否缓解 Freeze-A 固定随机 subspace 的容量限制；
- 观察低 rank、高 rank 与大/小客户端叠加后，tail support 和 dense update 是否被少数大客户端主导；
- 避免把 balanced two-label 分布上的结论直接外推到更复杂的双重不平衡场景。

主要比较量是 Strong-A 相对 Freeze-A 的 final/last-5 accuracy、`f1_score`、`ref_a_prox_drift`、`ref_agg_discrepancy` 和 tail energy。由于没有新增 double-imbalance RefDiag baseline，这一对实验用于比较 hard/soft alignment，不用于单独估计 anchoring 相对 vanilla RBLA 的绝对收益。

执行：

```bash
PYTHONPATH=src python src/test/rbla_reference_experiment/run_reference_experiments.py double-imbalance
```

运行全部七个训练实验：

```bash
PYTHONPATH=src python src/test/rbla_reference_experiment/run_reference_experiments.py all
```

## 7. 总体优先级与停止规则

```text
P0 代码级机制存在性
  ↓ pass
P1 训练中是否真实发生
  ↓ drift/discrepancy 增长
P2 hard intervention 是否改善结果
  ↓ discrepancy 降低且表现有收益/存在容量限制
P3 soft intervention 能否恢复适应能力
```

不要跳过 P1 直接以 P3 accuracy 作为方法有效性的证据。也不要在 P0–P3 阶段加入 Weak-B、adaptive lambda、coverage-aware 或 tail correction，否则无法识别收益来自 reference alignment 还是另一机制。

## 8. 自动测试

```bash
PYTHONPATH=src python src/unittest_ml/unittest_rbla_reference_frame.py
```

测试覆盖：

- 功能保持的 gauge transformation；
- RBLA 对换基的敏感性；
- shared A 时 discrepancy 为零；
- Strong-A loss 对方向/尺度变化有响应且梯度有限；
- 三个新 aggregation methods 已注册；
- 新 MNIST 模型在不同 rank 下 scaling 一致。

## 9. 首次 MNIST 实测结果（2026-07-15，历史基线）

本节记录默认配置的单 seed、30 轮验证结果。每个训练 CSV 含 round 0–30，共 31 个评估点；当前结果用于机制筛选，不作为最终统计显著性结论。

> 本节保留第一次 P1–P3 运行，便于检查重复运行稳定性。加入同秩和 double-imbalance 对照后的最新判断见第 10 节；第 10 节结论优先于本节当时的后续建议。

### P0：换基反例

| 指标 | 结果 |
|---|---:|
| dense update invariance error | `1.97e-7` |
| per-client function error | `3.02e-7` |
| RBLA reparameterization sensitivity | `1.2471` |

客户端函数和 dense update 在数值误差范围内保持不变，而 RBLA 聚合结果发生 O(1) 变化，P0 通过：RBLA 确实对 LoRA 因子参考系敏感。

### P1–P3 汇总

| 实验 | 最佳 accuracy（轮次） | 最终 accuracy | 末 5 轮 accuracy | 最终 loss | 末 5 轮 A proximal drift | 末 5 轮 aggregation discrepancy |
|---|---:|---:|---:|---:|---:|---:|
| P1 RefDiag | 0.3341（29） | 0.1837 | 0.2563 | 14.5795 | 0.9080 | 0.10081 |
| P2 Freeze-A | 0.3217（7） | 0.1835 | 0.1847 | 6.1020 | 0.0000 | `4.61e-7` |
| P3 Strong-A, lambda=0.1 | 0.2831（0） | 0.1831 | 0.1899 | 25.1919 | 0.01466 | 0.01894 |

附加诊断：

| 实验 | 最终 A cosine drift | 最终 A norm drift | 最终 frame transform drift | 最终 frame residual | 最终 tail energy ratio |
|---|---:|---:|---:|---:|---:|
| P1 | 0.22047 | 0.34791 | 39.9842 | 0.21541 | 0.60522 |
| P2 | `7.38e-9` | 0.00000 | `1.01e-7` | `8.92e-8` | 0.47083 |
| P3 | 0.00584 | 0.02939 | 0.13106 | 0.05990 | 0.59831 |

### 结论

1. **P1 通过机制观测门槛。** 训练过程中存在显著 A 漂移，末轮 A proximal drift 为 `1.0074`，RBLA 与等价 dense 聚合之间的 discrepancy 为 `0.10865`。因此参考系错位不是纯理论构造。
2. **P2 通过干预正确性门槛，但没有通过性能门槛。** Freeze-A 将 A 漂移降为零、discrepancy 降到约 `4.6e-7`，证明实现和因果干预有效；最终 accuracy 与 P1 基本相同，说明 mismatch 在本设置中存在，但不是性能坍塌的充分解释。固定随机子空间也可能限制表达能力。
3. **P3 通过 soft-alignment 机制门槛，但尚未证明性能收益。** 相比 P1，末 5 轮 discrepancy 从 `0.10081` 降至 `0.01894`，下降约 81%；A 仍有非零适应。P3 末 5 轮 accuracy 比 P2 高 `0.00512`，但最终 accuracy 低 `0.0004`，单 seed 下应视为无明确收益。
4. **当前最重要的后续实验不是继续增加机制复杂度。** 先增加一个 dense/FedAvg 或 full-rank LoRA 对照，确认该数据划分、学习率、5 local epochs 和模型本身能稳定学习；随后对 P3 做 `lambda_A={0.03,0.1,0.3}` 的短筛选和多 seed 复验。若健康对照同样退化，应先修正训练设置，而不是把低精度归因于 reference mismatch。

本次原始结果位于 `.training_results/mnist_p{1,2,3}_*.csv`，批处理日志为 `src/test/batch_summary-20260715_162223-1fc9cae2911e8177.log`。

## 10. 七实验完整对照结果（2026-07-15）

### 10.1 运行完整性与统计口径

最新批次按 `all` 顺序运行第 2.5 节定义的七个实验，批处理结果为 `7 success / 0 failed`。每个 CSV 包含 round 0–30 共 31 个评估点。除“最佳”和“最终”外，本节主要使用末 5 轮均值，因为 heterogeneous RefDiag 会在相邻轮之间剧烈振荡，单独使用最终轮容易得到偶然结论。

本批次仍是单个配置 seed。前三个 P1–P3 主配置有第 9 节的独立重复运行可供稳定性检查；两个同秩和两个 double-imbalance 配置目前各只有一次运行，因此效应量可以用于机制筛选，但不能作为多 seed 显著性结论。

批处理日志：[batch_summary-20260715_165908-fe02365c97e34062.log](../batch_summary-20260715_165908-fe02365c97e34062.log)。

### 10.2 性能汇总

| 实验 | 最佳 accuracy（轮次） | 最终 accuracy | 末 5 轮 accuracy | 末 5 轮 loss | 末 5 轮 F1 | 末 5 轮 MCC | best-final drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| RefDiag, heterogeneous | 0.3289（30） | 0.3289 | 0.2600 | 8.3960 | 0.1775 | 0.2483 | 0.0000 |
| RefDiag, uniform 1.00 | 0.8484（24） | 0.8478 | 0.8357 | 0.6787 | 0.8274 | 0.8188 | 0.0006 |
| RefDiag, uniform 0.55 | 0.8508（27） | 0.8369 | 0.8398 | 0.6547 | 0.8348 | 0.8225 | 0.0139 |
| Freeze-A, balanced heterogeneous | 0.3220（7） | 0.1837 | 0.1851 | 6.0107 | 0.0830 | 0.1659 | 0.1383 |
| Strong-A, balanced heterogeneous | 0.2829（0） | 0.1834 | 0.1834 | 14.7322 | 0.0756 | 0.1585 | 0.0995 |
| Freeze-A, double imbalance | 0.7895（30） | 0.7895 | 0.7840 | 1.3266 | 0.7310 | 0.7711 | 0.0000 |
| Strong-A, double imbalance | 0.7627（19） | 0.7355 | 0.7298 | 2.5092 | 0.6567 | 0.7157 | 0.0272 |

末 5 轮 reference diagnostics：

| 实验 | A cosine drift | A norm drift | A proximal drift | Aggregation discrepancy | Tail energy | Slot support | Frame transform drift | Frame residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RefDiag, heterogeneous | 0.20737 | 0.37622 | 0.94346 | 0.09704 | 0.55122 | 1–10 | 31.2776 | 0.18680 |
| RefDiag, uniform 1.00 | 0.15872 | 0.25098 | 0.53622 | 0.04058 | 0.50968 | 10–10 | 13.1781 | 0.14167 |
| RefDiag, uniform 0.55 | 0.08992 | 0.18671 | 0.26469 | 0.02487 | 0.46322 | 10–10 | 4.3476 | 0.15646 |
| Freeze-A, balanced heterogeneous | `7.38e-9` | 0 | 0 | `4.63e-7` | 0.46826 | 1–10 | `1.01e-7` | `8.92e-8` |
| Strong-A, balanced heterogeneous | 0.00576 | 0.03125 | 0.01599 | 0.02055 | 0.55377 | 1–10 | 0.1161 | 0.05935 |
| Freeze-A, double imbalance | `2.59e-9` | 0 | 0 | `5.36e-7` | 0.30996 | 1–10 | `7.27e-8` | `6.23e-8` |
| Strong-A, double imbalance | 0.01321 | 0.02499 | 0.02811 | 0.01820 | 0.45531 | 1–10 | 0.3188 | 0.10225 |

### 10.3 同秩对照：异构性而非总 rank budget 是关键变量

heterogeneous 与 uniform-0.55 的客户端 rank-ratio 总和都为 5.5，因此两者保持相同的总客户端 rank budget，只改变 rank 是否在客户端间异构：

```text
heterogeneous last-5 accuracy = 0.2600
uniform-0.55 last-5 accuracy = 0.8398
difference = +0.5798（+57.98 percentage points）
```

uniform-1.00 的末 5 轮 accuracy 为 0.8357，与 uniform-0.55 基本相同。增加总 rank budget 从 5.5 到 10 没有带来可辨认收益，uniform-0.55 的 best accuracy 反而略高。因此当前低性能不能解释为“总 rank 容量不足”。

最直接的结构差异是 slot support：

- heterogeneous 的高阶 slots 最少只有 1 个客户端参与，support 范围为 1–10；
- 两个 uniform 实验的每个 slot 都有全部 10 个客户端参与，support 恒为 10；
- uniform-0.55 相对 heterogeneous 将末 5 轮 discrepancy 从 0.09704 降到 0.02487，降幅约 74.4%；
- uniform-1.00 的 discrepancy 为 0.04058，降幅约 58.2%。

这组预算匹配对照提供了当前最强证据：主要问题来自 heterogeneous rank 引入的 slot-dependent participation/coverage，而不是简单的参数总量。

### 10.4 A drift 不是性能失败的充分条件

uniform-1.00 在末 5 轮仍有 0.5362 的 proximal drift 和 13.1781 的 frame transform drift，却能稳定达到约 84% accuracy。uniform-0.55 也有非零 drift 和 discrepancy，但同样正常收敛。

因此不能使用下面的简单因果链：

```text
A drift 非零或较大 => 模型必然失败
```

更符合结果的解释是：A drift 会产生 factor aggregation discrepancy，但当所有 slots 由同一组客户端共同支持时，该误差未必足以破坏学习；当高阶 slots 只由少数客户端、少数标签训练时，reference mismatch 会与 participation/coverage bias 叠加。

两个 uniform 实验中，discrepancy 与 accuracy 的 round-wise Pearson correlation 分别约为 -0.951 和 -0.956，但该相关同时包含训练时间趋势，不能单独解释为因果效应。

### 10.5 Balanced 异构秩：消除 discrepancy 仍不能恢复性能

Freeze-A 将 A drift 精确降为零，并把 discrepancy 降至约 `4.6e-7`；Strong-A 将末 5 轮 discrepancy 相对 RefDiag 降低约 78.8%，同时保留小幅 A adaptation。然而两者末 5 轮 accuracy 分别只有 0.1851 和 0.1834，均低于 RefDiag 的 0.2600。

这构成重要的 intervention 结果：

1. diagnostics 和 Freeze-A 实现符合预期，因为 shared A 时 factor/dense conditional aggregate 在浮点误差范围内一致；
2. reference mismatch 在训练中真实存在；
3. 但消除 mismatch 并不足以修复 heterogeneous-rank balanced 训练；
4. Strong-A 的低 discrepancy 也没有转化成 accuracy 收益，因此继续只调 `lambda_A` 的优先级下降。

Freeze-A 保留 heterogeneous slots：最高阶 slot 即使共享正确 A，也仍可能只由一个 two-label 客户端训练 B。服务器最终模型因而组合了针对不同客户端条件分布学习的 slots。这与 observed support `1–10` 一致，是当前比 reference mismatch 更强的失败机制候选。

### 10.6 Double imbalance：Hard Freeze-A 优于 Strong-A

double-imbalance 下，Freeze-A 的末 5 轮 accuracy 为 0.7840，Strong-A 为 0.7298，Freeze-A 高约 5.42 percentage points；末 5 轮 F1 高约 0.0744，loss 也从 2.5092 降至 1.3266。Strong-A 在 round 19 达到 0.7627 后回落到 0.7355，而 Freeze-A 的最佳值出现在最终轮。

该结果不能解释成“double imbalance 比 balanced 更容易”或“Freeze-A 普遍有效”，因为数据矩阵同时改变了客户端权重和标签覆盖：

- client 10 覆盖全部 10 类，拥有 17,580 个样本，占总训练权重约 29.3%；
- client 9 与 client 10 合计约占 48.6%；
- 大客户端拥有更广标签覆盖，使占主导权重的更新更接近全局任务；
- rank ratio 与样本量的 Pearson correlation 仅约 0.042，因此现象不是简单的 rank-volume 单调相关。

Freeze-A 的末 5 轮 tail energy 为 0.3100，Strong-A 为 0.4553。一个与结果一致但仍需验证的假设是：Strong-A 的适应重新增强了低-support tail components，而这些 components 更容易携带客户端条件偏差。由于当前没有 double-imbalance RefDiag baseline，这两个实验只能回答 hard/soft alignment 的相对表现，不能估计 anchoring 相对 vanilla RBLA 的绝对收益。

### 10.7 与首次运行的重复性比较

| 方法 | 首次 best | 最新 best | 首次末 5 轮 | 最新末 5 轮 | 首次 final | 最新 final |
|---|---:|---:|---:|---:|---:|---:|
| RefDiag heterogeneous | 0.3341 | 0.3289 | 0.2563 | 0.2600 | 0.1837 | 0.3289 |
| Freeze-A balanced | 0.3217 | 0.3220 | 0.1847 | 0.1851 | 0.1835 | 0.1837 |
| Strong-A balanced | 0.2831 | 0.2829 | 0.1899 | 0.1834 | 0.1831 | 0.1834 |

heterogeneous RefDiag 的最终轮差异很大，但两次 best accuracy 和末 5 轮均值高度接近：它重复表现为约 18%–33% 之间的剧烈轮间振荡。Freeze-A 和 Strong-A 的重复结果也保持同一失败方向。因此后续报告应以 last-5、best-final drop 和多 seed 分布为主，不能只报告 final round。

### 10.8 更新后的机制判断与实验优先级

当前证据支持以下分层结论：

1. **Reference mismatch 存在。** P0 换基反例、P1 drift/discrepancy 和 Freeze-A 数值不变量均支持这一点。
2. **Reference mismatch 不是 balanced heterogeneous-rank 性能失败的充分解释。** Freeze-A/Strong-A 显著降低 discrepancy 却不改善 accuracy。
3. **Rank-dependent slot participation/label coverage 是当前主导候选机制。** budget-matched uniform-0.55 将 accuracy 从约 26% 提升到约 84%，且所有 slots support 从 1–10 变为 10–10。
4. **A drift 在完整 slot support 下可被训练容忍。** uniform 实验在非零 drift 下稳定学习。
5. **数据分布会改变 heterogeneous tail 的危害。** double imbalance 中大权重客户端覆盖更多标签，Freeze-A 可以工作；Strong-A 反而增加 tail energy 并降低性能。

下一轮验证优先级更新为：

1. heterogeneous ranks + common-head-only aggregation：只保留所有客户端共同支持的 slots，直接测试 tail support 因果效应；
2. support-threshold aggregation：仅聚合 support 至少为 `k` 的 slots，并扫描 `k`；
3. coverage-aware 或 support-aware slot weighting，避免单客户端 tail 获得与全支持 head 相同的条件归一化强度；
4. 至少 3 个 rank-client permutation seeds，验证提升不依赖当前 mapping；
5. 增加 double-imbalance RefDiag baseline，补齐 vanilla/Freeze/Strong 三方对照；
6. 在完成上述因果隔离后，再决定是否继续 Strong-A `lambda_A={0.03,0.1,0.3}` 扫描。

### 10.9 最新原始结果

- [RefDiag heterogeneous](../.training_results/mnist_refdiag_two_label_balanced_rank_heterogeneous_-train-20260715_165912-bdd95764b0be56f2.csv)
- [RefDiag uniform 1.00](../.training_results/mnist_refdiag_two_label_balanced_rank_uniform_1_00_-train-20260715_170814-cd2c04274e45b50f.csv)
- [RefDiag uniform 0.55](../.training_results/mnist_refdiag_two_label_balanced_rank_uniform_0_55_-train-20260715_171659-5ff09f95b3d433f2.csv)
- [Freeze-A balanced heterogeneous](../.training_results/mnist_freeze_a_two_label_balanced_rank_heterogeneous_-train-20260715_172454-ebed8015f1cd0910.csv)
- [Strong-A balanced heterogeneous](../.training_results/mnist_strong_a_two_label_balanced_rank_heterogeneous_lambda_0_10_-train-20260715_173203-af9523eff5624efb.csv)
- [Freeze-A double imbalance](../.training_results/mnist_freeze_a_double_imbalance_rank_heterogeneous_-train-20260715_174159-bc9e6daa86136d8f.csv)
- [Strong-A double imbalance](../.training_results/mnist_strong_a_double_imbalance_rank_heterogeneous_lambda_0_10_-train-20260715_175102-2efc0a1e05743bb8.csv)

## 11. P4–P7：严格区分 frame、coverage 与 local capacity

本轮不加入 Weak-B、adaptive lambda、新 padding/embedding 或新的本地优化目标。新增实现全部使用独立的 analysis strategy、aggregation、runner、entry 和 YAML；已有 `rbla_refdiag`、`rbla_freeze_a`、`rbla_strong_a` 以及原始 `rbla/sara/rblasa/ffalora` 的行为不变。

三个待区分误差源的操作性定义如下：

| 误差源 | 本轮隔离方式 | 能回答的问题 | 仍不能回答的问题 |
|---|---|---|---|
| reference/frame error | Freeze-A 令所有 eligible clients 共享同一 A，并用匹配 weighting 的 direct rank-one aggregate 检查 discrepancy | 分别聚合 A/B 的 cross terms 是否仍存在 | 异构客户端本地容量是否充足 |
| coverage/support error | P4 删除 prefix 之后的 tail；P6 只将 B slot 乘 `q_s^gamma` | 低 support slot 是否具有负贡献；conditional normalization 是否放大它 | 标签覆盖与纯幅值放大各自的完整效应量 |
| local capacity error | P5 保持总 rank budget=5.5、只移除异构性；P4/P6 保持本地异构 rank 不变 | 固定/强约束 A 是否本身使模型失效 | heterogeneous local rank 的独立因果效应仍未被完全隔离 |

### 11.1 P4：checkpoint 与 post-hoc prefix truncation

原 analysis runner 的可选 logger 保存 round-30 global checkpoint。对每个 LoRA 层的最大 rank `r_l` 和保留比例 `rho`，使用：

```text
k_l = round(rho * r_l)
A_l[k_l:, :] = 0
B_l[:, k_l:] = 0
rho in {0.1, 0.2, ..., 1.0}
```

base model、bias 和前 `k_l` 个 slots 均不变。MNIST 三层最大 rank 为 `160/160/100`，所以这些比例均对应整数 rank。每个比例重新计算 accuracy、平均 cross-entropy、macro-F1、MCC、10 类 recall 和 confusion matrix。该实验只测“已经学到的 tail 对最终 global model 是正贡献还是负贡献”，不把 post-hoc 最优点当作新的训练方法。

执行：

```bash
PYTHONPATH=src python src/test/rbla_reference_experiment/evaluate_rank_truncation.py
```

### 11.2 P5：uniform-0.55 anchoring controls

新增 `uniform-0.55 + Freeze-A` 和 `uniform-0.55 + Strong-A(lambda_A=0.1)`。它们与已有 uniform-0.55 RefDiag 共享 two-label balanced 数据矩阵、10 clients、`[0.55] x 10`、模型/scaling、Adam、学习率、weight decay、5 local epochs、30 rounds、初始化与数据 seed 以及全参与设置。唯一方法差异是 A 的 hard/soft anchoring。

### 11.3 P6：只对 B 使用 support scaling

令全局客户端权重已经归一化为 `sum_i p_i=1`，slot `s` 的 eligible 集合与权重质量为：

```text
E_s = {i | r_i > s}
q_s = sum_(i in E_s) p_i
alpha_i,s = p_i / q_s
```

当前 conditional RBLA 是 `b_bar_s = sum_i alpha_i,s b_i,s`。新增 aggregation 只改为：

```text
b_bar_s^(gamma) = q_s^gamma sum_(i in E_s) alpha_i,s b_i,s
gamma in {0, 0.5, 1}
```

A 保持 shared/frozen conditional average，不乘 `q_s^gamma`，因而不会产生 `q_s^(2 gamma)`。`gamma=0` 是原 Freeze-A；`gamma=1` 等于 `sum_i p_i b_i,s`，即 absent slot 按零贡献；`gamma=0.5` 是两者之间的 shrinkage。

新增单元测试验证：`gamma=0` 与现有 Freeze-A 数值一致；`gamma=1` 等于未重新归一化的 eligible 加权和；shared A 下 factor aggregate 等于匹配 gamma 的 direct rank-one aggregate；`q_s=1` 时三个 gamma 完全相同；所有输出有限。

### 11.4 P7：slot coverage statistics

除 raw support `n_s=|E_s|` 外，每层、每 slot 记录：

```text
q_s = sum_(i in E_s) p_i
N_eff,s = 1 / sum_(i in E_s) alpha_i,s^2

pi_s,c = sum_(i in E_s) alpha_i,s * n_i,c / n_i
H_s = -sum_c pi_s,c log(pi_s,c) / log(10)
C_s = |{c | some eligible client has n_i,c > 0}| / 10

e_s = ||b_bar_s||_2 ||a_bar_s||_2
R_tail = sum_(s in tail) e_s (1-H_s) / (sum_s e_s + eps)
```

`pi_s,c` 在数值上再次归一化，防止输入矩阵舍入误差。`N_eff` 区分“多个客户端但权重被单一大客户端支配”和真正的多客户端支持；`C_s` 只看标签是否出现，`H_s` 同时看标签质量分布是否均衡。tail 定义为每层后半 slots。每个最终 checkpoint 输出 all-round slot CSV、含 eligible client IDs/indices/labels 和 `pi_s,c` 的 tail JSON，以及 support、entropy、energy、energy-vs-entropy 四张 SVG。

## 12. P4–P7 单 seed 结果（2026-07-15）

P4 五个训练、P5 两个训练、P6 三个训练均完成 30 rounds，训练 seed 与 rank mapping seed 为 42。P6 `gamma=0` 与 P4 Freeze-A balanced 的逐轮结果精确相同，构成实现回归检查。下表中除 best/final/truncation 外均为最后 5 轮均值。

### 12.1 统一汇总表

| Experiment | Rank | gamma | best/final/last-5 acc. | loss / F1 / MCC | discrepancy | tail energy / `R_tail` | support；tail H | best trunc.（acc.） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RefDiag balanced | hetero | – | .3341/.1837/.2563 | 9.305/.1660/.2426 | .10081 | .5572/.3214 | 1–10；.5045 | .7（.6303） |
| Freeze-A balanced | hetero | – | .3217/.1835/.1847 | 5.969/.0818/.1644 | `4.61e-7` | .4702/.1612 | 1–10；.6561 | .7（.5820） |
| Strong-A balanced | hetero | – | .2831/.1831/.1899 | 18.822/.0856/.1683 | .01894 | .5783/.3020 | 1–10；.5281 | .5（.6023） |
| Freeze-A double | hetero | – | .7913/.7912/.7847 | 1.311/.7308/.7717 | `5.35e-7` | .3071/.0377 | 1–10；.8875 | .6（.9436） |
| Strong-A double | hetero | – | .7589/.7474/.7508 | 2.162/.6748/.7373 | .01681 | .4455/.0458 | 1–10；.8865 | .6（.9778） |
| Freeze-A balanced | uniform .55 | – | .8288/.8288/.8254 | .704/.8184/.8073 | `5.10e-7` | .4999/`~0` | 10–10；1.000 | 1.0（.8288） |
| Strong-A balanced | uniform .55 | – | .8654/.8513/.8572 | .579/.8522/.8420 | .01328 | .4906/`~0` | 10–10；1.000 | 1.0（.8513） |
| Freeze support | hetero | 0 | .3217/.1835/.1847 | 5.969/.0818/.1644 | `4.61e-7` | .4702/.1612 | 1–10；.6561 | .7（.5820） |
| Freeze support | hetero | .5 | .7329/.7174/.7211 | .965/.7148/.6967 | `6.72e-7` | .2232/.0357 | 1–10；.7734 | .9（.7523） |
| Freeze support | hetero | 1 | .7182/.7182/.7162 | 1.078/.7106/.6869 | `6.95e-7` | .1946/.0148 | 1–10；.8166 | .9（.7208） |

这里 `tail H` 是按 tail slot energy 加权的 normalized label entropy；全 slot 的普通平均 entropy 在三个 balanced heterogeneous gamma 实验中均为 `.8503`，因为 eligibility 没有变化。gamma 改变的是低支持 slot 的能量贡献，因而改变 energy-weighted tail H 与 `R_tail`。

### 12.2 P4：tail slots 对五个 heterogeneous global models 都是负贡献

| Checkpoint | best rho | best acc./loss/F1/MCC | full acc./loss/F1/MCC | acc./F1 gain |
|---|---:|---:|---:|---:|
| RefDiag balanced | .7 | .6303/1.2246/.5577/.6036 | .1837/14.5795/.0755/.1587 | +.4466/+.4821 |
| Freeze-A balanced | .7 | .5820/1.5347/.4964/.5556 | .1835/6.1020/.0803/.1637 | +.3985/+.4161 |
| Strong-A balanced | .5 | .6023/1.2237/.5452/.5704 | .1831/25.1919/.0808/.1672 | +.4192/+.4643 |
| Freeze-A double | .6 | .9436/.2111/.9426/.9377 | .7912/1.2461/.7417/.7786 | +.1524/+.2009 |
| Strong-A double | .6 | .9778/.0880/.9776/.9753 | .7474/2.1259/.6701/.7335 | +.2304/+.3075 |

最关键的是 Freeze-A：A 完全 shared，aggregation discrepancy 只有 `O(1e-7)`，但删除 tail 仍使 balanced accuracy 提升 39.85 percentage points。因此这个 P4 效应不能由 reference mismatch 解释，直接证明已训练 tail 对 global inference 有负贡献。

效果并非简单的“rank 越小越好”。五个模型的最优点为 `.5–.7`，再继续截断会下降；P5 两个 uniform 模型的最优点都为 full rank。因此诊断指向 heterogeneous tail 的支持/语义，而不是全局 LoRA rank 普遍过大。

### 12.3 P5：hard freeze 和 lambda=0.1 本身都不会导致 18% 坍塌

uniform-0.55 Freeze-A 最后 5 轮 accuracy 为 `.8254`，接近原 uniform-0.55 RefDiag 的 `.8398`，并明显不同于 heterogeneous Freeze-A 的 `.1847`。这排除了“固定随机 A 子空间必然使 balanced MNIST 无法学习”的解释。

uniform-0.55 Strong-A 的最后 5 轮 accuracy 为 `.8572`、best 为 `.8654`，高于同设置 Freeze-A；所以 `lambda_A=0.1` 在完整 slot support 下并不过强。heterogeneous Strong-A 的失败不能简单归因于该 lambda 压制 A capacity。

P5 仍没有直接证明所有 heterogeneous clients 的本地 rank 都足够；它证明的是相同总 rank-ratio budget 下，anchoring 与整体容量可正常工作，真正危险的变化是 rank 在客户端间异构后产生的 slot-dependent eligibility。

### 12.4 P6：conditional support amplification 是因果机制

从 `gamma=0` 到 `.5`，最后 5 轮 accuracy 从 `.1847` 提升到 `.7211`，macro-F1 从 `.0818` 提升到 `.7148`，MCC 从 `.1644` 提升到 `.6967`；同时 loss 从 `5.969` 降至 `.965`。`gamma=1` 的 accuracy 为 `.7162`，同样恢复，但略低于 `.5`。

三个实验的 A、客户端本地 rank、rank-client mapping、数据、训练过程和 optimizer 相同；shared-A discrepancy 对各自匹配 weighting 均为 `O(1e-7)`。唯一干预是服务器 B slot 上的 `q_s^gamma`。因此，原 conditional rule 把低 `q_s` slots 重新放大到与 full-support slots 相同尺度，是 balanced heterogeneous 失败的一个已建立因果机制。单 seed 下 `.5` 略优说明完全 conditional 与完全 zero-missing 可能都是极端点，但 `.5` 相对 `1` 的差只有约 0.49 percentage point，尚不能据此确定最优 gamma。

P6 没有完全恢复 uniform-0.55 的约 `.84`。残余差距约 12 percentage points，可能来自 heterogeneous local capacity、仍保留的 slot 标签偏置、不同 slots 的优化质量或单 seed 波动。因而不能把 support amplification 称为唯一误差来源。

### 12.5 P7：最高风险 tail 的标签与 full-rank confusion 精确对应

balanced rank mapping 中，最高 rank client 是 `client.6`（zero-based index 5，ratio=1.0），只持有标签 `{5,6}`。三个 balanced 方法最终最高风险 tail slots 均满足：

```text
raw support = 1, q_s = 0.1, N_eff = 1
eligible client = client.6
eligible labels = {5,6}
H_s = log(2)/log(10) = 0.3010
```

full-rank confusion 的预测计数为：

| Method | predicted class 5 | predicted class 6 | other predicted classes | recall 5 / 6 |
|---|---:|---:|---:|---:|
| RefDiag | 7,449 | 2,551 | 0 | .990/.996 |
| Freeze-A | 7,682 | 2,305 | 13 | .990/.982 |
| Strong-A | 7,829 | 2,171 | 0 | .990/.990 |

也就是说，balanced full-rank 模型几乎把全部 10,000 个测试样本预测成最低支持 tail 唯一 client 的两个标签；P4 删除该 tail 后恢复多类预测。这比单独的 support count 更强：低 entropy tail 的语义与过预测类别逐项对齐，并且 Freeze-A 中不存在 frame cross-term 混杂。

double-imbalance 的最高风险 support-1 tail 仍来自 `client.6`，但其标签集合为 `{0,1,2,3,4,5}`，`q_s=.0862`、`H_s=.766`。full-rank 模型主要损失高类别：Freeze-A class 8/9 recall 只有 `.234/.052`，Strong-A 为 `0/.008`；截断到 `.6` 后分别恢复为 `.893/.793` 和 `.972/.963`。ratio=.6 的 `client.10` 覆盖全部 10 类，保留到该 prefix 而删除更高的窄覆盖 slots，与最优截断点一致。

在 10 个 P4–P6 配置的描述性横截面上，最后 5 轮 `R_tail` 与 accuracy/macro-F1 的 Pearson correlation 分别为 `-.883/-.887`，tail entropy 与二者分别为 `+.913/+.907`。这些配置共享 checkpoint、方法和时间趋势，观测并不独立，所以相关只作为与 P4/P6 干预一致的辅助证据，不作为额外因果证明。

## 13. 当前证据边界与下一步判断

### 13.1 已证明的事实

1. factor reference mismatch 在代码级反例和真实训练中存在；Freeze-A 能将匹配 conditional weighting 的 factor/direct discrepancy 降到浮点误差。
2. mismatch 不是 balanced heterogeneous 失败的充分原因：shared-A Freeze-A 仍失败。
3. heterogeneous global tail 在五个 checkpoint 中都有显著负贡献；该事实在 shared-A Freeze-A 上仍成立。
4. Freeze-A 与 Strong-A 在 uniform-0.55 下均能正常学习，hard freeze 与 `lambda_A=.1` 本身不是 18% 坍塌的主因。
5. 只缩放 B 的 `q_s^gamma` 干预使 balanced Freeze-A 从约 18% 恢复到约 72%；conditional support amplification 是因果机制之一。
6. balanced 的最低 support/highest-risk slots 只含标签 5/6，full-rank 过预测也几乎精确集中到 5/6。

### 13.2 当前最可能的机制

异构 rank 先产生 slot-dependent eligible client/label sets；conditional renormalization 再把 `q_s` 很小、entropy 很低的 tail 更新放大到与 head 相同的聚合尺度。高能量、低 entropy tail 因而把全局决策边界拉向少数客户端条件分布。reference mismatch 会在可训练 A 时叠加 cross terms，但不是该失败链的必要条件。

### 13.3 尚未隔离的混杂变量

- heterogeneous local capacity 本身：低 rank clients 的本地表示限制尚未通过保持服务器 support 不变的独立实验直接测量；
- support count、support weight 与 label coverage 在当前 rank mapping 中共同变化；P6 隔离了幅值放大，但没有把“同 support 不同 label entropy”单独随机化；
- P4 是对最终模型的诊断，不等价于从第一轮就以截断 rank 重新训练；
- `.5` 与 `1` 的优劣当前只有一个 seed，不能当作最优 gamma 结论；
- P4 最佳截断后的 balanced accuracy 只有 `.58–.63`，P6 约 `.72`，仍低于 uniform 约 `.84`，所以 support correction 尚未解释全部差距。

### 13.4 Weak-B 与 coverage-aware 是否值得进入下一阶段

当前不值得加入 Weak-B：shared-A 的 P4/P6 已在没有 frame discrepancy 的条件下定位到 support/coverage 机制，再约束 B 会同时改变本地适应和 slot energy，反而破坏现有因果可辨识性。

值得继续研究简单、预先规定的 coverage-aware aggregation，但前提是先完成多 seed：P6 已证明仅使用 `q_s` 的固定缩放有效，P7 又显示 label entropy 与错误类别有明确语义对应。下一阶段应优先比较固定 gamma support scaling、预先固定的 support threshold，以及只使用训练前数据统计的 coverage weight；不应先引入 adaptive lambda 或联合复杂优化。

## 14. P6 的 3 training seeds x 3 rank-assignment seeds 配对复验

P4/P6 已明确支持 tail coverage/support bias，因此按停止规则追加最小多 seed 验证。为控制计算量且保持问题单一，只对当前 Freeze-A baseline `gamma=0` 和单 seed 中最优的固定候选 `gamma=.5` 做逐格配对；`gamma=1` 保留第 12 节的单 seed 结果，不据此比较 `.5` 与 `1` 的最终优劣。

### 14.1 Seed 与可复现性处理

三个固定 rank permutations 均使用相同的 ratio multiset `{.1,.2,...,1.0}`，只改变 client mapping：

```text
rank seed 42: [.4, .9, .1, .7, .2, 1., .5, .8, .3, .6]
rank seed 43: [.9, .2, .6, .7, 1., .8, .4, .3, .5, .1]
rank seed 44: [.5, .6, .1, .3, 1., .4, .8, .2, .9, .7]
```

training seeds 为 `42/43/44`，同时作用于初始化、数据 pool 顺序和本地 shuffle。代码核查发现旧 `BaseStrategy` 和 `ModelTrainer` 构造函数会在 entry 设置 seed 后再次调用 legacy `set_seed(42)`。为了不改变它们的现有行为，新增 `entries.lora_reference_seeded` 只在该独立进程内把这个 legacy default 映射为指定 training seed，并在结束时恢复原方法。修正后 t=43 与 t=42 checkpoint 的最大 tensor difference 为 `1.6094`；修正前产生的同权重试跑使用旧文件前缀，汇总脚本只读取修正后的 `v2` 前缀，不计入结果。

每个配对使用相同 training seed、rank list、初始化和数据顺序，唯一差异为服务器 B aggregation 的 `gamma`。seed 42/rank 42 复用第 12 节已完成的两个单 seed 训练，其余格由 seed-matrix runner 运行。

### 14.2 九格配对结果

| Training seed | Rank seed | gamma=0 last-5 acc. | gamma=.5 last-5 acc. | acc. delta | F1 delta | MCC delta | loss delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 42 | .1847 | .7211 | +.5363 | +.6330 | +.5323 | -5.0041 |
| 42 | 43 | .1999 | .6820 | +.4822 | +.5669 | +.4822 | -5.3636 |
| 42 | 44 | .1869 | .6970 | +.5101 | +.6121 | +.5301 | -5.0586 |
| 43 | 42 | .1825 | .7420 | +.5595 | +.6484 | +.5445 | -5.4268 |
| 43 | 43 | .2307 | .6387 | +.4080 | +.4917 | +.3958 | -4.3063 |
| 43 | 44 | .1915 | .7303 | +.5388 | +.6420 | +.5555 | -4.0303 |
| 44 | 42 | .1828 | .6926 | +.5098 | +.5967 | +.5028 | -6.2275 |
| 44 | 43 | .1895 | .5730 | +.3835 | +.4565 | +.4108 | -5.3354 |
| 44 | 44 | .1876 | .6849 | +.4973 | +.5935 | +.5139 | -5.2594 |

所有 9 格的 accuracy、macro-F1 和 MCC 差值均为正，loss 差值均为负。逐方法汇总为 mean +/- sample standard deviation：

| Metric | gamma=0 | gamma=.5 | 方向 |
|---|---:|---:|---:|
| best accuracy | `.2230 +/- .0413` | `.7134 +/- .0319` | 改善 |
| final accuracy | `.1927 +/- .0163` | `.6835 +/- .0538` | 改善 |
| last-5 accuracy | `.1929 +/- .0151` | `.6846 +/- .0519` | 改善 |
| last-5 macro-F1 | `.0852 +/- .0227` | `.6675 +/- .0621` | 改善 |
| last-5 MCC | `.1644 +/- .0226` | `.6608 +/- .0503` | 改善 |
| last-5 loss | `6.1633 +/- .6421` | `1.0509 +/- .0769` | 降低 |
| last-5 tail energy | `.4436 +/- .0202` | `.2156 +/- .0066` | 降低 |
| last-5 `R_tail` | `.1544 +/- .0066` | `.0402 +/- .0067` | 降低 |

配对 last-5 accuracy delta 的均值为 `+.4917`、样本标准差 `.0595`、最小/最大为 `+.3835/+.5595`。把 9 个 crossed cells 暂按独立配对处理得到的双侧 95% t interval 为 `[+.4460,+.5374]`。由于同一 training seed 和 rank seed 会跨格复用，这个区间是描述性区间，并非严格的层级随机效应推断；稳健证据主要来自 9/9 方向一致和大效应量，而不是 p-value。

### 14.3 两类 seed 的边际变化

| Group | gamma=0 last-5 acc. | gamma=.5 last-5 acc. |
|---|---:|---:|
| training seed 42，跨 3 rank seeds | .1905 | .7000 |
| training seed 43，跨 3 rank seeds | .2016 | .7037 |
| training seed 44，跨 3 rank seeds | .1866 | .6501 |
| rank seed 42，跨 3 training seeds | .1834 | .7186 |
| rank seed 43，跨 3 training seeds | .2067 | .6312 |
| rank seed 44，跨 3 training seeds | .1887 | .7041 |

baseline 在所有边际组都停留在约 `.18–.21`；support scaling 在所有组都恢复到 `.63–.72`。rank seed 43 下 treatment 较低，且单格最低值出现在 t=44/r=43，说明 rank-label assignment 仍显著调节剩余性能。这与“support amplification 是重要因果机制，但不是唯一机制”的结论一致。

### 14.4 多 seed 后的更新结论

多 seed 将 P6 从单 seed 机制筛选提升为当前实验范围内的稳健因果证据：固定 `q_s^.5` intervention 对 3 个训练 seeds 和 3 个 rank mappings 的每一格都大幅改善 accuracy/F1/MCC，并同时降低 tail energy 与 `R_tail`。这支持 conditional slot amplification，而不是某个特定初始化或唯一 rank-client mapping，作为 balanced heterogeneous Freeze-A 坍塌的主要机制。

仍不能称 `gamma=.5` 为最终方法或普适最优值：候选由 seed42 的 `.5`/`1` 比较后选出；实验只覆盖 MNIST/two-label balanced/Freeze-A；rank seed 仍带来约 8.7 points 的 treatment 边际范围；并且 `.6846` 仍低于 uniform-0.55 的约 `.84`。所以下一阶段可以研究简单的 coverage-aware/support-threshold aggregation，但应继续保持预先规定、可解释的 weighting，并增加任务/数据分布复验；Weak-B、adaptive lambda 和新的复杂本地优化仍没有优先依据。

### 14.5 结果产物

- P4 完整 ratio 指标：[p4_truncation_metrics.csv](results/p4_truncation/p4_truncation_metrics.csv)
- P4 每类 recall 与 confusion JSON：[p4_truncation_details.json](results/p4_truncation/p4_truncation_details.json)
- P4 最优比例：[p4_truncation_best.csv](results/p4_truncation/p4_truncation_best.csv)
- P4 accuracy/F1/loss 图：[results/p4_truncation](results/p4_truncation)
- P4–P7 统一汇总：[p4_p7_unified_summary.csv](results/p4_p7_summary/p4_p7_unified_summary.csv)
- P6 18 个 seed-run 指标：[p6_seed_matrix_metrics.csv](results/p6_seed_matrix/p6_seed_matrix_metrics.csv)
- P6 9 个配对差值：[p6_seed_matrix_paired_deltas.csv](results/p6_seed_matrix/p6_seed_matrix_paired_deltas.csv)
- P6 聚合统计：[p6_seed_matrix_summary.json](results/p6_seed_matrix/p6_seed_matrix_summary.json)
- P6 执行 manifest：[run_manifest.json](results/p6_seed_matrix/run_manifest.json)
- P7 每个实验的 slot CSV、tail JSON 和四张 SVG：位于 [`.training_results`](../.training_results)，文件名与相应 checkpoint 前缀一致。

## 15. P8–P11：anchoring、support exponent 与 residual coverage（2026-07-16）

本阶段不改变 P0–P7 的任何 strategy 或 runner。新增实现使用独立的 strategy/aggregation 名称、YAML、runner、日志和结果前缀；没有加入 Weak-B、adaptive lambda、entropy-aware weighting、learned aggregation 或新的 rank allocation。所有训练仍为 MNIST reference MLP、10 clients 全参与、two-label balanced 或既有 double-imbalance 矩阵、LoRA `alpha/r=1`、Adam `1e-3`、weight decay `1e-4`、每轮 5 local epochs、共 30 rounds。

### 15.1 复用的 support-scaling 定义与实现不变量

新增的统一 helper 接受 `conditional`、`q_power`、`effective_support` 和 `population` 四种 scaling。对归一化客户端权重 `p_i`：

```text
E_s = {i | r_i > s}
q_s = sum_(i in E_s) p_i
alpha_i,s = p_i / q_s

a_bar_s = sum_(i in E_s) alpha_i,s a_i,s
b_bar_s = c_s sum_(i in E_s) alpha_i,s b_i,s
```

仅对 B 乘 `c_s`；A 始终保持原方法的 conditional aggregation。四种系数分别为：

```text
conditional:       c_s = 1
q_power:           c_s = q_s^gamma
population:        c_s = q_s
effective_support: c_s = sqrt(N_eff,s / N_eff,full)

N_eff,s    = 1 / sum_(i in E_s) alpha_i,s^2
N_eff,full = 1 / sum_i p_i^2
```

匹配 scaling 的 reference discrepancy 为：

```text
DeltaW_factor = sum_s c_s b_bar_cond,s a_bar_s
DeltaW_direct = sum_s c_s sum_(i in E_s) alpha_i,s b_i,s a_i,s

D_agg = ||DeltaW_factor - DeltaW_direct||_F
        / (||DeltaW_direct||_F + eps)
```

同时保存 numerator `||DeltaW_factor-DeltaW_direct||_F` 和 denominator `||DeltaW_direct||_F`。这样可以区分真实 mismatch 与小分母造成的相对误差放大。A cosine/norm/proximal drift、frame transform/residual 的定义沿用第 4.1 节；其中 proximal drift 是相对广播 A 的归一化平方距离，而不是 Strong-A loss 本身。`R_tail` 仍为第 11.4 节定义的 energy-weighted low-entropy tail risk。

新增单元测试确认：每个新方法在 `gamma=0` 时逐 tensor 复现自己的原 aggregation；`q_s=1` 时所有 scaling 相同；只缩放 B；Freeze-A 的 factor/direct 误差小于 `1e-6`；RefDiag/Strong-A 保持非零 discrepancy；effective-support 公式正确；全部输出无 NaN/Inf。每个新 checkpoint 的 sidecar metadata 记录 training seed、rank seed、完整 rank permutation、gamma/scaling type、git hash/dirty 状态、YAML 路径、checkpoint 路径和 coverage artifacts。

## 16. P8：A-side anchoring × `sqrt(q)` 单 seed 筛选

### 16.1 设计与停止判断

三种方法共享 training seed 42、rank seed 42 和完全相同的 balanced heterogeneous 设置，唯一机制差异是 A-side 训练：RefDiag 正常训练 A/B；Strong-A 使用 `lambda_A=.1`；Freeze-A 固定共享 A。B 均使用 `c_s=sqrt(q_s)`。既有 P6 Freeze-A checkpoint 用于性能比较；由于旧 CSV 缺少新的 absolute discrepancy/global norm 字段，另用独立 P8 前缀复跑 diagnostics。复跑与旧 P6 的公共 CSV 数值最大差为 0，checkpoint 最大 tensor difference 也为 0，因此不是额外的性能样本。

预设 gate 是：只有 RefDiag 或 Strong-A 相对 Freeze-A 提升约 2–3 points 且机制指标一致，才触发 P8-M。实际结果相反，因此没有运行新增的 18 个 RefDiag/Strong-A seed-grid 训练。

### 16.2 训练与机制结果

除 best/final 外，下表均为最后 5 轮均值。

| A mode | best（round）/final/last-5 acc. | loss / macro-F1 / MCC | A cosine / norm / proximal drift | transform / residual | relative discrepancy | absolute numerator / denominator |
|---|---:|---:|---:|---:|---:|---:|
| RefDiag | `.5899(30)/.5899/.5504` | `1.3880/.5100/.5321` | `.1200/.2452/.5312` | `32.2250/.1462` | `.04454` | `.5241/8.8785` |
| Strong-A | `.6442(30)/.6442/.5919` | `1.1469/.5584/.5748` | `.00601/.02533/.01455` | `.12690/.05916` | `.01962` | `.2921/8.4221` |
| Freeze-A | `.7329(17)/.7174/.7211` | `.9652/.7148/.6967` | `~0/0/0` | `~1.0e-7/~8.9e-8` | `6.72e-7` | `6.00e-6/8.5635` |

| A mode | tail energy | tail entropy | `R_tail` | global `||A|| / ||B|| / ||DeltaW||` |
|---|---:|---:|---:|
| RefDiag | `.3351` | `.5423` | `.1224` | `13.889/4.498/8.661` |
| Strong-A | `.3173` | `.5595` | `.1067` | `14.073/5.179/8.317` |
| Freeze-A | `.2232` | `.7734` | `.0357` | `11.807/13.184/8.563` |

Strong-A 相对 RefDiag 的 last-5 accuracy/F1 分别高 `4.15/4.84` points，且 A drift/discrepancy 明显下降，说明 soft anchoring 确实约束了 reference mismatch；但 Freeze-A 又比 Strong-A 高 `12.92/15.64` points，并有更低的 tail energy 与 `R_tail`。因此在 support scaling 已启用后，没有观察到 soft A adaptation 的独立收益；hard shared reference 是该筛选中最好的完整模型基础。

### 16.3 逐类行为、confusion 与 post-hoc truncation

full checkpoint 的 class 0–9 recalls 为：

```text
RefDiag:  [.460,.978,.205,.001,.754,.911,.983,.911,.402,.299]
Strong-A: [.512,.965,.652,.018,.743,.943,.972,.879,.346,.407]
Freeze-A: [.674,.987,.481,.512,.869,.840,.924,.874,.477,.525]
```

RefDiag/Strong-A 的主要共同错误是 class 3 几乎没有召回，并大量被预测为 class 5；Freeze-A 虽仍在 2/3/8/9 上较弱，但没有同等严重的单类坍塌。完整 10×10 confusion matrices 与每个 truncation ratio 的 recalls 保存在 [P8 truncation details](results/p8_p11/p8_single_evaluation/truncation_details.json)，对应矩阵图位于 [P8 evaluation directory](results/p8_p11/p8_single_evaluation)。

| A mode | best truncation | best/full accuracy | accuracy gain | best/full macro-F1 | F1 gain |
|---|---:|---:|---:|---:|---:|
| RefDiag | `.7` | `.7020/.5899` | `+.1121` | `.6459/.5428` | `+.1031` |
| Strong-A | `.8` | `.7551/.6442` | `+.1109` | `.7222/.6165` | `+.1057` |
| Freeze-A | `.9` | `.7523/.7174` | `+.0349` | `.7441/.7112` | `+.0329` |

Strong-A 的最佳截断准确率与 Freeze-A 接近，但其完整模型明显更差。这意味着 soft A 模式并非缺少可用 prefix capacity，而是训练出的 residual tail 更有害。P8-M gate 因而未触发；该结论仍属于单 seed 筛选，不应外推为普适的 anchoring 排名。

## 17. P9：`sqrt(q)` 与 population `q` 的 3×3 配对

P8 选择 Freeze-A。先运行固定 rank seed 42 的三个 training seeds 后，差值方向随 training seed 改变，因此按预设规则扩展到 training seeds `42/43/44` × rank seeds `42/43/44`。`sqrt(q)` 的 9 格复用 P6；population 的 seed42/rank42 复用既有 P6 `gamma=1`，其余 8 格使用 P9 新前缀。`gamma=1` 等于 absent slots 作为零贡献的 population weighting；等权 client 时 `sqrt(q)` 则有 `Var(sqrt(q_s) mean_s)≈sigma²/N` 的 variance-stabilization 解释。

| Training seed | Rank seed | `sqrt(q)` last-5 acc. | `q` last-5 acc. | delta `sqrt(q)-q` |
|---:|---:|---:|---:|---:|
| 42 | 42 | `.7211` | `.7162` | `+.0049` |
| 42 | 43 | `.6820` | `.6804` | `+.0016` |
| 42 | 44 | `.6970` | `.7398` | `-.0429` |
| 43 | 42 | `.7420` | `.6729` | `+.0691` |
| 43 | 43 | `.6387` | `.6038` | `+.0349` |
| 43 | 44 | `.7303` | `.6792` | `+.0511` |
| 44 | 42 | `.6926` | `.6865` | `+.0060` |
| 44 | 43 | `.5730` | `.6447` | `-.0717` |
| 44 | 44 | `.6849` | `.6789` | `+.0060` |

| Metric | `sqrt(q)` mean ± sample SD | `q` mean ± sample SD |
|---|---:|---:|
| last-5 accuracy | `.6846 ± .0519` | `.6781 ± .0387` |
| last-5 macro-F1 | `.6675 ± .0621` | `.6644 ± .0457` |
| last-5 MCC | `.6608 ± .0503` | `.6472 ± .0398` |
| last-5 loss | `1.0509 ± .0769` | `1.1529 ± .0425` |

accuracy paired delta 为 `.00656 ± .04373`，范围 `[-.07172,+.06912]`，方向为 `7/9` 支持 `sqrt(q)`。差值均值不到 1 point，且两个 rank/training cells 的反向差值超过 4 points，所以不能声称 `.5` 跨 seed 稳定最优。population `q` 同样有效，且 accuracy 离散度略低。

training-seed 边际均值中，`sqrt(q)/q` 分别为 t42 `.7000/.7121`、t43 `.7037/.6520`、t44 `.6501/.6700`；rank-seed 边际为 r42 `.7186/.6919`、r43 `.6312/.6430`、r44 `.7041/.6993`。相同 support profile 下仍有约 8.7-point 的 rank mapping 范围，明确表明剩余误差不能由 `q_s` 单独解释。

post-hoc 结果进一步区分两者：`sqrt(q)` 的 9/9 checkpoint 最优截断比例都小于 1，平均可恢复 `+.0472` accuracy 和 `+.0532` macro-F1；`q` 仅 3/9 的最优比例小于 1，平均恢复 `+.00069/+.00065`。因此 `.5` 的完整模型均值略高，但也一致保留更多有害 residual tail；`q` 更强的尾部衰减使完整模型更接近自身最佳 prefix。完整曲线、逐类 recall 和 confusion matrices 位于 [P9 evaluation directory](results/p8_p11/p9_evaluation)。

## 18. P10：double-imbalance 下的 effective support

P10 固定 Freeze-A、training/rank seed 42，使 reference discrepancy 保持约 `1e-6` 以下，从而隔离 unequal client weights 下的 support coefficient。conditional 复用 P4 checkpoint；其余三种 scaling 为新训练。

| Scaling | best/final/last-5 acc. | last-5 loss / F1 / MCC | tail energy / entropy / `R_tail` | best trunc.（acc.） |
|---|---:|---:|---:|---:|
| conditional `1` | `.7913/.7912/.7847` | `1.3111/.7308/.7717` | `.3071/.8875/.03765` | `.6 (.9436)` |
| `sqrt(q)` | `.8594/.8529/.8535` | `.5001/.8191/.8439` | `.1407/.9375/.00609` | `.6 (.8750)` |
| `c_eff` | `.8407/.8386/.8376` | `.5998/.7968/.8273` | `.1569/.9159/.00988` | `.6 (.8679)` |
| population `q` | `.8551/.8507/.8515` | `.5074/.8138/.8422` | `.1215/.9522/.00260` | `.6 (.8563)` |

`sqrt(q)` 与 population `q` 的 last-5 accuracy 只差 `.20` point；两者均优于 `c_eff`，其中 `sqrt(q)-c_eff` 为 `+1.59` accuracy points 和 `+2.23` F1 points。单 seed 下没有证据表明 effective sample size coefficient 更合理；简单的权重质量 scaling 至少同样好。它还不足以在 `sqrt(q)` 和 `q` 之间作最终选择。

slot-wise 上，所有 slots 的 `|c_eff-sqrt(q)|` 平均为 `.06614`、最大 `.16616`；只看 210 个 tail layer-slots，平均绝对差为 `.11464`。例如五组 tail `q/sqrt(q)/c_eff` 为：

```text
.58148 / .76255 / .71644
.30215 / .54968 / .69428
.25334 / .50333 / .60549
.10769 / .32817 / .49432
.08618 / .29357 / .40773
```

即 `c_eff` 在最宽的 tail group 略小于 `sqrt(q)`，在更窄且权重集中的 groups 反而更大；它不是简单的更强 shrinkage。完整 210 行保存在 [P10 coefficient comparison](results/p8_p11/p10_coefficient_comparison.csv)。

full checkpoint 的 class 8/9 recall 分别为：conditional `.234/.052`、`sqrt(q)` `.734/.044`、`c_eff` `.630/.006`、population `.784/.011`。三种 scaling 明显改善 class 8，却都没有解决 class 9；对应 confusion matrices 显示 class 9 主要被判为 3/4/7。并且所有 scaling 的最佳 post-hoc accuracy（`.856–.875`）仍低于 conditional checkpoint 截断后的 `.9436`。因此 support magnitude 已被改善，但 tail direction/label coverage 仍是主要残余问题。逐类详情与矩阵位于 [P10 evaluation directory](results/p8_p11/p10_evaluation)。

## 19. P11：rank-label coverage 的剩余影响

P8 的 3 个单 seed checkpoint和 P9 的 18 个 crossed runs 均已输出逐运行 residual diagnostics，包括 energy-weighted tail entropy、`R_tail`、最低 support 的 eligible clients/labels、最高风险 slot 的 client IDs/labels、完整 per-class recall、10×10 confusion matrix、低召回类别、过预测类别、最佳截断比例和增益，见 [P11 per-run diagnostics](results/p8_p11/p11_per_run_residual_diagnostics.csv)。

P9 的典型最高风险 slot 会随 rank mapping 变化：例如 r42 可由 `client.6`、labels `{5,6}` 主导；r43 的某些 runs 由 `client.1/4/5/6`、labels `{0,1,3,4,5,6}` 主导；r44 可由 `client.5`、labels `{4,5}` 主导。相应低召回/过预测类别也变化，而三个 rank seeds 的 rank multiset 和 `q_s` profile 完全相同。这是 rank-client-label assignment 的直接描述性证据。

在 18 个 P9 runs 上的 Pearson correlations 为：

| 描述性关系 | correlation |
|---|---:|
| `R_tail` vs last-5 accuracy | `-.211` |
| `R_tail` vs last-5 macro-F1 | `-.255` |
| tail entropy vs accuracy | `+.613` |
| tail entropy vs macro-F1 | `+.658` |
| `R_tail` vs post-hoc accuracy gain | `+.867` |
| `R_tail` vs post-hoc F1 gain | `+.880` |
| tail entropy vs post-hoc accuracy gain | `-.590` |

`R_tail` 对最终性能的线性关系较弱，但对“删掉 tail 后能恢复多少”有很强的同向关系；这更符合它作为有害尾部诊断量的定义。tail entropy 与最终 accuracy/F1 的中等正相关，以及相同 support mass 下 rank-seed 边际差异，都支持 residual performance 随 label coverage/rank mapping 变化。18 格共享 training/rank factors，且 `sqrt(q)` 与 `q` 会同时改变 `R_tail` 和 truncation gain；这些观测不是独立样本，相关不构成因果估计，也不支持现在实现 entropy-aware weighting。

## 20. 证据分层、未隔离混杂与最终回答

### 20.1 证据分层

- 已证明事实：support scaling 在 P6 的 3×3 配对中相对 conditional baseline 9/9 大幅改善；Freeze-A 在匹配 scaling 下使 factor/direct aggregate 在浮点误差内一致；相同 rank multiset 下 rank mapping 仍改变结果。
- 单 seed 筛选：P8 中 Freeze-A 完整模型明显优于 Strong-A/RefDiag；P10 中 `sqrt(q)` 和 `q` 优于 `c_eff`。这些排序尚未经过独立多 seed 验证。
- 多 seed 结果：P9 显示 `sqrt(q)` 与 `q` 的均值差不到 1 point、方向不稳定；两者都有效。
- 描述性相关：P11 的 entropy/`R_tail` correlations 与逐类错误模式支持 residual coverage 假设，但不作强因果声明。
- 未隔离混杂：heterogeneous local capacity 尚未在保持 server support/coverage 固定时单独干预；rank、client identity、label set 和部分数据权重仍共同变化；post-hoc truncation 不是重新训练；P8/P10 只有 MNIST 单 seed。

### 20.2 六个问题的结论

1. Support scaling 后，没有证据表明可训练 A 或 soft anchoring带来独立 accuracy/F1 收益。Strong-A 虽比 RefDiag 好，但 Freeze-A 在 P8 单 seed 的 last-5 accuracy/F1 又高 `12.9/15.6` points。严格表述是“当前筛选不支持 soft adaptation”，而不是多任务普适证明。
2. 当前最合适的实验基础是 Freeze-A：它性能最高、factor/direct identity 可验证，也最容易隔离 support weighting。Strong-A 可保留为软约束对照；Vanilla/RefDiag 不适合作为当前最终基础。
3. `gamma=.5` 的优势不跨 seed 稳定。它相对 `gamma=1` 平均只高 `.66` accuracy point，差值范围约 `-7.17` 到 `+6.91` points；population `q` 同样有效且方差更小。当前不应通过 MNIST gamma sweep 宣称 `.5` 最优。
4. unequal weights 的单 seed 证据更支持简单 `q_s` 系数，而不是 `N_eff`：`sqrt(q)` 和 population `q` 都优于 `c_eff`，但差值尚不足以作普适理论裁决。`N_eff` 仍是有意义的诊断量，不是当前首选聚合系数。
5. 剩余 uniform–heterogeneous gap 更接近 rank-label coverage 与有害 tail direction。reference mismatch 已在 Freeze-A 中消除；support weighting 已解释并修复主要坍塌；相同 `q_s` profile 仍有明显 rank-seed 差异，且 class 9/截断差距持续存在。local capacity 仍是未排除混杂，但现有证据尚不能把它排在 coverage 之前。
6. 已有充分证据保留“shared reference frame + support-aware aggregation”作为强候选架构和后续实验基线，但还不足以把两者都宣称为经独立验证的最终双组件方法。support-aware aggregation 有 3×3 因果复验证据；shared reference 的额外性能价值目前主要来自单 seed P8 和可辨识性/数值恒等式。更稳妥的结论是：Freeze-A 用作稳定、可解释的 reference frame，support-aware aggregation 是当前被实证支持的核心方法组件；reference frame 的普适独立收益仍应保留为问题分析与后续跨任务验证项。

### 20.3 P8–P11 结果产物

- 统一训练汇总：[p8_p11_unified_training_summary.csv](results/p8_p11/p8_p11_unified_training_summary.csv)
- P9 九格差值与聚合：[p9_paired_deltas.csv](results/p8_p11/p9_paired_deltas.csv)、[p9_summary.json](results/p8_p11/p9_summary.json)
- P10 slot-wise 系数：[p10_coefficient_comparison.csv](results/p8_p11/p10_coefficient_comparison.csv)
- P11 逐运行诊断与相关：[p11_per_run_residual_diagnostics.csv](results/p8_p11/p11_per_run_residual_diagnostics.csv)、[p11_descriptive_summary.json](results/p8_p11/p11_descriptive_summary.json)
- 训练 manifests：[P8](results/p8_p11/p8-single_run_manifest.json)、[P9 initial](results/p8_p11/p9_run_manifest.json)、[P9 matrix](results/p8_p11/p9-matrix_run_manifest.json)、[P10](results/p8_p11/p10_run_manifest.json)
- post-hoc 曲线、逐类 JSON、confusion SVG：[P8](results/p8_p11/p8_single_evaluation)、[P9](results/p8_p11/p9_evaluation)、[P10](results/p8_p11/p10_evaluation)
