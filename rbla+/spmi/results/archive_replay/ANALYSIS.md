# MNIST 归档实验的第一轮 SPMI 回放

## 主要结论

- 预先冻结的规则为：`SPMI_abs >= 0.10` 选择 RBLA+。
- 共完成 9 个第一轮回放；所有条件都被 0.10 门限判为 RBLA+。
- 有长期标签的运行共 5 个，其中 3 个通过复现审计，2 个未通过。
- 在通过复现审计的运行上，门限判断正确 0/3。
- 这说明固定 0.10 不是跨数据分布、秩范围和训练协议可迁移的绝对门限。

## 逐运行结果

| 运行 | 数据分布 | 秩范围 | SPMI_abs | 判断 | 历史赢家 | 严格复现 | 正确 |
|---|---|---|---:|---|---|---|---|
| aasb_double_low_seed42 | mnist_double_imbalance | low | 0.237944 | rbla_plus | rbla | True | False |
| aasb_double_low_seed43 | mnist_double_imbalance | low | 0.243307 | rbla_plus | rbla | True | False |
| aasb_double_low_seed44 | mnist_double_imbalance | low | 0.275786 | rbla_plus | rbla | True | False |
| compact_dirichlet_low_seed42 | mnist_dirichlet_0_1 | low | 0.328470 | rbla_plus | rbla_plus | False | None |
| compact_double_low_seed42 | mnist_double_imbalance | low | 0.283528 | rbla_plus | rbla | False | None |
| coverage_dirichlet_full_seed42 | mnist_dirichlet_0_1 | full | 0.226554 | rbla_plus | - | - | - |
| coverage_double_full_seed42 | mnist_double_imbalance | full | 0.148092 | rbla_plus | - | - | - |
| coverage_two_label_full_seed42 | mnist_two_label_balanced | full | 0.386159 | rbla_plus | - | - | - |
| coverage_two_label_low_seed42 | mnist_two_label_balanced | low | 0.438665 | rbla_plus | - | - | - |

## AASB 严格复现

- 三个种子的第一轮 accuracy/loss 均与归档完全一致；SPMI_abs 均值为 0.252346，样本标准差为 0.020476。
- 三个种子的长期赢家都是 RBLA，但第一轮 SPMI 均明显高于 0.10。
- 因此这是稳定的假阳性，而不是随机种子或回放误差造成的偶然结果。

## 数据分布与秩范围

- `mnist_double_imbalance`：low=0.283528，full=0.148092，full-low=-0.135436。
- `mnist_dirichlet_0_1`：low=0.328470，full=0.226554，full-low=-0.101916。
- `mnist_two_label_balanced`：low=0.438665，full=0.386159，full-low=-0.052506。

在当前回放中，三种分布的 low-rank SPMI 都高于对应 full-rank SPMI；two-label-balanced 又高于 Dirichlet 和 double-imbalance。这表明 SPMI 的绝对尺度明显依赖数据分布和秩配置。

## 为什么结果仍然值得保留

若事后同时使用两个未通过复现审计的 compact 标签，历史 RBLA 运行的最高分为 0.283528，历史 RBLA+ 运行的分数为 0.328470。
这在数值排序上留下一个间隔，但不能据此重新设置门限：它使用了标签，而且唯一的 RBLA+ 标签来自复现无效的运行。

## 机制解释

SPMI 衡量的是当前聚合矩阵中，原始槽位前缀相对最优 SVD 前缀损失了多少能量。它能检测“存在可改善的前缀”，但这不等价于“长期使用 canonicalization 一定提高准确率”。RBLA+ 会改变 A/B 因子坐标和后续梯度动力学；即使当轮前缀更接近最优低秩近似，长期优化轨迹仍可能变差。AASB 三个严格反例正好说明了这两种问题不能由同一个绝对几何门限直接等同。

## 解释边界

- 未通过复现审计的 compact 标签不计入判断准确率。
- 无标签覆盖条件只展示分数结构，不计为正确或错误。
- 三个 AASB 种子属于同一个数据分布—秩条件，不能当作三个独立条件。
- 长期指标只用于实验后对照，从未进入 SPMI 计算。
