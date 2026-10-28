# RBLA / RBLA+ Rank 错配检测与自适应广播设计

> 文档状态：设计提案，尚未接入生产训练流程  
> 适用范围：异构 LoRA rank 的联邦学习  
> 当前方法名称：RBLA、RBLA+（代码中的独立实现名称为 `SP+`）  
> 核心目标：在不混合客户端因子坐标系的前提下，自动判断当前轮应统一使用 RBLA 还是 RBLA+ 广播

---

## 1. 背景

### 1.1 异构 rank 联邦学习的问题

在异构 LoRA 联邦学习中，不同客户端可以使用不同的 LoRA rank。设第 \(i\) 个客户端的 LoRA 因子为：

\[
A_i\in\mathbb{R}^{r_i\times d_{\text{in}}},
\qquad
B_i\in\mathbb{R}^{d_{\text{out}}\times r_i}.
\]

该客户端产生的 LoRA 更新为：

\[
\Delta W_i=B_iA_i.
\]

当各客户端 rank \(r_i\) 不同时，服务器无法直接对所有 \(A_i\)、\(B_i\) 做普通的逐元素 FedAvg，因为这些矩阵在 rank 维度上的形状不同。

RBLA 通过 padding 和位置掩码解决形状不同的问题。对于 rank 位置 \(j\)，只有实际包含该位置的客户端才参与平均：

\[
\bar A_j=
\frac{\sum_i w_i m_{ij} A_{ij}}
{\sum_i w_i m_{ij}},
\qquad
\bar B_j=
\frac{\sum_i w_i m_{ij} B_{ij}}
{\sum_i w_i m_{ij}},
\]

其中：

- \(w_i\) 是客户端聚合权重，通常与客户端训练样本数有关；
- \(m_{ij}\in\{0,1\}\) 表示客户端 \(i\) 是否拥有 rank 位置 \(j\)；
- 没有该位置的客户端不会稀释这一位置的平均结果。

这种方法解决了形状问题，但没有自动解决 LoRA 因子的坐标或基底错位问题。

### 1.2 LoRA 因子的 gauge 非唯一性

LoRA 分解不是唯一的。对于任意可逆矩阵 \(Q\)：

\[
BA=(BQ)(Q^{-1}A).
\]

因此，两个因子对可能表示完全相同的 \(\Delta W\)，但其 rank 槽位、符号、尺度和隐空间方向不同。

这意味着：

- 客户端 1 的第 1 个 rank 方向不一定与客户端 2 的第 1 个方向语义一致；
- 直接按相同槽位聚合隐含了“槽位已经对齐”的假设；
- 对低 rank 客户端广播全局因子的前 \(r_i\) 个槽位时，这些槽位不一定是全局最重要的方向；
- 高聚合权重但低 rank 的客户端尤其容易受到影响，因为它只能接收很短的前缀。

本文将这种现象称为 **rank 前缀错配**，简称 **rank 错配**。

### 1.3 RBLA+ / SP+ 的作用

RBLA+ 在 RBLA 聚合完成后，对服务器 LoRA 因子做紧凑规范化：

1. 对 \(B\) 做薄 QR 分解；
2. 对 \(A^\top\) 做薄 QR 分解；
3. 构造 \(R\times R\) 核心矩阵；
4. 只对核心矩阵做 SVD；
5. 按奇异值降序排列方向；
6. 平衡 \(A\)、\(B\) 两侧的尺度；
7. 确定性修正每个方向的符号；
8. 向 rank 为 \(r_i\) 的客户端广播前 \(r_i\) 个规范方向。

设：

\[
B=Q_BR_B,\qquad A^\top=Q_AR_A,
\]

并定义：

\[
C=R_BR_A^\top=U\Sigma V^\top.
\]

规范化因子为：

\[
B^+=Q_BU\Sigma^{1/2},
\qquad
A^+=\Sigma^{1/2}V^\top Q_A^\top.
\]

完整 rank 下：

\[
B^+A^+=BA.
\]

因此，RBLA+ 在数值误差范围内保持完整服务器模型不变，但将 rank 槽位重新排列成按奇异值强度排序的规范前缀。

---

## 2. 设计目标

本设计希望回答：

> 当前 RBLA 聚合结果在截断到真实客户端 rank 后，原始 RBLA 前缀还是 RBLA+ 规范前缀更好？

设计目标包括：

1. 不要求提前知道 Dirichlet \(\alpha\)、数据异质性或 rank–数据量相关性；
2. 不需要人为标注某一个客户端是否错配；
3. 不对不同客户端混合广播两种坐标系；
4. 每次决策都基于当前全局模型和当前客户端 rank 分布；
5. 使用独立校准数据，不能使用最终测试集做在线决策；
6. 决策结果可解释、可记录、可复现；
7. 检测失败时安全回退到 RBLA；
8. 控制额外计算量，避免每轮都执行大量重复验证；
9. 保持现有 RBLA 和 SP+ 聚合实现可独立使用；
10. 后续能够扩展到动态客户端选择和动态 rank。

---

## 3. 非目标

本设计暂时不处理：

1. 为每个客户端独立选择 RBLA 或 RBLA+；
2. 在同一轮内混合两种坐标系上传并直接聚合；
3. 自动给每个客户端分配最优 rank；
4. 根据客户端隐私数据在服务器重建数据分布；
5. 用该检测器替代完整的联邦训练性能评估；
6. 保证即时校准收益一定转化为第 100 轮最终收益；
7. 在没有任何校准数据时提供与有校准数据相同的可靠性。

其中第 1、2 项被明确排除，是因为混合广播会使客户端下一轮上传的因子处于不同坐标系。若没有额外对齐过程，直接按槽位聚合可能重新引入更严重的 gauge 错位。

---

## 4. 核心设计原则

### 4.1 不直接“猜”哪个客户端错配

一个客户端是否错配并没有唯一、稳定的定义。可以比较：

- 数据量与 rank；
- 客户端 loss 与 rank；
- 更新范数与 rank；
- 标签多样性与 rank；
- 梯度方向与 rank；
- 客户端对全局模型的影响与 rank。

但这些启发式指标都不能直接回答“广播哪个模型前缀更好”。

因此，本设计不先分类客户端，而是直接评估两种候选广播在真实 rank 截断后的效果。

### 4.2 比较截断模型，而不是完整模型

完整 RBLA 模型和规范化后的 RBLA+ 模型满足：

\[
B^+A^+\approx BA.
\]

如果直接在完整 rank 上验证，两者的预测和 loss 应几乎相同，无法检测 rank 前缀错配。

真正产生差异的是 rank 截断：

\[
\Delta W_{\text{RBLA}}^{(r)}
=B_{[:,1:r]}A_{[1:r,:]},
\]

\[
\Delta W_{\text{RBLA+}}^{(r)}
=B^+_{[:,1:r]}A^+_{[1:r,:]}.
\]

所以检测器必须对每个实际 rank 或 rank profile 构造截断模型并分别验证。

### 4.3 每一轮只选择一种全局广播方式

决策结果必须是：

```text
本轮所有客户端统一广播 RBLA
```

或：

```text
本轮所有客户端统一广播 RBLA+
```

这样能够保证客户端在同一轮从相同类型的全局因子坐标系出发，避免混合 gauge。

### 4.4 使用 loss，而不是 accuracy 作为主要判据

推荐使用平均交叉熵 loss：

- loss 是连续值，对小幅模型差异更敏感；
- accuracy 在样本量较小时呈离散阶梯；
- 两个模型可能 accuracy 相同，但置信度和 loss 已经有明显差异；
- loss 更适合作为 0.5% 或 1% 的相对收益指标。

accuracy、F1、MCC 可以记录，但不建议作为主切换信号。

---

## 5. 输入与输出

### 5.1 输入

检测器需要：

1. RBLA 聚合后的原始全局 state dict；
2. 当前轮实际参与广播或预计下一轮参与训练的客户端集合；
3. 每个客户端的本地 LoRA rank profile；
4. 每个客户端的聚合或重要性权重；
5. 一份服务器校准数据集；
6. 当前通信轮编号；
7. 检测器历史状态。

### 5.2 rank profile

不应假设所有 LoRA 层使用完全相同的 rank。

客户端 \(i\) 的 rank profile 应表示为：

```python
RankProfile(
    layer_1_rank=...,
    layer_2_rank=...,
    layer_3_rank=...,
)
```

或者使用稳定、可哈希的结构：

```python
(
    ("layer1.lora_A", 16),
    ("layer2.lora_A", 16),
    ("layer3.lora_A", 10),
)
```

只有 rank profile 完全相同的客户端才能复用同一个截断候选模型。

### 5.3 输出

检测器至少输出：

```python
DetectionDecision(
    selected_method="rbla" | "rbla_plus",
    checked=True | False,
    raw_weighted_loss=float,
    plus_weighted_loss=float,
    relative_gain=float,
    smoothed_gain=float,
    reason=str,
)
```

如果本轮不是检查轮，应输出：

```python
checked=False
selected_method=<继续使用当前方法>
reason="check_interval"
```

---

## 6. 校准数据集

### 6.1 数据来源优先级

从优到劣推荐：

1. 与任务同分布、独立保留的公共校准集；
2. 服务器拥有的独立训练域校准集；
3. 从训练数据中预先固定划分、且从未参与本地训练的子集；
4. 经授权聚合得到的公共代理数据；
5. 合成数据或蒸馏数据。

不得使用最终测试集做在线方法选择。测试集只能用于离线报告。

### 6.2 样本数量

MNIST 控制实验中：

- 200 个样本已经能识别明显错配；
- 阈值附近的 0.8%–1.1% gain 会随样本划分发生小幅波动；
- 实际使用建议 500–1,000 个样本。

默认建议：

```yaml
calibration:
  sample_count: 1000
```

### 6.3 类别平衡

对于分类任务，校准集应尽量类别平衡。

MNIST 十分类任务中，1,000 个校准样本可以尽量保持每类约 100 个样本。否则，如果校准集被少数类别主导，检测器可能选择只对这些类别更有利的前缀。

### 6.4 固定性

校准集应：

- 在整个实验中固定；
- 固定样本顺序；
- `shuffle=False`；
- 记录样本索引或数据版本；
- 不在不同方法之间变化；
- 不进行随机增强，或者使用完全确定性的增强。

### 6.5 模型评估模式

评估时必须：

```python
model.eval()
torch.inference_mode()
```

并确保：

- Dropout 关闭；
- BatchNorm 使用已有统计量，不更新 running statistics；
- 不计算梯度；
- 不修改 optimizer；
- 不修改候选 state dict；
- 两个候选使用完全相同的数据批次和顺序。

---

## 7. 候选模型构造

### 7.1 保存 RBLA 原始聚合结果

RBLA 聚合完成后，首先保留原始结果：

```python
raw_global_state = clone_state_dict(aggregated_state)
```

这里必须是安全副本，不能让后续规范化覆盖原始 RBLA 结果。

### 7.2 构造 RBLA+ 候选

对原始结果的副本进行规范化：

```python
plus_result = canonicalize_lora_state_dict(raw_global_state)
plus_global_state = plus_result.state_dict
```

规范化应使用与正式 SP+ 相同的参数：

```yaml
canonicalization:
  enabled: true
  deterministic_sign: true
  svd_fallback: true
  ordering: singular_value
```

### 7.3 完整模型等价性检查

在开发和诊断模式下，应检查每一个 LoRA 因子对：

\[
\epsilon_{\text{full}}
=
\frac{\|BA-B^+A^+\|_F}
{\max(\|BA\|_F,\varepsilon)}.
\]

建议阈值：

```text
float32: epsilon_full <= 1e-5
float64: epsilon_full <= 1e-10
```

如果超过阈值：

1. 记录异常；
2. 本轮禁止使用 RBLA+；
3. 回退到 RBLA；
4. 不继续比较截断候选；
5. 可选择中止训练，具体取决于严格模式配置。

生产环境不一定要显式构造稠密 \(BA\)。可以使用现有 canonicalization diagnostics 中的核心重构误差作为代理。

### 7.4 按 rank profile 截断

对每一个唯一 rank profile \(g\)，分别构造：

```python
raw_candidate[g] = fit_global_to_profile(raw_global_state, g)
plus_candidate[g] = fit_global_to_profile(plus_global_state, g)
```

对于标准 LoRA：

- `lora_A` 的 rank 维是第 0 维；
- `lora_B` 的 rank 维是第 1 维。

若本地 rank 小于全局 rank：

```python
local_A = global_A[:local_rank, :]
local_B = global_B[:, :local_rank]
```

若本地 rank 大于规范化后某层的有效全局 rank：

- 保留已有部分；
- 剩余部分用零填充；
- 保证 `load_state_dict(strict=True)` 可成功；
- 记录 zero-padded rank 数量。

### 7.5 非 LoRA 参数

非 LoRA 参数必须在两个候选中保持相同：

- 不能因为检测流程重复平均；
- 不能重新初始化；
- 不能分别更新 BatchNorm；
- 不能改变 bias；
- 不能改变基础模型权重。

如果完整候选在非 LoRA 参数上不同，应视为实现错误。

---

## 8. rank-aware 加权评分

### 8.1 客户端权重

设当前用于决策的客户端集合为 \(\mathcal{C}\)，客户端权重为 \(w_i\)。

推荐直接使用下一轮聚合预计使用的权重定义。例如：

\[
w_i=n_i,
\]

其中 \(n_i\) 为客户端样本数。

归一化：

\[
\tilde w_i=
\frac{w_i}
{\sum_{j\in\mathcal C}w_j}.
\]

### 8.2 按 rank profile 分组

对于唯一 rank profile \(g\)，定义该组权重：

\[
p_g=
\sum_{i\in\mathcal C:\,g_i=g}\tilde w_i.
\]

显然：

\[
\sum_g p_g=1.
\]

### 8.3 每个 profile 的 loss

在相同校准集 \(\mathcal D_{\text{cal}}\) 上计算：

\[
L_{\text{raw}}(g)
=
\frac{1}{|\mathcal D_{\text{cal}}|}
\sum_{(x,y)\in\mathcal D_{\text{cal}}}
\ell(f(x;\theta_{\text{raw}}^{(g)}),y),
\]

\[
L_{\text{plus}}(g)
=
\frac{1}{|\mathcal D_{\text{cal}}|}
\sum_{(x,y)\in\mathcal D_{\text{cal}}}
\ell(f(x;\theta_{\text{plus}}^{(g)}),y).
\]

### 8.4 全局加权 loss

\[
L_{\text{RBLA}}
=
\sum_g p_g L_{\text{raw}}(g),
\]

\[
L_{\text{RBLA+}}
=
\sum_g p_g L_{\text{plus}}(g).
\]

这里的权重非常重要。如果 80% 的训练数据位于低 rank 客户端，低 rank 截断结果应在决策中占更大权重。

### 8.5 相对收益

定义：

\[
\text{gain}_t
=
\frac{L_{\text{RBLA},t}-L_{\text{RBLA+},t}}
{\max(L_{\text{RBLA},t},\varepsilon)}.
\]

解释：

- `gain > 0`：RBLA+ 截断候选的校准 loss 更低；
- `gain = 0`：两者基本等价；
- `gain < 0`：RBLA 原始前缀更好；
- `gain = 0.01`：RBLA+ 相对降低约 1% 的加权校准 loss。

默认：

```python
epsilon = 1e-12
```

若 `raw_loss` 非有限值或小于异常阈值，应进入异常处理，而不是直接计算 gain。

---

## 9. 决策状态机

仅使用单轮 gain 直接切换会导致方法在阈值附近抖动。推荐使用：

1. warmup；
2. 检查间隔；
3. 指数滑动平均；
4. 双阈值；
5. 连续确认；
6. 最短驻留时间。

### 9.1 默认参数

```yaml
rank_mismatch_detection:
  enabled: true

  warmup_rounds: 20
  check_interval: 5

  calibration_sample_count: 1000
  calibration_batch_size: 256

  ema_decay: 0.8

  enter_plus_threshold: 0.01
  exit_plus_threshold: 0.005

  enter_confirmation_checks: 2
  exit_confirmation_checks: 2

  minimum_residence_rounds: 10
  default_method: rbla

  fallback_on_error: rbla
```

### 9.2 warmup

前 20 轮固定使用 RBLA：

```text
round < warmup_rounds -> RBLA
```

原因：

- 现有实验中 RBLA+ 前期收敛略慢；
- 初始模型的奇异方向仍可能快速变化；
- 过早频繁规范化可能扰动因子空间优化；
- warmup 后再检测更符合“出现错配才纠正”的目标。

### 9.3 检查间隔

只在以下轮次运行完整检测：

```python
if (round - warmup_rounds) % check_interval == 0:
    run_detection()
else:
    keep_current_method()
```

默认每 5 轮检查一次。

### 9.4 EMA 平滑

\[
\bar g_t
=
\beta \bar g_{t-1}
+(1-\beta)g_t,
\]

默认：

\[
\beta=0.8.
\]

第一次有效检查时，可以令：

\[
\bar g_t=g_t.
\]

### 9.5 双阈值

进入 RBLA+：

```text
smoothed_gain > 1%
```

退出 RBLA+：

```text
smoothed_gain < 0.5%
```

0.5%–1% 是滞后区间。在该区间内保持当前方法。

### 9.6 连续确认

进入 RBLA+ 前要求连续两次检查满足进入条件：

```text
连续两次 smoothed_gain > 1%
```

退出前同样要求连续两次检查满足退出条件：

```text
连续两次 smoothed_gain < 0.5%
```

如果 `check_interval=5`，连续两次意味着至少观察约 5 个通信轮的趋势。

### 9.7 最短驻留时间

切换后至少维持 10 轮：

```text
round - last_switch_round < minimum_residence_rounds
-> 禁止切换
```

### 9.8 状态定义

推荐状态：

```python
class DetectionState:
    selected_method: Literal["rbla", "rbla_plus"]
    smoothed_gain: float | None
    consecutive_enter_checks: int
    consecutive_exit_checks: int
    last_check_round: int | None
    last_switch_round: int | None
    total_switch_count: int
```

### 9.9 状态转移

```text
                 gain > enter threshold
       +---------------------------------------+
       |                                       v
   [RBLA]                                  [RBLA+]
       ^                                       |
       |                                       |
       +---------------------------------------+
                 gain < exit threshold
```

两条转移都必须同时满足：

- 连续确认次数；
- 最短驻留时间；
- 当前不是异常回退状态。

---

## 10. 完整伪代码

```python
def after_rbla_aggregation(
    round_index,
    raw_aggregated_state,
    active_clients,
    calibration_loader,
    detector_state,
    config,
):
    # ---------------------------------------------------------
    # 1. Warmup：不运行检测
    # ---------------------------------------------------------
    if round_index < config.warmup_rounds:
        detector_state.selected_method = "rbla"
        return raw_aggregated_state, make_decision(
            selected_method="rbla",
            checked=False,
            reason="warmup",
        )

    # ---------------------------------------------------------
    # 2. 非检查轮：保持当前方法
    # ---------------------------------------------------------
    if (round_index - config.warmup_rounds) % config.check_interval != 0:
        selected_state = materialize_current_method(
            detector_state.selected_method,
            raw_aggregated_state,
        )
        return selected_state, make_decision(
            selected_method=detector_state.selected_method,
            checked=False,
            reason="check_interval",
        )

    try:
        # -----------------------------------------------------
        # 3. 生成两个完整候选
        # -----------------------------------------------------
        raw_state = clone_state_dict(raw_aggregated_state)
        plus_result = canonicalize_lora_state_dict(raw_state)
        plus_state = plus_result.state_dict

        # -----------------------------------------------------
        # 4. 数值诊断
        # -----------------------------------------------------
        validate_canonicalization_diagnostics(
            plus_result.diagnostics,
            maximum_core_error=config.maximum_core_error,
            maximum_balance_error=config.maximum_balance_error,
        )

        # -----------------------------------------------------
        # 5. 构造 rank profile 及其权重
        # -----------------------------------------------------
        profile_weights = group_client_weights_by_rank_profile(
            active_clients
        )

        # -----------------------------------------------------
        # 6. 对每个唯一 profile 只评估一次
        # -----------------------------------------------------
        raw_weighted_loss = 0.0
        plus_weighted_loss = 0.0
        profile_details = []

        for profile, profile_weight in profile_weights.items():
            raw_local_state = fit_global_to_profile(raw_state, profile)
            plus_local_state = fit_global_to_profile(plus_state, profile)

            raw_loss = evaluate_loss(
                raw_local_state,
                calibration_loader,
            )
            plus_loss = evaluate_loss(
                plus_local_state,
                calibration_loader,
            )

            require_finite(raw_loss, plus_loss)

            raw_weighted_loss += profile_weight * raw_loss
            plus_weighted_loss += profile_weight * plus_loss

            profile_details.append(
                {
                    "profile": profile,
                    "weight": profile_weight,
                    "raw_loss": raw_loss,
                    "plus_loss": plus_loss,
                    "relative_gain": (
                        raw_loss - plus_loss
                    ) / max(raw_loss, config.epsilon),
                }
            )

        # -----------------------------------------------------
        # 7. 当前检查的总 gain
        # -----------------------------------------------------
        current_gain = (
            raw_weighted_loss - plus_weighted_loss
        ) / max(raw_weighted_loss, config.epsilon)

        # -----------------------------------------------------
        # 8. 更新 EMA
        # -----------------------------------------------------
        if detector_state.smoothed_gain is None:
            detector_state.smoothed_gain = current_gain
        else:
            detector_state.smoothed_gain = (
                config.ema_decay * detector_state.smoothed_gain
                + (1.0 - config.ema_decay) * current_gain
            )

        smoothed_gain = detector_state.smoothed_gain
        detector_state.last_check_round = round_index

        # -----------------------------------------------------
        # 9. 更新确认计数
        # -----------------------------------------------------
        if smoothed_gain > config.enter_plus_threshold:
            detector_state.consecutive_enter_checks += 1
        else:
            detector_state.consecutive_enter_checks = 0

        if smoothed_gain < config.exit_plus_threshold:
            detector_state.consecutive_exit_checks += 1
        else:
            detector_state.consecutive_exit_checks = 0

        # -----------------------------------------------------
        # 10. 检查最短驻留时间
        # -----------------------------------------------------
        residence_satisfied = (
            detector_state.last_switch_round is None
            or round_index - detector_state.last_switch_round
               >= config.minimum_residence_rounds
        )

        # -----------------------------------------------------
        # 11. RBLA -> RBLA+
        # -----------------------------------------------------
        if (
            detector_state.selected_method == "rbla"
            and residence_satisfied
            and detector_state.consecutive_enter_checks
                >= config.enter_confirmation_checks
        ):
            detector_state.selected_method = "rbla_plus"
            detector_state.last_switch_round = round_index
            detector_state.total_switch_count += 1
            detector_state.consecutive_enter_checks = 0
            detector_state.consecutive_exit_checks = 0
            reason = "confirmed_rank_prefix_mismatch"

        # -----------------------------------------------------
        # 12. RBLA+ -> RBLA
        # -----------------------------------------------------
        elif (
            detector_state.selected_method == "rbla_plus"
            and residence_satisfied
            and detector_state.consecutive_exit_checks
                >= config.exit_confirmation_checks
        ):
            detector_state.selected_method = "rbla"
            detector_state.last_switch_round = round_index
            detector_state.total_switch_count += 1
            detector_state.consecutive_enter_checks = 0
            detector_state.consecutive_exit_checks = 0
            reason = "canonical_prefix_gain_disappeared"

        else:
            reason = "keep_current_method"

        # -----------------------------------------------------
        # 13. 本轮全局选择
        # -----------------------------------------------------
        if detector_state.selected_method == "rbla_plus":
            selected_state = plus_state
        else:
            selected_state = raw_state

        return selected_state, make_decision(
            selected_method=detector_state.selected_method,
            checked=True,
            raw_weighted_loss=raw_weighted_loss,
            plus_weighted_loss=plus_weighted_loss,
            relative_gain=current_gain,
            smoothed_gain=smoothed_gain,
            reason=reason,
            profile_details=profile_details,
        )

    except Exception as error:
        # -----------------------------------------------------
        # 14. 安全回退
        # -----------------------------------------------------
        detector_state.selected_method = config.fallback_on_error
        return raw_aggregated_state, make_decision(
            selected_method="rbla",
            checked=True,
            reason="detection_error_fallback",
            error_type=type(error).__name__,
            error_message=str(error),
        )
```

---

## 11. 与当前训练流程的接入位置

推荐顺序：

```text
客户端本地训练
    ↓
客户端上传 state dict
    ↓
RBLA mask-normalized aggregation
    ↓
保留 raw global state
    ↓
如需要检查：
    ├─ 构造 RBLA 截断候选
    ├─ 构造 RBLA+ 截断候选
    └─ rank-aware calibration
    ↓
状态机决定本轮统一方法
    ↓
选择 raw state 或 canonical state
    ↓
按每个客户端本地形状切片/补零
    ↓
广播
```

当前代码结构中：

- RBLA 聚合器负责 mask-normalized 聚合；
- SP+ 是“RBLA 聚合后强制启用 canonicalization”的独立方法；
- 两者使用相同的按本地 shape 拟合广播逻辑；
- 检测器应位于 RBLA 聚合结束与正式广播之间；
- 检测器不应修改客户端训练器；
- 检测器不应让同一轮中的不同客户端选择不同全局候选。

建议新增独立组件，而不是把大量判断写入聚合器：

```text
RankMismatchDetector
AdaptiveRblaBroadcastPolicy
RankAwareCalibrationEvaluator
```

这样便于：

- 单元测试；
- 关闭功能；
- 替换评分方式；
- 独立记录状态；
- 保持 RBLA 和 SP+ 原实现清晰。

---

## 12. 计算优化

### 12.1 只评估唯一 rank profile

如果 10 个客户端的 rank ratios 分别为：

```text
0.1, 0.2, ..., 1.0
```

最多有 10 个唯一 profile，需要评估：

```text
10 个 RBLA 截断模型
+ 10 个 RBLA+ 截断模型
= 20 个候选
```

如果多个客户端共享 rank，只评估一次。

### 12.2 降低检查频率

默认每 5 轮检查一次，额外验证成本被摊薄：

\[
\text{平均每轮候选评估数}
=
\frac{2\times|\mathcal G|}{5}.
\]

若有 10 种 profile，平均相当于每轮 4 个小型候选验证。

### 12.3 小校准集

推荐 500–1,000 个样本，不必使用完整测试规模。

### 12.4 缓存

可缓存：

- rank profile 到模型结构的映射；
- 校准 DataLoader；
- 每个 profile 的本地 shape 模板；
- 非 LoRA 参数；
- profile 权重，直到客户端集合或 rank 发生变化。

不能跨检查轮缓存候选 loss，因为全局模型已经改变。

### 12.5 避免重复模型初始化

可以为每个唯一 profile 预先创建一个评估模型，然后反复加载不同候选 state dict：

```python
evaluation_models[profile].load_state_dict(candidate)
```

评估结束后必须保证模型没有残留梯度或训练状态。

### 12.6 可选的两阶段筛选

如果模型很大，可以先用 100 个样本做快速筛选：

```text
|gain| < 0.2% -> 直接保持当前方法
|gain| >= 0.2% -> 使用完整 1,000 样本确认
```

该优化需要额外验证后再启用，初版不建议加入。

---

## 13. 日志设计

每次检查至少记录：

```text
round
checked
selected_method
previous_method
switched
reason

raw_weighted_loss
plus_weighted_loss
relative_gain
smoothed_gain

enter_threshold
exit_threshold
enter_confirmation_count
exit_confirmation_count

unique_rank_profile_count
evaluated_candidate_count
calibration_sample_count

canonicalization_layer_count
canonicalization_mean_effective_rank
canonicalization_maximum_core_reconstruction_error
canonicalization_maximum_factor_balance_error

detection_duration_seconds
total_switch_count
```

建议另外保存每个 profile：

```text
profile_id
profile_weight
client_count
raw_loss
plus_loss
relative_gain
raw_accuracy
plus_accuracy
```

### 13.1 推荐 CSV 字段

```csv
round,
rank_mismatch_checked,
rank_mismatch_selected_method,
rank_mismatch_switched,
rank_mismatch_reason,
rank_mismatch_raw_loss,
rank_mismatch_plus_loss,
rank_mismatch_relative_gain,
rank_mismatch_smoothed_gain,
rank_mismatch_unique_profile_count,
rank_mismatch_candidate_count,
rank_mismatch_detection_seconds
```

### 13.2 可解释性

当选择 RBLA+ 时，应能够回答：

```text
第 40 轮选择 RBLA+：
- raw weighted loss = 0.8231
- plus weighted loss = 0.8074
- current gain = 1.91%
- smoothed gain = 1.36%
- 连续 2 次超过 1%
- 低 rank profile 0.1–0.4 贡献了主要收益
```

---

## 14. 异常处理与安全回退

### 14.1 canonicalization 失败

可能原因：

- 输入包含 NaN/Inf；
- A/B rank 不一致；
- LoRA pair 不完整；
- SVD 失败；
- rank 超过层维度；
- dtype 不支持。

处理：

```text
记录异常 -> 本轮使用 RBLA -> 不切换状态 -> 可选告警
```

### 14.2 候选 loss 非有限

如果：

```python
not isfinite(raw_loss) or not isfinite(plus_loss)
```

不允许计算 gain。

建议：

- raw 有限、plus 非有限：使用 RBLA；
- raw 非有限、plus 有限：记录严重异常，可使用 RBLA+，但默认建议中止或进入严格恢复流程；
- 两者都非有限：中止训练或恢复检查点。

### 14.3 所有客户端都是 full rank

如果所有 profile 都等于完整全局 rank：

- 两个候选应函数等价；
- 检测 gain 应接近 0；
- 可以跳过检测并使用 RBLA；
- 没有必要为 full-rank-only 场景支付额外验证成本。

### 14.4 只有一种低 rank

检测仍然有效：

\[
L_{\text{RBLA}}=L_{\text{raw}}(r),
\qquad
L_{\text{RBLA+}}=L_{\text{plus}}(r).
\]

### 14.5 阈值附近波动

如果 gain 长期位于 0.5%–1%：

- 不切换；
- 保持当前方法；
- 依赖 EMA 和滞后区间；
- 在日志中标记为 `hysteresis_hold`。

### 14.6 客户端集合变化

如果每轮随机选择不同客户端，则 profile 权重也会变化。

可选权重范围：

1. 下一轮预计参与客户端；
2. 当前轮实际参与客户端；
3. 全体客户端的长期分布。

初版建议使用 **当前轮实际参与聚合的客户端**，因为这些权重与当前全局结果对应。若客户端选择变化剧烈，可以使用最近 \(K\) 轮 profile 权重的滑动平均。

### 14.7 rank 动态变化

如果客户端 rank 会动态调整：

- 每次检查重新构造 profile；
- profile cache 按结构 key 失效；
- 不能只依赖实验初始 YAML；
- 日志记录实际有效 rank，而不是配置声明值。

### 14.8 重复或接近重复奇异值

奇异值重复时，子空间内部仍存在不可消除的正交旋转自由度。确定性符号修正不能唯一确定重复奇异值子空间中的基。

检测器不应假设 canonicalization 在数学上永远产生唯一方向。最终是否使用规范前缀仍由校准 loss 决定。

---

## 15. 隐私与安全

### 15.1 服务器校准模式

若服务器拥有公共校准集：

- 不需要额外客户端信息；
- 只使用 rank、聚合权重和全局模型；
- 不新增原始数据上传；
- 是首选模式。

### 15.2 客户端协作校准模式

若没有服务器校准集，可以让客户端分别评估两个截断候选，只上传标量：

```text
raw_loss
plus_loss
sample_count
```

但这会引入：

- 双倍候选下行；
- 客户端额外推理；
- 标量 loss 的隐私问题；
- 恶意客户端操纵选择；
- 不同客户端数据不可比；
- 安全聚合需求。

该模式不属于初版范围。

### 15.3 测试集泄漏

绝不能：

- 用测试集在线选择 RBLA/RBLA+；
- 在多轮训练中反复查看测试指标后调阈值；
- 用最终报告数据作为 calibration；
- 把同一数据同时作为 calibration 和最终无偏测试。

---

## 16. 测试计划

### 16.1 单元测试

#### 测试 A：完整模型等价

给定随机合法 LoRA 因子：

```python
raw = B @ A
plus = canonicalize(B, A)
assert relative_error(raw, plus.B @ plus.A) < tolerance
```

覆盖：

- float32；
- float64；
- 不同 rank；
- 不同输入输出维度；
- 奇异值接近重复；
- rank=1；
- rank=min(d_in, d_out)。

#### 测试 B：完全对齐不误报

以已经按奇异值排序的规范因子作为 raw：

```text
raw prefix loss ≈ plus prefix loss
gain ≈ 0
selected method = RBLA
```

#### 测试 C：排列错配

随机排列规范方向，保持完整 \(BA\) 不变：

```text
raw full model == plus full model
raw truncated model worse
gain > enter threshold
selected method = RBLA+
```

#### 测试 D：正交旋转错配

应用随机正交 \(Q\)：

\[
B'=BQ,\qquad A'=Q^\top A.
\]

验证检测器能识别截断差异。

#### 测试 E：阈值边界

构造不同旋转角度，验证：

- 小于 1% 时保持 RBLA；
- 大于 1% 且连续确认后切换；
- 0.5%–1% 时滞后保持；
- 不发生频繁切换。

#### 测试 F：异常回退

注入：

- NaN；
- Inf；
- A/B rank mismatch；
- 缺失 pair；
- calibration 空集；
- loss 非有限。

期望：

```text
selected method = RBLA
reason = detection_error_fallback
```

### 16.2 MNIST 控制测试

已经完成的独立后验探针使用：

- 58,000 个训练样本；
- 2,000 个校准样本；
- 10,000 个测试样本；
- 三层 MLP；
- 完整 rank 测试准确率约 97.4%；
- 六组现有 MNIST rank/数据量配置；
- 五种旋转强度和三种严重错配方式。

边界结果：

| 错配 | 测试集 loss 改善 | 200 样本校准改善 | 1% 决策 |
|---|---:|---:|---|
| 完全对齐 | 约 0% | 约 0% | RBLA |
| 1° 旋转 | 0.42%–0.86% | 0.33%–0.69% | RBLA |
| 3° 旋转 | 1.35%–2.69% | 1.08%–2.23% | RBLA+ |
| 6° 旋转 | 2.94%–5.73% | 2.48%–4.96% | RBLA+ |
| 12° 旋转 | 6.92%–12.62% | 6.48%–12.03% | RBLA+ |

严重错配：

| 错配 | 测试集 loss 改善 |
|---|---:|
| 随机排列 | 52.0%–64.2% |
| 完全反序 | 58.4%–70.4% |
| 随机正交混合 | 43.7%–54.1% |

这些结果证明检测器能够识别当前截断前缀质量，但还不构成完整联邦训练最终收益的证据。

### 16.3 端到端集成测试

至少运行：

1. 固定 RBLA；
2. 固定 RBLA+；
3. 自适应检测；
4. 自适应检测但强制永不切换，用于 A/A 验证；
5. 自适应检测但只记录、不执行切换，用于 shadow mode。

建议先运行 shadow mode：

```text
检测器正常计算和记录决策
训练仍固定使用 RBLA
```

确认：

- 没有修改训练结果；
- 检测结果稳定；
- 额外耗时可接受；
- 日志完整；
- 不出现 NaN；
- profile 权重计算正确。

### 16.4 建议实验矩阵

MNIST：

```text
Dirichlet alpha: 0.4, 0.8
rank-volume rho: -0.5, 0, +0.5
seed: 42, 43, 44, 45, 46
method:
  - fixed_rbla
  - fixed_rbla_plus
  - adaptive
```

总计：

\[
2\times3\times5\times3=90
\]

个实验。

### 16.5 主要评估指标

最终效果：

- 第 100 轮 accuracy；
- 最后 10 轮平均 accuracy；
- 最后 10 轮标准差；
- 最佳 accuracy；
- loss；
- macro-F1；
- MCC。

检测质量：

- 实际切换次数；
- 首次切换轮次；
- RBLA+ 使用轮数占比；
- 每次切换前后的短期收益；
- calibration gain 与下一检查周期真实收益的相关性；
- 检测耗时；
- 总训练时间增加比例。

---

## 17. 成功标准

初版可以采用以下标准。

### 17.1 正确性

- 完全对齐输入不误报；
- 严重排列/正交错配能稳定触发；
- canonicalization 异常时安全回退；
- 同一轮所有客户端使用统一候选；
- profile 加权和为 1；
- 不使用测试集做在线选择。

### 17.2 稳定性

- 单个实验切换次数不超过 4 次；
- 不发生相邻检查轮反复切换；
- 不引入新的 NaN/Inf；
- 检测器关闭时与原 RBLA bitwise 或数值等价。

### 17.3 性能

- 相比固定 RBLA，跨条件平均最终 accuracy 不下降；
- 在固定 RBLA+ 退化的条件中能更多保留 RBLA；
- 在 rank 前缀错配明显的条件中能切换到 RBLA+；
- 额外训练耗时建议不超过 10%，目标不超过 5%。

### 17.4 泛化

在至少 5 个随机种子上报告：

- 均值；
- 标准差；
- 配对差值；
- 置信区间。

不能将不同 \(\alpha\)、ρ 条件当作随机种子替代重复实验。

---

## 18. 推荐上线步骤

### 阶段 1：离线探针

- 使用保存的全局 checkpoint；
- 后验构造 raw/plus 截断候选；
- 验证 calibration gain 与测试 gain；
- 不修改训练。

### 阶段 2：shadow mode

- 在线计算检测结果；
- 固定继续使用 RBLA；
- 记录如果执行切换会选择什么；
- 检查成本和稳定性。

### 阶段 3：受控切换

- 只允许 RBLA → RBLA+；
- 不允许切回；
- 限定 MNIST；
- 使用 warmup 和连续确认。

### 阶段 4：完整双向状态机

- 启用 RBLA ↔ RBLA+；
- 加入最短驻留和双阈值；
- 多种子评估。

### 阶段 5：扩展任务

- FMNIST；
- KMNIST；
- QMNIST；
- CNN；
- NLP/Transformer LoRA；
- 动态客户端选择；
- 动态 rank。

---

## 19. 推荐配置

首个正式 MNIST 实验建议：

```yaml
rank_mismatch_detection:
  enabled: true
  mode: rank_aware_calibration

  warmup_rounds: 20
  check_interval: 5

  calibration_sample_count: 1000
  calibration_batch_size: 256
  calibration_shuffle: false
  primary_metric: cross_entropy_loss

  client_weighting: aggregation_weight
  group_by: effective_rank_profile

  ema_decay: 0.8

  enter_plus_threshold: 0.01
  exit_plus_threshold: 0.005
  enter_confirmation_checks: 2
  exit_confirmation_checks: 2
  minimum_residence_rounds: 10

  default_method: rbla
  fallback_on_error: rbla

  canonicalization:
    deterministic_sign: true
    svd_fallback: true
    ordering: singular_value

  diagnostics:
    enabled: true
    log_profile_metrics: true
    verify_full_equivalence: true
    maximum_core_reconstruction_error: 1.0e-5
    maximum_factor_balance_error: 1.0e-5
```

---

## 20. 简化版实施方案

如果初版不希望引入 EMA 和完整状态机，可以先实现：

```python
if round < 20:
    method = "rbla"
elif round % 5 != 0:
    method = previous_method
else:
    gain = rank_aware_calibration_gain()

    if gain > 0.01:
        method = "rbla_plus"
    else:
        method = "rbla"
```

但该版本必须至少增加：

- 两次连续确认；
- 统一全局广播；
- 异常回退；
- 独立校准集；
- 详细日志。

完整状态机仍是推荐方案。

---

## 21. 常见问题

### 21.1 为什么不直接检查 rank 与数据量的 Spearman ρ？

ρ 只能检测“高数据量客户端是否得到高 rank”，不能检测客户端更新子空间是否已经错位。

现有实验中，Dirichlet \(\alpha\) 对 RBLA+ 收益的影响比 ρ 更明显。因此只用 ρ 会漏掉：

- rank–数据量正相关但更新方向高度错位；
- 数据量相近但非 IID 很强；
- rank 看似合理但原始前缀顺序较差。

rank-aware calibration 直接比较最终要广播的截断模型，信息更完整。

### 21.2 为什么不直接比较奇异值能量？

RBLA+ 前缀按 SVD 排序，在 Frobenius 范数意义下通常具有更好的低 rank 重构性质，但：

- 参数误差不等于任务 loss；
- 不同输入方向的重要性不同；
- 分类边界对不同奇异方向的敏感度不同；
- 相同参数重构误差可能有不同预测结果。

所以奇异值能量适合作为诊断，不应替代任务校准 loss。

### 21.3 为什么 RBLA+ 完整模型等价，训练结果仍可能不同？

因为因子空间中的 SGD 动态依赖具体分解：

\[
\nabla_A L=B^\top\nabla_{\Delta W}L,
\qquad
\nabla_B L=\nabla_{\Delta W}L A^\top.
\]

同一个 \(BA\) 的不同因子表示会产生不同的下一步 \(A/B\) 梯度。规范化保持当前函数，但不保证后续优化轨迹不变。

### 21.4 为什么需要 warmup？

实验显示 RBLA+ 前期宏平均略慢，主要收益在较后轮次出现。warmup 可减少早期不必要的因子旋转。

### 21.5 200 个校准样本是否足够？

对明显错配足够；对 1% 左右的边界收益可能不稳定。正式实验建议使用 500–1,000 个样本，并要求连续两次确认。

### 21.6 如果 RBLA+ 只改善低 rank 客户端，但高 rank 客户端略差怎么办？

加权评分会按客户端聚合权重综合两者。如果低 rank 客户端权重高，其收益会自然占据更大比例。日志中的 profile 级结果应保留，以便检查是否存在不公平问题。

### 21.7 能否给错配客户端用 RBLA+，其他客户端用 RBLA？

不建议在没有上传对齐机制时这样做。两组客户端下一轮会处于不同因子坐标系，直接 RBLA 聚合可能混合不对应的槽位。

本设计选择整轮统一方法，正是为了避免该问题。

---

## 22. 实施检查清单

### 数据

- [ ] 使用独立 calibration，不使用 test；
- [ ] calibration 固定且可复现；
- [ ] 分类任务尽量类别平衡；
- [ ] 记录 calibration 数据版本和索引；
- [ ] `shuffle=False`；
- [ ] 不使用随机增强。

### 候选模型

- [ ] 保存 raw RBLA state；
- [ ] 对副本做 canonicalization；
- [ ] 验证完整模型等价；
- [ ] 非 LoRA 参数完全一致；
- [ ] 按真实有效 rank profile 截断；
- [ ] 大于全局 rank 时正确零填充；
- [ ] `load_state_dict(strict=True)` 成功。

### 评分

- [ ] 每个唯一 profile 只评估一次；
- [ ] profile 权重总和为 1；
- [ ] 使用与聚合一致的客户端权重；
- [ ] 主要指标使用平均 loss；
- [ ] loss 和 gain 均为有限值；
- [ ] accuracy/F1 仅作辅助日志。

### 状态机

- [ ] warmup；
- [ ] check interval；
- [ ] EMA；
- [ ] 双阈值；
- [ ] 连续确认；
- [ ] 最短驻留时间；
- [ ] 切换状态可持久化；
- [ ] 恢复 checkpoint 时恢复检测状态。

### 安全

- [ ] 异常时回退 RBLA；
- [ ] 同轮统一广播；
- [ ] 不混合客户端坐标系；
- [ ] canonicalization NaN/Inf 检查；
- [ ] 检测器关闭时不改变原训练结果；
- [ ] 日志不泄露客户端原始数据。

### 验证

- [ ] 单元测试；
- [ ] MNIST 后验探针；
- [ ] shadow mode；
- [ ] 端到端多种子实验；
- [ ] 报告额外耗时；
- [ ] 报告切换次数和轮次；
- [ ] 对固定 RBLA、固定 RBLA+、adaptive 做配对比较。

---

## 23. 最终推荐

推荐采用以下最小可靠策略：

1. 始终先执行 RBLA 聚合；
2. 前 20 轮固定广播 RBLA；
3. 从第 20 轮开始每 5 轮生成一次 RBLA+ 规范候选；
4. 对当前唯一 rank profile 分别截断 raw 和 plus；
5. 使用 500–1,000 个固定、平衡的校准样本计算加权交叉熵；
6. 计算相对 loss gain；
7. gain 的 EMA 连续两次超过 1% 后，全局切换到 RBLA+；
8. gain 的 EMA 连续两次低于 0.5% 后，全局切回 RBLA；
9. 每次切换后至少保持 10 轮；
10. 任意检测或规范化异常都回退 RBLA；
11. 同一轮所有客户端统一使用同一种广播坐标系；
12. 先以 shadow mode 验证，再启用真实切换。

该方案不试图用单一统计量猜测“哪个客户端错配”，而是直接测试当前真实 rank 分布下哪种广播前缀具有更低的任务 loss。它在实现复杂度、可解释性、数值安全和实验可信度之间提供了较好的平衡。
