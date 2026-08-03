# SPMI 算法（精简版）

SPMI（Spectral Prefix Mismatch Index）用于衡量 RBLA 聚合后的原始 rank
前缀，相比最优低秩前缀损失了多少更新能量。它不需要服务器数据集，只读取
全局 LoRA 因子、客户端 rank 和聚合权重。

## 1. 输入

对于每个 LoRA 层 \(l\)：

\[
A_l\in\mathbb{R}^{R_l\times d_{\mathrm{in}}},
\qquad
B_l\in\mathbb{R}^{d_{\mathrm{out}}\times R_l}.
\]

还需要：

- 客户端 \(i\) 在层 \(l\) 的实际 rank：\(r_{i,l}\)；
- 客户端聚合量 \(v_i\)，例如本地样本数；
- 数值稳定常数 \(\varepsilon\)。

归一化客户端权重：

\[
w_i=\frac{v_i}{\sum_k v_k}.
\]

## 2. 逐层计算

### 2.1 完整更新

\[
M_l=B_lA_l.
\]

完整更新能量：

\[
E=\sum_l\lVert M_l\rVert_F^2.
\]

### 2.2 原始 RBLA 前缀的截断残差

客户端 \(i\) 只能接收前 \(r_{i,l}\) 个 rank 槽位：

\[
M^{\mathrm{raw}}_{i,l}
=B_l[:,1:r_{i,l}]A_l[1:r_{i,l},:].
\]

因此原始前缀残差为：

\[
R^{\mathrm{raw}}_{i,l}
=
\left\|
B_l[:,r_{i,l}:R_l]A_l[r_{i,l}:R_l,:]
\right\|_F^2.
\]

### 2.3 最优低秩残差

计算 \(M_l\) 的奇异值：

\[
\sigma_{l,1}\ge \sigma_{l,2}\ge\cdots\ge\sigma_{l,R_l}.
\]

根据 Eckart–Young 定理，rank \(r_{i,l}\) 下的最小残差为：

\[
R^{\mathrm{opt}}_{i,l}
=\sum_{j>r_{i,l}}\sigma_{l,j}^2.
\]

该层的前缀错配机会量为：

\[
G_{i,l}
=
\max\left(
R^{\mathrm{raw}}_{i,l}-R^{\mathrm{opt}}_{i,l},
0
\right).
\]

`max` 主要用于消除浮点误差；理论上原始前缀不可能优于最优低秩近似。

## 3. 聚合为 SPMI

先对层求和，再按客户端权重聚合：

\[
R_{\mathrm{raw}}
=\sum_iw_i\sum_lR^{\mathrm{raw}}_{i,l},
\]

\[
R_{\mathrm{opt}}
=\sum_iw_i\sum_lR^{\mathrm{opt}}_{i,l},
\]

\[
G=\max(R_{\mathrm{raw}}-R_{\mathrm{opt}},0).
\]

主指标：

\[
\boxed{
\mathrm{SPMI}_{\mathrm{abs}}
=\frac{G}{\max(E,\varepsilon)}
}
\]

可选相对指标：

\[
\mathrm{SPMI}_{\mathrm{rel}}
=\frac{G}{\max(R_{\mathrm{raw}},\varepsilon)}.
\]

## 4. 伪代码

```text
input: global factors {A_l, B_l}, client ranks {r_i,l}, volumes {v_i}

w_i = v_i / sum(v)
E = 0
R_raw = 0
R_opt = 0

for each layer l:
    sigma = singular_values(B_l @ A_l)
    E += ||B_l @ A_l||_F^2

    for each client i:
        r = min(r_i,l, R_l)
        raw = ||B_l[:, r:] @ A_l[r:, :]||_F^2
        optimal = sum(sigma[r:]^2)
        R_raw += w_i * raw
        R_opt += w_i * optimal

G = max(R_raw - R_opt, 0)
SPMI_abs = G / max(E, epsilon)
SPMI_rel = G / max(R_raw, epsilon)

return SPMI_abs, SPMI_rel
```

## 5. 输出解释

- \(\mathrm{SPMI}_{\mathrm{abs}}\approx0\)：原始 RBLA 前缀接近最优低秩前缀；
- \(\mathrm{SPMI}_{\mathrm{abs}}\) 越大：原始槽位顺序造成的谱前缀损失越明显；
- SPMI 只检测当前第一轮聚合结果中的几何错配，不保证 RBLA+ 的长期精度
  一定优于 RBLA；
- 不建议直接使用未经校准的通用绝对阈值。若需要二元判断，应在相同模型、
  rank 预算和训练协议下校准阈值。

## 6. 复杂度

不必显式构造稠密矩阵 \(B_lA_l\)。可以通过 QR 分解和
\(R_l\times R_l\) 核心矩阵 SVD 获得奇异值。

单层主要复杂度约为：

\[
O\left(
(d_{\mathrm{in}}+d_{\mathrm{out}})R_l^2+R_l^3
\right).
\]

相同 rank 的客户端可以复用残差。设该层只有 \(K_l\) 种不同 rank，
残差计算只需执行 \(K_l\) 次，而不是客户端数量次。

额外空间复杂度约为：

\[
O\left(
(d_{\mathrm{in}}+d_{\mathrm{out}})R_l+R_l^2
\right).
\]
