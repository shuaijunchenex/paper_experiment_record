# FedSS-LoRA 理论主线：尽可能简单的解释

> 这份说明只解释当前稿件中**聚合、广播、收敛上界和 FedSS-LoRA 设计**这条理论主线。原稿中关于“收敛下界 / residual lower bound”的内容这里**完全不讨论**，因为后续准备删除。

---

## 1. 一句话概括整篇理论

这篇工作的核心其实可以压缩成一句话：

**在异构 Federated LoRA 中，不同客户端使用不同的 LoRA rank，因此服务器在“把各客户端模型合成一个全局模型”和“再把全局模型压回不同 rank 的客户端”时都会丢失信息；本文证明了这两种信息损失会直接进入收敛上界，因此设计聚合和广播方法时，应该同时尽量减小这两种误差。FedSS-LoRA 就是按照这个原则直接解出来的一个方法。**

整条逻辑链是：

```text
不同客户端 rank 不同
        ↓
服务器聚合时会有信息损失
        ↓
服务器广播回低 rank 客户端时还会有信息损失
        ↓
定义 aggregation error 和 broadcast error
        ↓
证明这两个误差出现在收敛上界中
        ↓
因此：减小这两个误差应该有利于训练
        ↓
直接最小化两种误差的加权和
        ↓
得到 FedSS-LoRA
```

---

## 2. 最基本的问题是什么？

普通 LoRA 把模型更新写成

\[
\Delta W = BA,
\]

其中 \(B\) 和 \(A\) 都是比较小的矩阵，rank 为 \(r\)。

这样做的好处是：不需要训练和传输整个大矩阵 \(W\)，只需要训练和传输 \(B,A\)。

在联邦学习里，问题是不同客户端的计算能力不一样，因此它们可能使用不同的 LoRA rank：

\[
r_1,r_2,\ldots,r_m.
\]

例如：

- 客户端 1 的 rank 是 2；
- 客户端 2 的 rank 是 4；
- 客户端 3 的 rank 是 8。

于是它们上传的 LoRA 参数尺寸不同。

这时服务器面临两个问题：

1. **怎么把这些不同 rank 的本地模型合成一个全局模型？**
2. **合成以后，又怎么把这个全局模型重新发送给 rank 不同的客户端？**

这就是本文理论真正研究的对象。

---

## 3. 第一步：把各种具体算法抽象成两个操作

为了不只分析某一个具体算法，稿件把所有方法统一抽象成两个操作。

### 3.1 Aggregation Operator：聚合算子

每个客户端训练结束后，都有自己的更新

\[
B_iA_i.
\]

服务器通过某种规则把它们合成为一个矩阵：

\[
\mathcal A^{(t)}.
\]

不管具体方法是求平均、zero-padding、SVD，还是别的方法，只要它的作用是“把多个客户端更新变成一个全局更新”，就可以看成一个 **Aggregation Operator**。

因此可以简单理解成：

```text
客户端 1: B1A1 ┐
客户端 2: B2A2 ├──> 聚合算子 ──> 全局更新 A
客户端 3: B3A3 ┘
```

### 3.2 Broadcast Operator：广播算子

聚合之后，服务器得到一个全局矩阵 \(\mathcal A^{(t)}\)。

但是客户端 \(i\) 只能保存 rank 不超过 \(r_i\) 的 LoRA，因此服务器不能直接把任意一个高 rank 矩阵完整交给它。

所以服务器还需要把全局矩阵重新压缩成客户端可以接受的形式：

\[
P_iQ_i,
\qquad
\operatorname{rank}(P_iQ_i)\le r_i.
\]

这就是 **Broadcast Operator**。

简单理解：

```text
                 ┌──> rank = 2 的客户端
全局矩阵 A ──> 压缩 ├──> rank = 4 的客户端
                 └──> rank = 8 的客户端
```

所以，一轮通信其实就是：

```text
本地模型
   ↓
Aggregation
   ↓
全局模型
   ↓
Broadcast
   ↓
新的本地模型
```

---

## 4. 最关键的想法：这两个过程都会丢信息

本文真正重要的定义不是“算子”本身，而是进一步给这两个过程定义了两个误差。

---

## 5. Aggregation Error：聚合误差

服务器最终得到的是一个统一的全局矩阵 \(\mathcal A\)，但每个客户端原本都有自己的更新 \(B_iA_i\)。

因此定义

\[
\varepsilon_{\mathrm{agg}}^2
\approx
\sum_i w_i\|B_iA_i-\mathcal A\|_F^2.
\]

它衡量的是：

> **服务器得到的全局模型，和各个客户端真正训练出来的模型相差多少。**

如果这个误差很大，说明聚合过程中丢掉了很多客户端的信息。

### 一个非常简单的例子

假设两个客户端分别学到了：

```text
客户端 1：100
客户端 2：0
```

服务器如果聚合成 50，那么它同时没有完全保留客户端 1 和客户端 2 的结果。

这个“聚合后和原始客户端结果之间的差距”，就是 aggregation error 所表达的东西。

---

## 6. Broadcast Error：广播误差

服务器已经有了全局矩阵 \(\mathcal A\)，但客户端 \(i\) 只能接收低 rank 的矩阵 \(P_iQ_i\)。

因此定义

\[
\varepsilon_{\mathrm{br}}^2
\approx
\sum_i w_i\|\mathcal A-P_iQ_i\|_F^2.
\]

它衡量的是：

> **服务器的全局模型在压缩成客户端能够接收的 rank 后，又丢掉了多少信息。**

如果客户端 rank 很小，这个误差往往会比较明显，因为一个复杂的高 rank 矩阵不可能被一个很低 rank 的矩阵完全表示。

---

## 7. 为什么 aggregation error 和 broadcast error 会互相冲突？

这是 FedSS 的核心动机。

假设服务器为了尽可能保留所有客户端的信息，生成了一个非常丰富、rank 很高的全局矩阵。

那么：

- aggregation error 可能很小；
- 但是低 rank 客户端接不住这么多信息；
- 所以 broadcast error 可能很大。

反过来，如果服务器为了照顾低 rank 客户端，一开始就把全局模型压得很简单：

- broadcast error 可能变小；
- 但是高 rank 客户端学到的很多信息在聚合时就直接被扔掉；
- aggregation error 可能变大。

所以这里天然存在一个 trade-off：

```text
更忠实地保留客户端上传的信息
            ↕
让低 rank 客户端能够更完整地接收全局信息
```

FedSS 实际上就是在解决这个 trade-off。

---

## 8. Core Subspace：为什么定理只讨论“核心子空间”里的收敛？

这是整篇理论里最容易被公式掩盖、但实际上非常重要的一点。

LoRA 本身就是低 rank 参数化。

这意味着客户端不是在整个参数空间里随便更新，而只能沿着某些低维方向更新。

不同客户端的数据不同、rank 也不同，所以它们能更新的方向也可能不同。

稿件中的 Assumption 5 大致要求：

> 虽然各个客户端和各轮训练的更新方向不完全一样，但它们之间至少长期存在一部分共同的、稳定的方向。

这些共同方向被称为 **core subspace（核心子空间）**。

可以把它想象成：

```text
客户端 1 能学习的方向：A B C D
客户端 2 能学习的方向：    C D E F
客户端 3 能学习的方向：  B C D

大家长期共同拥有的方向：C D
```

那么 \(C,D\) 就可以类比成 core subspace。

稿件用两个投影矩阵 \(P_A\) 和 \(P_B\) 来表示这种共同子空间。

### 这一点意味着什么？

你的主定理**不是在证明整个参数空间里的完整梯度一定趋近于 0**。

它控制的是类似

\[
\|\nabla L(W)P_A\|_F^2
+
\|P_B\nabla L(W)\|_F^2
\]

这样的量。

用简单语言说：

> **模型至少会在 LoRA 持续能够覆盖到的那些共同方向上逐渐接近稳定状态。**

这比直接说“模型一定收敛到全空间最优点”更准确。

---

## 9. 主定理到底证明了什么？

把长公式全部拿掉以后，Theorem 1 的核心结论可以写成：

\[
\text{核心子空间中的平均梯度大小}
\le
\text{正常的优化项}
+
\text{数据异质性影响}
+
\text{本地训练造成的漂移}
+
\text{aggregation error 的影响}
+
\text{broadcast error 的影响}.
\]

其中和本文算法设计最直接相关的是：

\[
\varepsilon_{\mathrm{agg}},
\qquad
\varepsilon_{\mathrm{br}}
\]

都会直接出现在这个上界中。

也就是说：

> **聚合越不准确、广播压缩损失越大，理论上能得到的收敛保证就越差。**

反过来说：

> **如果我们能够把 aggregation error 和 broadcast error 做得更小，那么这个收敛上界中的相关项也会更小。**

这就是后面设计 FedSS 的理论依据。

---

## 10. 为什么这是一个“统一理论框架”？

因为这个定理并没有规定你必须使用某一种具体聚合方法。

只要一种算法能够写成：

1. 一个 aggregation operator；
2. 一组 broadcast operators；
3. 并且它们的 aggregation error 和 broadcast error 可以被控制；

那么就可以放进同一个理论框架分析。

例如 FlexLoRA 也可以这样解释：

1. 先对客户端完整更新 \(B_iA_i\) 做加权平均；
2. 再对平均矩阵做 SVD；
3. 给 rank 为 \(r_i\) 的客户端保留前 \(r_i\) 个奇异分量。

因此，这个理论框架的价值不在于只证明 FedSS，而在于：

> **把“异构 Federated LoRA 的聚合和广播”统一变成两个可以度量的信息损失问题。**

---

## 11. 主定理的证明，外行版只需要理解这几步

原稿里的完整证明很长，但真正的证明结构其实比较清楚。

### 第一步：利用 smoothness

因为 loss 是 smooth 的，所以一次更新后的 loss 不会无缘无故剧烈变化。

数学上可以把下一轮 loss 写成：

```text
当前 loss
+ 一阶下降项
+ 一个二阶误差项
```

这是很多非凸优化收敛证明都会采用的标准起点。

### 第二步：把一次全局变化拆成三段

稿件把一次全局模型变化拆成：

```text
聚合造成的误差
+
客户端真正进行本地训练产生的更新
+
广播造成的误差
```

这样 aggregation error 和 broadcast error 就自然进入证明。

### 第三步：控制客户端训练本身的偏差

客户端会连续训练多个 local epochs，而且每个客户端的数据不同。

因此客户端更新方向不可能完全等于全局梯度。

稿件使用：

- smoothness；
- gradient boundedness；
- LoRA 参数 boundedness；
- data heterogeneity boundedness；

来控制这一部分偏差。

### 第四步：利用 core subspace

Assumption 5 保证 LoRA 的更新不是每一轮都跑向完全没有共同部分的方向。

于是可以把客户端低 rank 参数产生的下降，转换成“全局梯度在 core subspace 中的下降”。

### 第五步：把很多轮加起来

最后把每一轮的不等式累加，前后很多 loss 项会抵消，得到一个长期平均的梯度上界。

所以整段证明可以概括成：

```text
smoothness
   ↓
把模型变化拆成 aggregation + local training + broadcast
   ↓
分别给三部分做上界
   ↓
用 core subspace 把 LoRA 更新和全局梯度联系起来
   ↓
跨轮累加
   ↓
得到 Theorem 1
```

---

## 12. 从定理到 FedSS：最重要的一步

Theorem 1 给出的信息是：

> aggregation error 和 broadcast error 都会影响收敛上界。

那么最自然的问题就是：

> 能不能直接设计一个聚合/广播方法，让这两个误差尽可能小？

于是稿件定义总误差：

\[
\mathcal E
=
\lambda
\sum_i w_i\|B_iA_i-\mathcal A\|_F^2
+
(1-\lambda)
\sum_i w_i\|\mathcal A-P_iQ_i\|_F^2,
\]

并要求

\[
\operatorname{rank}(P_iQ_i)\le r_i.
\]

这里：

- 第一项 = aggregation error；
- 第二项 = broadcast error；
- \(\lambda\) = 两者之间的权重。

FedSS 的目标就是：

\[
\min_{\mathcal A,P_i,Q_i}\mathcal E.
\]

这一步非常重要，因为 FedSS 并不是凭经验拍出来的规则，而是直接从前面的理论指标构造了一个优化问题。

---

## 13. 这个优化问题为什么会得到 SVD？

先固定服务器的全局矩阵 \(\mathcal A\)。

对于客户端 \(i\)，它要解决的是：

\[
\min_{P_i,Q_i}
\|\mathcal A-P_iQ_i\|_F^2,
\qquad
\operatorname{rank}(P_iQ_i)\le r_i.
\]

这个问题的意思非常简单：

> **用一个 rank 不超过 \(r_i\) 的矩阵，尽可能准确地逼近服务器的全局矩阵。**

经典的 Eckart-Young-Mirsky 定理告诉我们：

> 最优办法就是保留 \(\mathcal A\) 最大的前 \(r_i\) 个奇异值和对应奇异向量，也就是 truncated SVD。

因此广播部分不是人为指定成 SVD，而是这个最小二乘低 rank 逼近问题的最优解。

---

## 14. Weighted Coverage Rate 是什么？

这是 FedSS 里面很有解释性的一个量。

设第 \(j\) 个奇异方向是一个信息分量。

只有满足

\[
r_i\ge j
\]

的客户端，才有能力在自己的 rank 限制下接收到第 \(j\) 个奇异分量。

所以定义

\[
c_j^w
=
\sum_{i:r_i\ge j} w_i.
\]

它的意思就是：

> **第 \(j\) 个奇异方向能够被多少“客户端权重”覆盖。**

例如所有客户端权重相同：

- 10 个客户端中有 10 个能接收第 1 个分量，则 \(c_1^w=1\)；
- 只有 6 个能接收第 5 个分量，则 \(c_5^w=0.6\)；
- 只有 1 个能接收第 10 个分量，则 \(c_{10}^w=0.1\)。

所以 \(c_j^w\) 实际上在描述：

> **这个信息分量在整个客户端群体里到底有多“普适”。**

---

## 15. FedSS 最终到底对奇异值做了什么？

先计算普通的客户端加权平均：

\[
\bar M_w=\sum_i w_iB_iA_i.
\]

然后对它做 SVD：

\[
\bar M_w
=
\sum_j s_j u_jv_j^T.
\]

FedSS 保留奇异向量 \(u_j,v_j\) 不变，但是重新调整每个奇异值：

\[
\sigma_j
=
\frac{\lambda}
{1-(1-\lambda)c_j^w}
\,s_j.
\]

因此最终服务器使用的是

\[
\mathcal A^*
=
\sum_j \sigma_j u_jv_j^T.
\]

这就是所谓的 **Spectral Shrinkage**。

---

## 16. 这个 shrinkage 公式怎么直观理解？

定义缩放系数

\[
\alpha_j
=
\frac{\lambda}
{1-(1-\lambda)c_j^w}.
\]

对于 \(0<\lambda\le 1\)，这个系数通常位于 \(\lambda\) 和 1 之间。

### 情况 1：几乎所有客户端都能接收这个方向

如果

\[
c_j^w=1,
\]

那么

\[
\alpha_j=1.
\]

也就是说：

> 这个方向大家都能接收，没有必要削弱，完整保留。

### 情况 2：很少客户端能接收这个方向

如果

\[
c_j^w\approx 0,
\]

那么

\[
\alpha_j\approx \lambda.
\]

也就是说：

> 这个方向虽然可能存在于服务器的平均模型里，但大多数客户端根本接收不了，因此 FedSS 会主动减小它的权重。

所以 FedSS 的直观行为就是：

```text
大家都能接收的信息        → 尽量保留
只有少数高-rank客户端能接收 → 适当缩小
```

这比“先无条件做完整平均，再在广播时把高阶信息直接截掉”多考虑了一步：

> **服务器在形成全局模型的时候，就已经考虑了后面客户端到底能不能接收这些信息。**

这可以看成 FedSS 与单纯“先平均、后截断”的方法之间最核心的区别之一。

---

## 17. 参数 λ 到底控制什么？

\(\lambda\) 控制你更在意哪一种信息损失。

### λ 较大

更重视：

\[
B_iA_i \rightarrow \mathcal A
\]

也就是尽量忠实地保留客户端上传的信息。

此时 spectral shrinkage 比较弱。

### λ 较小

更重视：

\[
\mathcal A \rightarrow P_iQ_i
\]

也就是更考虑“这个全局信息最后客户端能不能真正接收到”。

此时，对于覆盖率低的高阶奇异方向，shrinkage 会更明显。

因此 \(\lambda\) 的本质不是一个普通超参数，而是在控制：

```text
上传信息的保真度
        vs.
下发信息的可接收性
```

---

## 18. FedSS 和 FlexLoRA 是什么关系？

当

\[
\lambda=1
\]

时，目标函数完全不考虑 broadcast error。

此时

\[
\sigma_j=s_j.
\]

也就是说完全不做 spectral shrinkage。

服务器直接使用：

\[
\mathcal A
=
\sum_iw_iB_iA_i,
\]

然后再通过 truncated SVD 向不同 rank 的客户端广播。

这正好退化成稿件中描述的 FlexLoRA 形式。

因此可以把 FedSS 理解为：

> **在 FlexLoRA 的“先平均、再 SVD 广播”基础上，进一步让服务器的聚合结果提前考虑后续的广播损失。**

所以 FedSS 是一个更一般的形式，而 FlexLoRA 可以看成 \(\lambda=1\) 的特殊情况。

---

## 19. 整篇理论真正建立了怎样的因果关系？

最重要的是下面这条链，而不是某一个复杂公式：

### 第一层：系统现实

不同客户端资源不同

\[
\Downarrow
\]

不同客户端使用不同 LoRA rank。

### 第二层：数学问题

rank 不同导致：

\[
\text{Aggregation Loss}
+
\text{Broadcast Loss}.
\]

### 第三层：理论结果

主定理表明这两个误差会进入 convergence upper bound。

### 第四层：算法设计原则

因此应该设计：

\[
\min
\big(
\text{Aggregation Error}
+
\text{Broadcast Error}
\big).
\]

### 第五层：解析解

低 rank 广播的最优解由 truncated SVD 给出；进一步推导后得到按客户端 rank coverage 调整奇异值的 spectral shrinkage。

### 第六层：FedSS

最终算法就是：

```text
1. 对客户端更新做加权平均；
2. 对加权平均做 SVD；
3. 计算每个奇异方向的 rank coverage；
4. 根据 coverage 和 λ 调整奇异值；
5. 得到新的全局聚合矩阵；
6. 每个客户端截取自己 rank 能容纳的前 ri 个奇异分量。
```

---

## 20. 如果只给一个完全不懂联邦学习的人讲，可以这样说

可以把每个客户端想象成一个容量不同的学生。

有的学生笔记本能写 100 页，有的只能写 20 页。

每个人先根据自己的数据写了一份笔记，然后把笔记交给老师。

老师需要做两件事：

1. 把所有人的笔记合成一份总笔记；
2. 再把总笔记发回每个学生。

问题是：

- 如果老师在合并时删掉太多内容，会损失学生原本学到的信息；
- 如果老师把总笔记写得太长，容量小的学生又根本抄不下来。

所以老师不能只考虑“怎样把大家的笔记合并得最完整”，还必须考虑“合并后的东西最后大家能不能接收”。

本文先证明：

> **这两类信息损失都会影响最终学习效果的理论上界。**

然后 FedSS 做的事情就是：

> **在生成总笔记时，对大家都能接收的重要内容尽量完整保留，而对只有少数高容量学生才能接收的内容适当降低权重。**

这就是整个方法最直观的含义。

---

## 21. 这篇理论“证明了什么”和“没有证明什么”

这一部分在以后写论文或答 reviewer 时很重要。

### 理论证明了

1. 可以把异构 Federated LoRA 的通信过程统一写成 aggregation operator 和 broadcast operator。
2. 可以用 aggregation error 和 broadcast error 定量描述两次通信中的信息损失。
3. 在给定假设下，核心子空间中的梯度可以得到一个收敛上界。
4. aggregation error 和 broadcast error 明确进入这个上界。
5. 因此，减小这两个误差具有直接的理论动机。
6. 对它们的加权和进行优化，可以解析地得到 FedSS 的 spectral shrinkage 形式。
7. 对固定全局矩阵而言，truncated SVD 是满足客户端 rank 限制时的最优低 rank 广播方式。
8. \(\lambda=1\) 时 FedSS 退化为 FlexLoRA 式的“加权平均 + truncated SVD”。

### 理论没有直接证明

1. **没有证明 FedSS 在所有数据集上 accuracy 一定最高。**
2. **没有证明最小化这两个误差就等价于直接最小化 test loss。**
3. **没有证明整个参数空间中的完整梯度都一定趋于 0；主定理控制的是 core subspace 中的梯度分量。**
4. **没有证明某个固定 \(\lambda\) 对所有任务都是最优的。**

更准确的表述应该是：

> FedSS 是由收敛上界中可由算法直接控制的两个误差项所驱动的一种理论化设计。

而不是：

> FedSS 被证明在所有情况下都是最优训练算法。

---

## 22. 最后用 8 句话概括整篇理论

1. Federated LoRA 中不同客户端可能使用不同 rank。
2. 不同 rank 使得服务器无法无损地完成聚合和广播。
3. 因此本文把所有算法统一抽象成 aggregation operator 和 broadcast operator。
4. 聚合过程的信息损失定义为 aggregation error，广播过程的信息损失定义为 broadcast error。
5. 在 smoothness、bounded gradient、bounded LoRA parameters、bounded data heterogeneity 和 shared core subspace 等条件下，主定理给出 core subspace 中的收敛上界。
6. 这个上界明确包含 aggregation error 和 broadcast error，因此这两个误差是算法可以直接优化的重要量。
7. FedSS 通过最小化两种误差的加权和，得到一个根据客户端 rank coverage 对奇异值进行自适应 shrinkage 的解析解。
8. 当 \(\lambda=1\) 时不再考虑广播误差，FedSS 退化为普通加权平均后再做 truncated SVD 的 FlexLoRA 形式。

---

## 24. 最短版本

如果以后有人问“你的理论到底做了什么”，最短可以回答：

> **我们研究的是异构 rank 的 Federated LoRA。不同客户端 rank 不同，所以服务器聚合时会丢一次信息，广播回客户端时又会丢一次信息。我们把这两种损失分别定义成 aggregation error 和 broadcast error，并证明它们都会出现在模型的收敛上界里。因此我们进一步直接最小化这两种误差的加权和，最终得到 FedSS：它通过 SVD 分解全局更新，并根据每个奇异方向能够被多少客户端 rank 覆盖来自适应缩小奇异值。大家都能接收的方向基本保留，只有少数高-rank 客户端能接收的方向会被更多压缩。**
