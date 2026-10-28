# FedSS 当前主要问题说明

## 1. 目前最核心的问题

目前 FedSS 最大的问题不是理论推导，而是：

> **FedSS 对最终模型性能的影响很小。**

也就是说，FedSS 虽然从理论上是在同时减少 **聚合误差（aggregation error）** 和 **广播误差（broadcast error）**，但是在实际实验中，这种改进没有明显转化成更高的准确率、更低的 loss，或者更快的收敛速度。

很多情况下，FedSS 和现有方法的最终结果非常接近。

例如，可以简单理解成：

- Baseline：85.0%
- FedSS：85.3%

或者：

- Baseline：90.1%
- FedSS：90.0%

这种差距通常只有零点几个百分点。

因此，目前不能说：

> FedSS 显著提高了 Federated LoRA 的性能。

更准确的说法应该是：

> FedSS 对性能有一定影响，但这种影响目前比较小，而且并不总是稳定地优于 baseline。

---

## 2. 为什么会出现这个问题？

FedSS 的出发点是：

在 heterogeneous-rank Federated LoRA 中，服务器每轮都要做两件事情：

1. 把不同 client 的 LoRA 模型合并成一个 global model；
2. 再把 global model 压缩成不同 rank 的 LoRA 模型广播给各个 client。

这两个步骤都会产生信息损失。

因此论文定义了：

### 聚合误差

\[
\epsilon_{\mathrm{agg}}
\]

它描述：

> client 学到的模型，与服务器最终得到的 global model 之间有多大差别。

### 广播误差

\[
\epsilon_{\mathrm{br}}
\]

它描述：

> global model 与 client 最后真正收到的低秩模型之间有多大差别。

FedSS 的想法就是：

> 同时让这两个误差尽可能小。

因此定义了一个加权目标：

\[
\lambda \epsilon_{\mathrm{agg}}^2
+
(1-\lambda)\epsilon_{\mathrm{br}}^2.
\]

理论上，这个思路是合理的。

但是问题在于：

> **聚合误差和广播误差只是影响 Federated LoRA 训练的因素之一，而不是全部。**

最终的 accuracy 或 loss 还会受到很多其他因素影响，例如：

- 不同 client 的数据分布；
- Non-IID 程度；
- local epoch；
- learning rate；
- LoRA rank；
- optimizer；
- 模型本身的能力；
- 随机 client selection；
- 随机初始化和训练噪声。

所以，即使 FedSS 把 aggregation error 和 broadcast error 降低了一些，也不一定会导致最终 accuracy 明显提高。

---

## 3. 一个很简单的理解方式

可以把整个训练过程想成开车。

Federated LoRA 最终的性能由很多因素共同决定。

例如：

- 数据异质性相当于路况；
- learning rate 相当于车速；
- local epoch 相当于每次开多远；
- rank 相当于车辆本身的能力；
- aggregation 和 broadcast error 相当于方向盘的一点误差。

FedSS 做的事情主要是：

> 让方向盘更准一点。

但是如果原本方向盘的误差就不是很大，那么把它继续调得更准，对最终到达目的地的时间影响可能非常小。

所以现在的实验结果可能说明：

> aggregation 和 broadcast 确实会产生误差，但是在目前这些实验设置中，它们并不是决定最终性能的主要因素。

---

## 4. 这对论文意味着什么？

这意味着论文不能继续把 FedSS 写成：

> 一个显著提高性能的新 Federated LoRA 算法。

因为实验结果目前并不支持这个说法。

如果最终提升只有零点几个百分点，那么使用下面这些表达会比较危险：

- significantly outperforms
- achieves superior performance
- substantially improves convergence
- consistently improves accuracy
- significantly accelerates training

这些说法都容易被 reviewer 直接质疑。

更加合适的定位应该是：

> **FedSS 是理论框架推导出来的一个具体算法实例。**

也就是说，论文的重点应该从：

> “FedSS 比其他方法更强。”

改成：

> “我们提出了一套统一分析 heterogeneous-rank Federated LoRA aggregation 和 broadcast 的理论框架，并由这个框架自然得到 FedSS。”

---

## 5. 论文真正应该强调什么？

如果 FedSS 的性能提升一直很小，那么论文真正应该强调下面三件事情。

### 第一：统一描述 aggregation 和 broadcast

不同 Federated LoRA 方法表面上看起来不一样。

但是它们基本都可以写成：

\[
\text{Local LoRA models}
\rightarrow
\text{Aggregation}
\rightarrow
\text{Global model}
\rightarrow
\text{Broadcast}
\rightarrow
\text{Local LoRA models}.
\]

因此，可以用统一的 Aggregation Operator 和 Broadcast Operator 来描述不同方法。

这个贡献和 FedSS 最终 accuracy 高不高没有直接关系。

### 第二：定义 aggregation error 和 broadcast error

论文提出两个量：

\[
\epsilon_{\mathrm{agg}},
\qquad
\epsilon_{\mathrm{br}}.
\]

这两个量分别描述：

- aggregation 过程中损失了多少信息；
- broadcast 过程中损失了多少信息。

因此，不同算法可以不再只通过 accuracy 比较，也可以通过：

\[
(\epsilon_{\mathrm{agg}},\epsilon_{\mathrm{br}})
\]

来比较它们在 communication 过程中的行为。

### 第三：理论说明这两个误差会进入收敛上界

论文的主要定理表明：

> aggregation error 和 broadcast error 会出现在模型的 convergence bound 中。

这说明这两个误差不是随便定义的指标。

它们确实和模型的优化过程有关。

但是这里需要非常注意：

> **“进入 convergence bound” 不等于 “减小一点 error 就一定能提高很多 accuracy”。**

理论只能说明它们是影响训练的因素之一。

它不能保证：

\[
\epsilon_{\mathrm{agg}}\downarrow
\quad \Rightarrow \quad
\text{accuracy 大幅上升}.
\]

也不能保证：

\[
\epsilon_{\mathrm{br}}\downarrow
\quad \Rightarrow \quad
\text{最终性能一定更好}.
\]

这可能正是目前实验提升很小的原因。

---

## 6. FedSS 在论文中的新定位

FedSS 最适合被定位成：

> **Theory-guided instantiation**

也就是：

> 根据理论得到的一个具体算法。

逻辑是：

### Step 1

理论告诉我们：

\[
\epsilon_{\mathrm{agg}}
\]

和

\[
\epsilon_{\mathrm{br}}
\]

都会影响优化过程。

### Step 2

于是我们自然提出：

\[
\min
\lambda \epsilon_{\mathrm{agg}}^2
+
(1-\lambda)\epsilon_{\mathrm{br}}^2.
\]

### Step 3

这个优化问题可以求出一个解析解。

### Step 4

这个解析解就是 FedSS。

因此，FedSS 的意义可以改成：

> 它证明了前面的理论框架不仅可以分析已有方法，也可以直接指导一个新的 aggregation-broadcast rule 的设计。

这种定位并不要求 FedSS 必须比 baseline 高很多。

---

## 7. 如果 FedSS 只提升 0.2%–0.5%，还能不能写？

可以。

但是论文必须改故事。

不能写成：

> FedSS achieves significant performance improvement.

可以写成：

> FedSS achieves competitive performance while explicitly optimizing the aggregation-broadcast distortion identified by our theoretical analysis.

简单来说：

> FedSS 的实验作用不是证明“我们比所有人都强”，而是证明“这个理论可以实际构造出一个可运行的方法”。

---

## 8. 如果 FedSS 有时候还不如 baseline 怎么办？

这也不是完全不能写。

如果结果是：

- 有些数据集更好；
- 有些差不多；
- 有些稍微差一点；

那么可以比较诚实地说：

> Optimizing aggregation and broadcast errors alone does not always translate into better predictive accuracy.

这反而可以得到一个比较有意义的观察：

> Federated LoRA 的性能不是由 aggregation 和 broadcast error 单独决定的。

最终性能还受到：

\[
\text{data heterogeneity}
+
\text{local training}
+
\text{rank}
+
\text{optimization}
+
\text{aggregation/broadcast}
\]

共同影响。

所以 FedSS 效果很普通并不一定说明理论完全没有意义。

它更可能说明：

> **aggregation 和 broadcast distortion 是一个真实存在的因素，但不是唯一的主导因素。**

---

## 9. 接下来实验最应该验证什么？

如果 FedSS 的 accuracy 提升已经确定很小，那么继续疯狂调参数试图让 accuracy 多涨几个百分点，意义可能不大。

更重要的是验证：

\[
\boxed{
\text{aggregation/broadcast error}
\longleftrightarrow
\text{training behavior}
}
\]

例如：

### 实验 A：直接比较两种误差

对不同方法计算：

\[
\epsilon_{\mathrm{agg}}
\]

和

\[
\epsilon_{\mathrm{br}}.
\]

比较：

- RBLA
- Zero-Padding
- FlexLoRA
- FedSS

观察它们的误差到底有什么差别。

### 实验 B：看误差和 loss 的关系

记录每个实验设置中的：

\[
\epsilon_{\mathrm{agg}},
\qquad
\epsilon_{\mathrm{br}},
\qquad
\text{training loss}.
\]

看 error 更小的时候，loss 是否通常也更小。

### 实验 C：看误差和 accuracy 的关系

可以计算：

\[
\text{correlation}
\left(
\epsilon_{\mathrm{agg}},
\text{accuracy}
\right)
\]

以及：

\[
\text{correlation}
\left(
\epsilon_{\mathrm{br}},
\text{accuracy}
\right).
\]

如果两者关系比较弱，这也可以直接说明：

> aggregation/broadcast error 会影响 optimization，但并不能单独决定最终 generalization performance。

---

## 10. 当前这篇论文最合理的故事

目前最适合的故事不是：

> 我们发现现有方法不好，因此提出 FedSS，FedSS 大幅提高性能。

而是：

> heterogeneous-rank Federated LoRA 中，服务器必须不断进行 aggregation 和 rank-constrained broadcast。

然后：

> 这两个过程都会产生信息损失。

然后：

> 我们定义 aggregation error 和 broadcast error 来描述这种信息损失。

然后：

> 我们建立 convergence analysis，说明这两个误差会进入优化上界。

然后：

> 根据这个理论，我们进一步构造一个直接优化这两个误差的算法 FedSS。

最后：

> 实验表明，FedSS 对最终 accuracy 的改善通常比较有限，这说明 aggregation/broadcast distortion 是影响训练的一个因素，但并不是决定最终性能的唯一因素。

---

## 11. 一句话总结目前的问题

目前 FedSS 最大的问题可以简单概括为：

> **理论上，我们找到了一个可以优化的目标；但是实验上，优化这个目标对最终模型性能的影响很小。**

这并不一定意味着整篇工作无效。

但它意味着：

> **论文应该从“性能提升型算法论文”转向“理论分析 + 机制解释 + theory-guided algorithm”的定位。**

---

## 12. 最简短版本

如果只用几句话向别人解释现在的问题，可以这样说：

> 我们的理论把 Federated LoRA 的 communication 过程分成 aggregation 和 broadcast 两步，并定义了对应的 aggregation error 和 broadcast error。理论上，这两个误差会影响模型的优化过程，所以我们设计了 FedSS 来同时减小它们。
>
> 但目前实验发现，FedSS 虽然改变了这两个误差，对最终 accuracy 和 loss 的改善却非常有限。也就是说，aggregation 和 broadcast error 可能确实是影响训练的因素，但不是决定最终性能的主要因素。
>
> 因此，这篇论文不能再把“FedSS 显著提高性能”作为核心贡献，而应该把重点放在统一理论框架、误差定义、收敛分析，以及 FedSS 作为这个理论框架的一个具体实例上。
