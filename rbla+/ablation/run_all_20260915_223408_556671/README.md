# RBLA+ ablation 原始实验结果归档

运行批次：`run_all_20260915_223408_556671`。

本目录按原始字节复制已有正式实验结果，保留原文件名、数值、配置头、元数据和远端路径；没有重新训练或重新计算结果。`archive_manifest.json` 记录每个源文件、归档相对路径、大小和 SHA-256，复制后已逐一校验。源文件与其他历史实验均保留。

- `spectum_ablation/`：CIFAR10、MNLI、QQP × RBLA/RBLA+ 两条训练路径，共 18 份 CSV。每条路径包含完整服务器模型训练日志，以及 RBLA、RBLA+ 两种广播的逐轮准确率/损失记录。广播评估是 10 个客户端模型的平均值，与完整服务器模型指标不同。当前 RBLA+ 含双向 RankScale。
- `server_memory/`：144 次方法/阶段/输入组合的原始 JSON、`measurements.csv`、运行状态、环境与输入元数据，以及原有 PDF。该批次测量 CPU FP32、单线程，预热 5 次、重复计时 20 次；内存为 CPU RSS，不能解释成 GPU 显存。
- `checkpoint_reconstruction/`：18 个真实训练 checkpoint 的离线重构结果，包括逐层明细、满宽度误差、逐输入 JSON、汇总、元数据、状态和原有 PDF；选取 round 9、49、100，FP64。
- `status.json`：原始统一入口状态和执行命令。

源目录：

- `H:\flex-src\src\test\rbla+\ablation\spectum_ablation\results`
- `H:\flex-src\src\test\rbla+\ablation\results\run_all_20260915_223408_556671`

## 原始 checkpoint 文件尚未同步

此归档包含本机已有的完整实验结果文件，但不包含原始 `.pt` 输入因子 checkpoint：原运行记录列出的 24 个文件在本机对应目录中不可用。离线测量和重构已经完成，结果已归档；若需重新执行这些离线实验，仍需从 RTX 5080 实验机同步 `.pt` 文件。完整缺失路径保留在 `archive_manifest.json` 的 `unavailable_checkpoints` 中，未用其他数据替代。