# RBLA+ RankScale 结果归档

- 来源：`H:\flex-src\src\test\experiment_results`。
- 按配置字段 `effective_yaml.server_yaml.strategy.strategy_name = rbla_plus_rank_scale` 筛选，不仅凭文件名。
- 共 20 份 CSV，覆盖 9 个数据集。18 份完整训练，另有 2 份未完成记录。
- 完整覆盖：agnews, cifar10, cinic10, imdb, mnli, qqp, rte, sst2, svhn，均为 Dirichlet α=0.4，包含 rank-correct 和 rank-mismatch。
- 原始 CSV 按 `by_dataset/<dataset>/<rank_assignment>/alpha_0.4/` 保存，文件名和字节内容保持原样；源目录未修改。
- 20 份副本均经 SHA-256 校验，文件大小、来源、配置、轮数和峰值记录于 [archive_manifest.json](archive_manifest.json)。
- SVHN correct 和 MNLI mismatch 各有一份未完成记录与同 run_id 的完整记录，均保留，仅完整记录参与对比。
- [accuracy_comparison.md](accuracy_comparison.md)：与当前库旧版 RBLA+ 的 ACC 峰值比较，包含所有 14 个旧版数据集的覆盖情况。
- [accuracy_comparison.json](accuracy_comparison.json)：未舍入指标、对应源文件和已记录的配置差异。
- 此批次与 `08_experiment_results` 历史快照及 `ablation` 广播消融批次独立归档。
