# RBLA+ Experiment Results Archive

本目录从 `H:\flex-src\src\test\experiment_results` 整理生成。源文件保持不变，归档目录中保存其副本。

## 目录结构

- `01_dirichlet_main/`：主要 Dirichlet 对比实验，按 `alpha / dataset / rank` 分类。
- `02_double_imbalance/`：double-imbalance 实验，按 `dataset / rank` 分类。
- `03_cost_matched/`：Dirichlet α=0.4、rank mismatch、成本匹配的补充实验。
- `04_legacy_momentum09/`：较早的 MNIST momentum=0.9 实验，按数据分布与 rank 分类。
- `05_reports/`：已有实验报告。
- `manifest.csv`：所有归档文件的分类信息、原始文件名及相对路径。

## 文件数量

- Dirichlet 主实验：120 个 CSV。
- Double-imbalance：48 个 CSV。
- Cost-matched：16 个 CSV。
- Legacy momentum/extreme：24 个 CSV。
- 报告：1 个 Markdown 文件。
- 合计：208 个 CSV + 1 个报告。

## 主实验覆盖情况

- MNIST、FMNIST、KMNIST：α=0.1、0.4、0.8；3 种 rank；4 种方法，分别为 36 个结果。
- QMNIST：目前仅有 α=0.4；3 种 rank；4 种方法，共 12 个结果。
- QMNIST 的 α=0.1 和 α=0.8 尚未出现在源结果目录中，共缺少 24 个主实验组合。

每个主实验 CSV 包含 round 0 至 round 100，共 101 条评估记录。
