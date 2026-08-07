# RBLA+ Experiment Results Archive

本目录从 `H:\flex-src\src\test\experiment_results` 整理生成。源文件保持不变，归档目录中保存其副本。

## 目录结构

- `01_dirichlet_main/`：主要视觉 Dirichlet 对比实验，按 `alpha / dataset / rank` 分类。
- `02_double_imbalance/`：double-imbalance 实验，按 `dataset / rank` 分类。
- `03_cost_matched/`：Dirichlet α=0.4、rank mismatch、成本匹配的补充实验。
- `04_legacy_momentum09/`：较早的 MNIST momentum=0.9 实验。
- `05_reports/`：已有视觉实验报告。
- `06_nlp_dirichlet_rank_correct/`：TinyBERT 文本实验，包含结果、配置、依赖、日志和汇总。
- `07_figures/`：论文补充图、绘图 Notebook 与相关绘图数据。
- `manifest.csv`：正式归档 CSV 的分类、来源、相对路径与 SHA-256。

## 文件数量

- 原视觉归档：208 个 CSV。
- NLP 正式结果：28 个 CSV。
- NLP 失败/已替代记录：2 个 CSV。
- 当前合计：238 个 CSV。

## NLP 覆盖情况

- GLUE：SST-2、QQP、RTE、MRPC、MNLI。
- 其他文本分类：IMDb、AG News。
- 所有正式 NLP 组合均覆盖 RBLA、RBLA+、SP、ZeroPadding，使用 Dirichlet α=0.4 与 rank-correct。
- NLP 详细结果见 `06_nlp_dirichlet_rank_correct/README.md` 和 `summary.csv`。


## 图表归档

- 共 17 个源文件：11 PDF、2 PNG、3 Notebook、1 CSV。
- 所有文件保留原目录层级并通过 SHA-256 校验。
