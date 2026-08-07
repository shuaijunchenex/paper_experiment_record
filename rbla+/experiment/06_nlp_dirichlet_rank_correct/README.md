# NLP Dirichlet 0.4 / Rank-Correct Experiments

本目录归档当前 TinyBERT 文本实验。所有正式结果均为 Dirichlet α=0.4、10 clients、rank-correct、100 rounds，并比较 RBLA、RBLA+、SP、ZeroPadding。

## 覆盖范围

- GLUE：SST-2、QQP、RTE、MRPC、MNLI，共 20 个完整结果。
- 其他文本分类：IMDb、AG News，共 8 个完整结果。
- 正式结果：28 个；每个包含 round 0–100，共 101 条评估记录。
- 失败记录：2 个，分别为 QQP/RTE 的早期 RBLA+ 无指标文件；均已有完整重跑结果，禁止用于统计。

## 目录结构

- `results/glue/`：五个 GLUE 数据集的正式结果。
- `results/other_nlp/`：IMDb、AG News 的正式结果。
- `failed_attempts/`：被完整重跑替代的失败文件。
- `configs/`：28 个顶层实验 YAML。
- `supporting_configs/yamls/`：实验 YAML 引用的依赖配置，保持仓库相对目录。
- `metadata/batch_logs/`：相关批运行日志。
- `summary.csv`：逐实验最终值与峰值汇总。
- `manifest.csv`：本目录所有复制文件的来源、SHA-256 和复制校验状态。

## 指标说明

- SST-2、QQP、RTE、MNLI 使用日志中的 accuracy 作为 primary metric。
- MRPC 使用日志中的 F1 作为 primary metric。
- IMDb、AG News 的旧版日志没有 primary 字段，本归档以 accuracy 作为 primary metric。
- `best` 表示 0–100 轮内的峰值；论文若报告最终模型，应使用 round 100 的 `final` 值。

## 完整结果表

| Dataset | Method | Final Acc | Final F1 | Primary | Best Primary | Final Loss |
|---|---|---:|---:|---|---:|---:|
| SST2 | rbla | 69.15% | 68.62% | accuracy | 78.44% (R1) | 2.329187 |
| SST2 | rbla_plus | 74.31% | 73.83% | accuracy | 79.36% (R2) | 1.095963 |
| SST2 | sp | 70.30% | 69.38% | accuracy | 79.70% (R7) | 0.932102 |
| SST2 | zeropadding | 71.67% | 70.49% | accuracy | 80.62% (R2) | 0.849573 |
| QQP | rbla | 73.45% | 66.97% | f1_score | 76.28% (R1) | 0.933753 |
| QQP | rbla_plus | 79.39% | 77.97% | f1_score | 80.50% (R36) | 0.425726 |
| QQP | sp | 81.07% | 79.88% | f1_score | 80.05% (R71) | 0.406382 |
| QQP | zeropadding | 76.88% | 76.61% | f1_score | 77.57% (R96) | 0.476876 |
| RTE | rbla | 48.01% | 45.88% | accuracy | 55.96% (R13) | 7.316556 |
| RTE | rbla_plus | 48.74% | 48.09% | accuracy | 51.99% (R20) | 0.903983 |
| RTE | sp | 54.51% | 54.38% | accuracy | 57.76% (R94) | 0.961160 |
| RTE | zeropadding | 54.15% | 53.91% | accuracy | 54.15% (R100) | 1.463852 |
| MRPC | rbla | 66.84% | 42.30% | f1_score | 44.12% (R10) | 9.274737 |
| MRPC | rbla_plus | 68.06% | 46.03% | f1_score | 47.08% (R78) | 1.743520 |
| MRPC | sp | 68.35% | 48.00% | f1_score | 48.19% (R97) | 1.219267 |
| MRPC | zeropadding | 68.17% | 48.51% | f1_score | 49.39% (R85) | 1.551485 |
| MNLI | rbla | 44.47% | 40.30% | accuracy | 50.25% (R3) | 2.093185 |
| MNLI | rbla_plus | 62.28% | 61.51% | accuracy | 64.51% (R37) | 0.876560 |
| MNLI | sp | 61.31% | 59.62% | accuracy | 61.67% (R65) | 0.843964 |
| MNLI | zeropadding | 62.10% | 60.69% | accuracy | 62.34% (R79) | 0.834748 |
| IMDB | rbla | 76.51% | 75.93% | accuracy | 84.15% (R3) | 4.811537 |
| IMDB | rbla_plus | 80.70% | 80.63% | accuracy | 82.25% (R19) | 2.531010 |
| IMDB | sp | 81.16% | 81.16% | accuracy | 81.89% (R42) | 1.065231 |
| IMDB | zeropadding | 80.57% | 80.57% | accuracy | 81.22% (R28) | 1.271984 |
| AGNEWS | rbla | 83.00% | 83.13% | accuracy | 90.68% (R4) | 1.057458 |
| AGNEWS | rbla_plus | 88.72% | 88.64% | accuracy | 90.99% (R15) | 0.463815 |
| AGNEWS | sp | 89.67% | 89.66% | accuracy | 90.75% (R30) | 0.335275 |
| AGNEWS | zeropadding | 88.78% | 88.72% | accuracy | 89.50% (R28) | 0.360516 |

## 当前主要观察

- RBLA+ 在 MNLI 的最终 accuracy 最好，并在 MNLI、IMDb、AG News 获得三种稳定方法中的最高峰值 accuracy。
- SP 在 MRPC、IMDb、AG News 的 round-100 accuracy 最好，整体后期稳定性较强。
- ZeroPadding 在 MRPC 的最终和峰值 F1 最好。
- 原始 RBLA 在 IMDb、AG News、MNLI 中出现明显的早期峰值后退化，不能只按峰值判断。

归档生成日期：2026-08-07；源代码提交：a278b20。
