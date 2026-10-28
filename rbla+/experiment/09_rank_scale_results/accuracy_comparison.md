# RBLA+ rank scaling 最高 ACC 对比

口径：完整训练日志中服务器模型 `accuracy` 的峰值，取已评估轮次，单位为百分比。差值为带 scaling 减去不带 scaling，单位为百分点。round 沿用日志的 0-based 编号。

两侧均限定 Dirichlet α=0.4，并按数据集、rank assignment 分组。旧版来源为 `08_experiment_results` 中配置为 `sp_plus` 的完整记录；新版仅使用本次从指定源目录归档的 `rbla_plus_rank_scale` 完整记录。已有 9 月 16 日广播消融批次不合并到本表。

同组有多个完整记录时分别取最高峰值，因此不是多 seed 均值，也不是严格控制所有实现差异的因果消融。配置字段差异保存在 `accuracy_comparison.json`。

| 数据集 | rank assignment | 不带 rank scaling：最高 ACC（round） | 带 rank scaling：最高 ACC（round） | 差值（百分点） |
|---|---|---:|---:|---:|
| agnews | correct | [90.9868%](<../08_experiment_results/raw/agnews_dirichlet_0_4_rank_correct_rbla_plus_-train-20260805_113023-880bccddb94fd2be.csv>)（r15） | [91.3289%](<by_dataset/agnews/correct/alpha_0.4/agnews_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260903_001433-cac77897798a03ce.csv>)（r10） | +0.3421 |
| agnews | mismatch | [90.5000%](<../08_experiment_results/raw/agnews_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260813_011518-9bb2eb6f839edfa9.csv>)（r28） | [91.6842%](<by_dataset/agnews/mismatch/alpha_0.4/agnews_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260904_115754-ec642dc2dc42c5ad.csv>)（r18） | +1.1842 |
| cifar10 | correct | [78.3000%](<../08_experiment_results/raw/cifar10_dirichlet_0_4_rank_correct_rbla_plus_-train-20260807_084017-3abd855adee994bc.csv>)（r71） | [79.1200%](<by_dataset/cifar10/correct/alpha_0.4/cifar10_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260830_114801-17922021afed9cdb.csv>)（r94） | +0.8200 |
| cifar10 | mismatch | [79.2100%](<../08_experiment_results/raw/cifar10_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260810_115818-1caefa68815d261f.csv>)（r67） | [78.9500%](<by_dataset/cifar10/mismatch/alpha_0.4/cifar10_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260830_154124-3dc18c46d1c4f90e.csv>)（r59） | -0.2600 |
| cinic10 | correct | [69.1256%](<../08_experiment_results/raw/cinic10_dirichlet_0_4_rank_correct_rbla_plus_-train-20260807_132750-2cbdf6d449b43542.csv>)（r67） | [69.6500%](<by_dataset/cinic10/correct/alpha_0.4/cinic10_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260830_193615-8a64b050e747b3cb.csv>)（r69） | +0.5244 |
| cinic10 | mismatch | [69.2544%](<../08_experiment_results/raw/cinic10_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260810_164327-b84e284057947491.csv>)（r91） | [68.8933%](<by_dataset/cinic10/mismatch/alpha_0.4/cinic10_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260831_084024-e5867a139c3fbb79.csv>)（r99） | -0.3611 |
| fmnist | correct | [67.7300%](<../08_experiment_results/raw/rbla_plus_-train-20260804_051742-ca4aed633eb4b28b.csv>)（r99） | 缺少结果 | — |
| fmnist | medium | [66.3400%](<../08_experiment_results/raw/rbla_plus_-train-20260807_020323-2bd5bbf81848a475.csv>)（r100） | 缺少结果 | — |
| fmnist | mismatch | [64.9700%](<../08_experiment_results/raw/rbla_plus_-train-20260809_230530-90ed46e561cb3f8d.csv>)（r99） | 缺少结果 | — |
| imdb | correct | [82.2480%](<../08_experiment_results/raw/imdb_dirichlet_0_4_rank_correct_rbla_plus_-train-20260805_080142-8605ba97eeae5647.csv>)（r19） | [84.3160%](<by_dataset/imdb/correct/alpha_0.4/imdb_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260905_224859-6a64706e2055c12c.csv>)（r8） | +2.0680 |
| imdb | mismatch | [81.4120%](<../08_experiment_results/raw/imdb_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260812_214623-8ac377ab29d18c10.csv>)（r88） | [83.6400%](<by_dataset/imdb/mismatch/alpha_0.4/imdb_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260906_063020-a84a2401ad48de6c.csv>)（r13） | +2.2280 |
| kmnist | correct | [47.0300%](<../08_experiment_results/raw/rbla_plus_-train-20260804_220009-4356ea2ab574445e.csv>)（r100） | 缺少结果 | — |
| kmnist | medium | [42.1000%](<../08_experiment_results/raw/rbla_plus_-train-20260807_152207-64df0b7d985a20fd.csv>)（r100） | 缺少结果 | — |
| kmnist | mismatch | [41.2000%](<../08_experiment_results/raw/rbla_plus_-train-20260810_122109-34624a5216cef24c.csv>)（r100） | 缺少结果 | — |
| mnist | correct | [84.1000%](<../08_experiment_results/raw/rbla_plus_-train-20260803_161028-de5bfa0c692d3cd8.csv>)（r100） | 缺少结果 | — |
| mnist | medium | [83.8000%](<../08_experiment_results/raw/rbla_plus_-train-20260806_123628-d55ffe210f0572cd.csv>)（r100） | 缺少结果 | — |
| mnist | mismatch | [81.6400%](<../08_experiment_results/raw/rbla_plus_-train-20260809_094220-63cf628e503b4560.csv>)（r100） | 缺少结果 | — |
| mnli | correct | [64.5135%](<../08_experiment_results/raw/mnli_dirichlet_0_4_rank_correct_rbla_plus_-train-20260805_200341-087df6e45310afec.csv>)（r37） | [65.1554%](<by_dataset/mnli/correct/alpha_0.4/mnli_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260906_141523-e9768cd4ae91f2e0.csv>)（r24） | +0.6419 |
| mnli | mismatch | [60.0204%](<../08_experiment_results/raw/mnli_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260812_080957-8088910697a516e4.csv>)（r46） | [64.3709%](<by_dataset/mnli/mismatch/alpha_0.4/mnli_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260915_003558-3b2cc50e630c1b24.csv>)（r19） | +4.3505 |
| mrpc | correct | [68.2899%](<../08_experiment_results/raw/mrpc_dirichlet_0_4_rank_correct_rbla_plus_-train-20260805_064736-ff9188e272cecf1f.csv>)（r54） | 缺少结果 | — |
| mrpc | mismatch | [69.0435%](<../08_experiment_results/raw/mrpc_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260812_031659-aafd5f5f07dd0759.csv>)（r96） | 缺少结果 | — |
| qmnist | correct | [83.5550%](<../08_experiment_results/raw/rbla_plus_-train-20260805_175804-eef18e0b2c30e747.csv>)（r99） | 缺少结果 | — |
| qmnist | medium | [83.3167%](<../08_experiment_results/raw/rbla_plus_-train-20260808_145629-bf60b16bef90fc96.csv>)（r100） | 缺少结果 | — |
| qmnist | mismatch | [80.8300%](<../08_experiment_results/raw/rbla_plus_-train-20260811_111257-f84c23dc4efcf675.csv>)（r100） | 缺少结果 | — |
| qqp | correct | [81.6646%](<../08_experiment_results/raw/qqp_dirichlet_0_4_rank_correct_rbla_plus_-train-20260804_100118-8d6731dc8a05a445.csv>)（r34） | [82.4808%](<by_dataset/qqp/correct/alpha_0.4/qqp_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260915_025204-ed54eab36796bbea.csv>)（r35） | +0.8162 |
| qqp | mismatch | [79.8095%](<../08_experiment_results/raw/qqp_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260811_140550-48c3771295a533e3.csv>)（r75） | [81.0166%](<by_dataset/qqp/mismatch/alpha_0.4/qqp_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260915_050230-934e4918a8f22a11.csv>)（r32） | +1.2070 |
| rte | correct | [54.5126%](<../08_experiment_results/raw/rte_dirichlet_0_4_rank_correct_rbla_plus_-train-20260817_133039-f37f3add962d215a.csv>)（r47） | [57.0397%](<by_dataset/rte/correct/alpha_0.4/rte_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260915_071134-b3f407d3b3a6b16f.csv>)（r95） | +2.5271 |
| rte | mismatch | [55.9567%](<../08_experiment_results/raw/rte_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260812_023337-0a17fd1846542b93.csv>)（r97） | [55.5957%](<by_dataset/rte/mismatch/alpha_0.4/rte_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260915_071905-cac2ae5492898ba2.csv>)（r36） | -0.3610 |
| sst2 | correct | [79.3578%](<../08_experiment_results/raw/sst2_dirichlet_0_4_rank_correct_rbla_plus_-train-20260804_062330-61ffb73e61d951e2.csv>)（r2） | [80.6193%](<by_dataset/sst2/correct/alpha_0.4/sst2_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260915_072652-314a009e60e8e561.csv>)（r1） | +1.2615 |
| sst2 | mismatch | [81.0780%](<../08_experiment_results/raw/sst2_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260811_062005-8b48d26bf8d664f8.csv>)（r16） | [80.5046%](<by_dataset/sst2/mismatch/alpha_0.4/sst2_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260915_081944-a1516d783ae81cff.csv>)（r1） | -0.5734 |
| svhn | correct | [92.9010%](<../08_experiment_results/raw/svhn_dirichlet_0_4_rank_correct_rbla_plus_-train-20260807_212758-c3111fc3a4d81dff.csv>)（r90） | [92.9779%](<by_dataset/svhn/correct/alpha_0.4/svhn_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260914_230656-f3810428dff251cc.csv>)（r93） | +0.0768 |
| svhn | mismatch | [93.4504%](<../08_experiment_results/raw/svhn_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260811_004052-c1bec1b6c8f18a26.csv>)（r35） | [93.2045%](<by_dataset/svhn/mismatch/alpha_0.4/svhn_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260901_223021-5135fb3adeee0962.csv>)（r19） | -0.2459 |

## 按数据集取两种 rank assignment 中的最高值

本表两侧的最佳 rank assignment 可能不同；用于汇总已观察到的最高值，不能单独用于估计 scaling 效果。

| 数据集 | 不带最高 ACC | 带最高 ACC | 差值（百分点） |
|---|---:|---:|---:|
| agnews | [90.9868%](<../08_experiment_results/raw/agnews_dirichlet_0_4_rank_correct_rbla_plus_-train-20260805_113023-880bccddb94fd2be.csv>)（r15） | [91.6842%](<by_dataset/agnews/mismatch/alpha_0.4/agnews_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260904_115754-ec642dc2dc42c5ad.csv>)（r18） | +0.6974 |
| cifar10 | [79.2100%](<../08_experiment_results/raw/cifar10_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260810_115818-1caefa68815d261f.csv>)（r67） | [79.1200%](<by_dataset/cifar10/correct/alpha_0.4/cifar10_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260830_114801-17922021afed9cdb.csv>)（r94） | -0.0900 |
| cinic10 | [69.2544%](<../08_experiment_results/raw/cinic10_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260810_164327-b84e284057947491.csv>)（r91） | [69.6500%](<by_dataset/cinic10/correct/alpha_0.4/cinic10_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260830_193615-8a64b050e747b3cb.csv>)（r69） | +0.3956 |
| fmnist | [67.7300%](<../08_experiment_results/raw/rbla_plus_-train-20260804_051742-ca4aed633eb4b28b.csv>)（r99） | 缺少结果 | — |
| imdb | [82.2480%](<../08_experiment_results/raw/imdb_dirichlet_0_4_rank_correct_rbla_plus_-train-20260805_080142-8605ba97eeae5647.csv>)（r19） | [84.3160%](<by_dataset/imdb/correct/alpha_0.4/imdb_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260905_224859-6a64706e2055c12c.csv>)（r8） | +2.0680 |
| kmnist | [47.0300%](<../08_experiment_results/raw/rbla_plus_-train-20260804_220009-4356ea2ab574445e.csv>)（r100） | 缺少结果 | — |
| mnist | [84.1000%](<../08_experiment_results/raw/rbla_plus_-train-20260803_161028-de5bfa0c692d3cd8.csv>)（r100） | 缺少结果 | — |
| mnli | [64.5135%](<../08_experiment_results/raw/mnli_dirichlet_0_4_rank_correct_rbla_plus_-train-20260805_200341-087df6e45310afec.csv>)（r37） | [65.1554%](<by_dataset/mnli/correct/alpha_0.4/mnli_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260906_141523-e9768cd4ae91f2e0.csv>)（r24） | +0.6419 |
| mrpc | [69.0435%](<../08_experiment_results/raw/mrpc_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260812_031659-aafd5f5f07dd0759.csv>)（r96） | 缺少结果 | — |
| qmnist | [83.5550%](<../08_experiment_results/raw/rbla_plus_-train-20260805_175804-eef18e0b2c30e747.csv>)（r99） | 缺少结果 | — |
| qqp | [81.6646%](<../08_experiment_results/raw/qqp_dirichlet_0_4_rank_correct_rbla_plus_-train-20260804_100118-8d6731dc8a05a445.csv>)（r34） | [82.4808%](<by_dataset/qqp/correct/alpha_0.4/qqp_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260915_025204-ed54eab36796bbea.csv>)（r35） | +0.8162 |
| rte | [55.9567%](<../08_experiment_results/raw/rte_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260812_023337-0a17fd1846542b93.csv>)（r97） | [57.0397%](<by_dataset/rte/correct/alpha_0.4/rte_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260915_071134-b3f407d3b3a6b16f.csv>)（r95） | +1.0830 |
| sst2 | [81.0780%](<../08_experiment_results/raw/sst2_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260811_062005-8b48d26bf8d664f8.csv>)（r16） | [80.6193%](<by_dataset/sst2/correct/alpha_0.4/sst2_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260915_072652-314a009e60e8e561.csv>)（r1） | -0.4587 |
| svhn | [93.4504%](<../08_experiment_results/raw/svhn_dirichlet_0_4_rank_mismatch_rbla_plus_-train-20260811_004052-c1bec1b6c8f18a26.csv>)（r35） | [93.2045%](<by_dataset/svhn/mismatch/alpha_0.4/svhn_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260901_223021-5135fb3adeee0962.csv>)（r19） | -0.2459 |

## 不完整记录

下列原始文件一并保留，但不参与完整训练的最高 ACC 对比：

- `mnli_dirichlet_0_4_rank_mismatch_rbla_plus_rank_scale_-train-20260911_091142-3b2cc50e630c1b24.csv`：截至 round 69，共 70 个有效轮次；同 run_id 的完整文件见归档清单。
- `svhn_dirichlet_0_4_rank_correct_rbla_plus_rank_scale_-train-20260831_204747-f3810428dff251cc.csv`：截至 round 63，共 64 个有效轮次；同 run_id 的完整文件见归档清单。
