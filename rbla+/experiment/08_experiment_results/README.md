# RBLA+ experiment_results CSV archive

- Archived at: 2026-08-18T19:48:57+10:00
- Source: H:\flex-src\src\test\experiment_results
- Source CSV files: 690
- Source payload: 23040605 bytes
- Complete metric files: 688
- Incomplete metric files: 2

## Layout

- raw/: exact flat mirror of every source CSV.
- by_dataset/: second copy organized as family/dataset/method/variant/rank assignment/alpha.
- results_manifest.csv: one row per experiment, including configuration, peak/final accuracy, status, size, and SHA-256.
- summary_by_group.csv: derived group-level peak/final accuracy statistics.
- archive_inventory.csv: size and SHA-256 for every archived file except the inventory itself.
- metadata/source.txt: source and archive provenance.

## Dataset counts

- agnews: 12
- cifar10: 12
- cinic10: 12
- fmnist: 141
- imdb: 12
- kmnist: 141
- mnist: 141
- mnli: 12
- mrpc: 12
- qmnist: 141
- qqp: 13
- rte: 17
- sst2: 12
- svhn: 12

## Method counts

- flora: 172
- rbla: 129
- rbla_plus: 131
- sp: 129
- zeropadding: 129

## FLoRA variant labels

- flora/corrected: 32
- flora/unspecified: 140

## Rank-assignment counts

- correct: 254
- medium: 188
- mismatch: 248

For old robustness runs whose filenames omit the rank assignment, the label is recovered from the Spearman correlation between client sample counts and client rank ratios: +1 = correct, -1 = mismatch, and approximately 0.006061 = medium. The manifest records whether each label was explicit or inferred.

FLoRA files explicitly named flora_correct are labeled corrected. Files named only flora are labeled unspecified rather than being assumed corrected or legacy.

## Incomplete files

- qqp_dirichlet_0_4_rank_correct_rbla_plus_-train-20260803_140728-a7e29b31043a1428.csv
- rte_dirichlet_0_4_rank_correct_rbla_plus_-train-20260803_223404-ffc1ff95ff0656d6.csv
