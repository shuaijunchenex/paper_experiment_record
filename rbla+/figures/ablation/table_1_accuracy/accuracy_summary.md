# Table 1. Training-path and broadcast accuracy

| Training path | Broadcast | CIFAR10 final | CIFAR10 last 10 | MNLI final | MNLI last 10 | QQP final | QQP last 10 |
|---|---|---:|---:|---:|---:|---:|---:|
| RBLA | RBLA | 16.97 | 17.84 | 56.81 | 56.93 | 75.66 | 75.99 |
| RBLA | RBLA+ | 35.49 | 38.45 | 42.60 | 42.56 | 70.13 | 69.83 |
| RBLA+ | RBLA | **72.96** | **72.84** | 58.04 | 58.14 | **76.64** | 77.07 |
| RBLA+ | RBLA+ | 72.73 | 72.76 | **60.53** | **62.59** | 75.27 | **77.25** |

Accuracy (%), unweighted mean over ten client broadcast models. Final: update 101 (logged round 100). Last 10: mean of updates 92-101 (logged rounds 91-100). One seed (42), correct rank assignment, Dirichlet alpha=0.4. Cross-broadcast evaluation does not change the training path. RBLA+ includes bidirectional RankScale. Bold marks the highest value in each dataset/metric column. Last 10 is a temporal mean, not an average over seeds. Full-width server-model accuracy is excluded.
