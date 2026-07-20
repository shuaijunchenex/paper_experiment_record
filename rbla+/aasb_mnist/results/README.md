# MNIST RBLA Activation-Aware Compact Broadcasting

All values below are mean +/- sample standard deviation over seeds 42, 43, and 44.
D_cal uses 200 fixed, class-balanced training samples excluded from all clients; D_func_eval uses 1,000 fixed, class-balanced test samples and never affects ordering.

| Method | Acc@5 | Acc@10 | Acc@20 | Best | AUC 1-20 | Extra server s/round |
|---|---:|---:|---:|---:|---:|---:|
| rbla | 0.906900 +/- 0.014095 | 0.930767 +/- 0.006711 | 0.942233 +/- 0.005273 | 0.944667 +/- 0.004650 | 0.906242 +/- 0.004278 | 0.000000 +/- 0.000000 |
| rbla_cc_svd | 0.859633 +/- 0.012851 | 0.899167 +/- 0.016587 | 0.928600 +/- 0.003318 | 0.931400 +/- 0.005110 | 0.880317 +/- 0.009282 | 0.003013 +/- 0.000046 |
| rbla_cc_activation | 0.862333 +/- 0.016327 | 0.907633 +/- 0.015645 | 0.932033 +/- 0.006068 | 0.932900 +/- 0.005565 | 0.883447 +/- 0.006194 | 0.021770 +/- 0.000469 |

## Prefix and error metrics

### Rank 1

| Method | Prefix accuracy AUC | Held-out functional error | Weight error | Reorder rate |
|---|---:|---:|---:|---:|
| rbla | 0.126603 +/- 0.020706 | 0.922026 +/- 0.001783 | 0.942095 +/- 0.011496 | nan +/- 0.000000 |
| rbla_cc_svd | 0.100702 +/- 0.018761 | 0.808028 +/- 0.024366 | 0.775146 +/- 0.013503 | 0.000000 +/- 0.000000 |
| rbla_cc_activation | 0.107237 +/- 0.013063 | 0.790597 +/- 0.026578 | 0.799074 +/- 0.009676 | 0.511111 +/- 0.083887 |

### Rank 2

| Method | Prefix accuracy AUC | Held-out functional error | Weight error | Reorder rate |
|---|---:|---:|---:|---:|
| rbla | 0.223023 +/- 0.061303 | 0.845815 +/- 0.006793 | 0.880901 +/- 0.006449 | nan +/- 0.000000 |
| rbla_cc_svd | 0.222478 +/- 0.045951 | 0.645560 +/- 0.013963 | 0.607408 +/- 0.015578 | 0.000000 +/- 0.000000 |
| rbla_cc_activation | 0.190375 +/- 0.041892 | 0.630450 +/- 0.031349 | 0.645701 +/- 0.009415 | 0.205556 +/- 0.091414 |

### Rank 4

| Method | Prefix accuracy AUC | Held-out functional error | Weight error | Reorder rate |
|---|---:|---:|---:|---:|
| rbla | 0.455047 +/- 0.054989 | 0.703553 +/- 0.020587 | 0.776980 +/- 0.024639 | nan +/- 0.000000 |
| rbla_cc_svd | 0.581062 +/- 0.024375 | 0.413004 +/- 0.017183 | 0.381889 +/- 0.013391 | 0.000000 +/- 0.000000 |
| rbla_cc_activation | 0.556820 +/- 0.048417 | 0.389803 +/- 0.028018 | 0.415008 +/- 0.011554 | 0.177778 +/- 0.035924 |

### Rank 8

| Method | Prefix accuracy AUC | Held-out functional error | Weight error | Reorder rate |
|---|---:|---:|---:|---:|
| rbla | 0.764057 +/- 0.011529 | 0.465188 +/- 0.017864 | 0.546291 +/- 0.002566 | nan +/- 0.000000 |
| rbla_cc_svd | 0.857337 +/- 0.018020 | 0.171291 +/- 0.021859 | 0.155038 +/- 0.004175 | 0.000000 +/- 0.000000 |
| rbla_cc_activation | 0.861228 +/- 0.007962 | 0.142019 +/- 0.009961 | 0.174361 +/- 0.006876 | 0.100694 +/- 0.024855 |

## Main comparison: activation-aware CC minus singular-value CC

- Accuracy AUC delta: +0.003130.
- Accuracy deltas at rounds 5/10/20: +0.002700, +0.008467, +0.003433.
- Held-out functional-error deltas at ranks 1/2/4/8: -0.017431, -0.015110, -0.023201, -0.029272.
- Weight-error deltas at ranks 1/2/4/8: +0.023929, +0.038292, +0.033119, +0.019323.
- Prefix-accuracy AUC deltas at ranks 1/2/4/8: +0.006535, -0.032103, -0.024242, +0.003892.
- Extra server time delta: +0.018757 seconds/round.

## Integrity audit

- Complete runs/round rows: 9/180.
- Activation-aware layer-round records: 180; fallbacks: 0; changed orderings: 180.
- Maximum full-rank probe relative error: 1.378e-06.
- Maximum compact-core reconstruction error: 1.260e-06.
- The nine sample-index files are identical; all scalar audit fields are finite.

## Interpretation guardrails

Activation-aware CC is supported as a functional improvement only where held-out, not merely calibration, error decreases. Accuracy is reported separately because layer-output preservation does not guarantee task-accuracy improvement. No dense LoRA update matrix is formed by the experiment or production paths.

The r labels in functional/weight diagnostics mean a uniform r-component prefix in each LoRA layer. In prefix-test accuracy, r labels the configured output-layer rank and rank ratio: r=1/2/4/8 maps to hidden-layer ranks 1/3/6/12 respectively for this model.
