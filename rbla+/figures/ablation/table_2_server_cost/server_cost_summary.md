# Table 2. Measured server cost

| Dataset | Method | Broadcast time (ms) | Broadcast peak incremental RSS (MiB) | Full-step time (ms) | Full-step peak incremental RSS (MiB) |
|---|---|---:|---:|---:|---:|
| CIFAR10 | RBLA | 2.03 [0.01] | 2.24 [<0.01] | 13.28 [0.43] | 28.88 [0.21] |
| CIFAR10 | RBLA+ | 26.41 [0.31] | 15.18 [0.10] | 39.23 [0.72] | 48.99 [0.11] |
| CIFAR10 | Dense SVD | 20.80 [0.33] | 13.23 [0.08] | 34.21 [1.17] | 47.00 [0.06] |
| MNLI | RBLA | 9.72 [0.47] | 6.84 [2.53] | 44.73 [0.36] | 311.03 [11.25] |
| MNLI | RBLA+ | 20.87 [0.84] | 12.34 [0.08] | 57.01 [0.53] | 293.37 [30.29] |
| MNLI | Dense SVD | 186.48 [0.33] | 8.54 [0.07] | 212.64 [0.64] | 313.19 [22.65] |
| QQP | RBLA | 9.64 [0.19] | 5.87 [0.03] | 44.66 [0.28] | 311.04 [12.54] |
| QQP | RBLA+ | 20.44 [0.76] | 10.68 [0.10] | 56.47 [0.40] | 277.90 [56.17] |
| QQP | Dense SVD | 186.54 [0.28] | 8.09 [0.04] | 215.79 [1.04] | 318.07 [26.59] |

Entries are median [IQR], where IQR = Q3 - Q1, across eight checkpoint inputs per dataset (two training paths x logged rounds 0, 9, 49, 100). Each input's time is the median of 20 warm repetitions after five warmups. CPU, FP32, one PyTorch thread. Memory is cold-pass peak incremental CPU RSS above the resident-input baseline, sampled every 1 ms; it is not GPU memory. Positive IQRs that round to 0.00 are shown as <0.01. Server step includes aggregation and all requested broadcasts. File I/O, serialization, device transfers, model loading and evaluation are excluded. Dense SVD constructs the dense matrix, uses economy full SVD, then truncates; it is not an optimized truncated-SVD solver.
