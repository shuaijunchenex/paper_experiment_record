# RBLA+ RankScale accuracy comparison

- `rank_scale_accuracy.tex`: single-column booktabs table, with correct/mismatch sections stacked vertically and paired winners in bold. Uses `table` and `tabular*` at `columnwidth`. Requires `booktabs`.
- `rank_scale_accuracy.pdf`: visual preview of the same formatted values. Rendered with Matplotlib; no local LaTeX compiler was available, so it is not a compiled TeX preview.
- `rank_scale_acc_difference.pdf`: 3.45-inch-wide single-column vector figure, with two grouped bars per dataset and a shared axis.
- `rank_scale_acc_difference.png`: high-resolution image of the figure.
- `rank_scale_difference_figure.tex`: figure inclusion and caption. Requires `graphicx`.
- `metadata.json`: source paths, SHA-256 hashes, peak rounds and recorded configuration differences.

Both variants use complete server-model training logs, Dirichlet alpha=0.4, grouped by dataset and rank assignment. ACC is the maximum over logged rounds 0-100. Differences are calculated before rounding as with scale minus without scale (percentage points). No partial runs, broadcast metrics, or September 16 broadcast-ablation runs are included. Missing scaling results for MNIST, Fashion-MNIST, KMNIST, QMNIST and MRPC are explicitly noted rather than treated as zeros. These are not multi-seed averages. The historical comparison alone does not isolate all implementation effects. MNLI correct has slightly different recorded rank ratios; see metadata.

Reproduce with `python -B generate.py` using Python, NumPy and Matplotlib. Original CSVs are read and their peak values and rounds verified before generation. To use in a paper, copy the table snippet into the manuscript; place the figure PDF alongside the figure snippet or adjust its relative path.
