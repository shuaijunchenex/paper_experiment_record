# Spectrum robustness heatmap

## Purpose

This synthetic experiment tests whether gauge-induced RBLA prefix error depends on one particular singular spectrum. It varies only the spectral-decay parameter while keeping the singular bases, matrix dimensions, ranks, gauge families, and trial seeds fixed.

The experiment concerns broadcast representation only. It does not test training accuracy, repair aggregation-stage semantic mismatch, or claim that SP+ eliminates LoRA gauge freedom.

## Fixed update and spectrum

For each decay parameter \(\alpha\),

\[
\sigma_i(\alpha)=\exp\left(-\alpha\frac{i}{R-1}\right),
\qquad i=0,\ldots,R-1,
\]

and \(\sigma(\alpha)\) is normalized to unit Euclidean norm. With fixed seeded orthonormal \(U,V\),

\[
B_g=U\Sigma^{1/2},\qquad
A_g=\Sigma^{1/2}V^\top,\qquad
\Delta W_g=B_gA_g.
\]

The configuration uses \(d_{\rm out}=128\), \(d_{\rm in}=96\), \(R=16\), ranks \(r=1,\ldots,16\), and

\[
\alpha\in\{0.5,1,2,3,4,6,8\}.
\]

The row \(\alpha=4\) matches the main gauge-sensitivity experiment.

## Gauge construction

For every \(\alpha\), the same 100 seeded gauges per family are reused:

- permutation: a random permutation matrix;
- orthogonal: a seeded Gaussian matrix followed by QR with deterministic sign normalization.

For either invertible gauge \(G\),

\[
\widetilde B_g=B_gG,\qquad
\widetilde A_g=G^{-1}A_g.
\]

The implementation evaluates numpy.linalg.solve(G, A_g), never an explicit inverse.

## Broadcast calculations

RBLA uses

\[
\widehat{\Delta W}_{r,j}^{\rm RBLA}
=\widetilde B_g^{(j)}[:,:r]\widetilde A_g^{(j)}[:r,:].
\]

Compact SP+ performs two thin QR factorizations and an SVD only on

\[
C=R_BR_A^\top\in\mathbb{R}^{R\times R}.
\]

The relative error and excess error are

\[
E_{r,j}
=\frac{\|\Delta W_g-\widehat{\Delta W}_{r,j}\|_F}
{\|\Delta W_g\|_F+\epsilon},
\qquad
\Delta E_{r,j}=E_{r,j}^{\rm RBLA}-E_r^*.
\]

A dense SVD obtains \(E_r^*\) once per \(\alpha\), only in the offline evaluator. Each heatmap cell is the median excess error across 100 trials. The CSV retains every trial, so mean, standard deviation, quartiles, minimum, and maximum remain recoverable.

## Figure

The PDF contains two heatmaps with a shared color scale:

- panel (a): permutation gauges;
- panel (b): orthogonal gauges.

The horizontal axis is normalized rank \(r/R\), the vertical axis is \(\alpha\), and color is median excess error. A white dashed rectangle marks \(\alpha=4\). SP+ has no redundant heatmap because the notebook checks its excess error against numerical zero for every cell.

## Numerical checks

The notebook verifies full-update invariance, SP+ equivalence to the dense optimum, absence of dense-shaped intermediates in compact SP+, full-rank zero excess, valid condition numbers, and finite values, all with a float64 tolerance of \(10^{-10}\). No monotonicity in \(\alpha\), rank, or excess error is assumed.

## Outputs

- plot.ipynb: complete reproducible experiment;
- spectrum_robustness_heatmap.csv: trial-level results;
- spectrum_robustness_heatmap.pdf: vector figure.
