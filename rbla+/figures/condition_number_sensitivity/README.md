# Condition-number sensitivity

## Purpose

This synthetic experiment tests broadcast reconstruction and numerical stability under moderately non-orthogonal, rank-coordinate-mixing gauges with prescribed condition numbers. It concerns broadcast representation only.

Simple diagonal scaling is not the main transformation because reciprocal column/row scaling leaves every indexed rank-one term unchanged.

## Fixed update

The experiment uses \(d_{\rm out}=128\), \(d_{\rm in}=96\), \(R=16\), fixed client rank \(r=8\), and

\[
\sigma_i\propto\exp\left(-4\frac{i}{R-1}\right).
\]

With fixed seeded orthonormal \(U,V\),

\[
B_g=U\Sigma^{1/2},\qquad
A_g=\Sigma^{1/2}V^\top,\qquad
\Delta W_g=B_gA_g.
\]

## Prescribed-condition gauge

For trial \(j\), two orthogonal matrices are generated once and reused for all condition levels:

\[
G_j(\kappa)=Q_{L,j}D(\kappa)Q_{R,j}^\top,
\]

\[
D(\kappa)=\operatorname{diag}\left[
\exp\left(\operatorname{linspace}
\left(-\tfrac12\log\kappa,\tfrac12\log\kappa,R\right)\right)
\right].
\]

Thus \(\operatorname{cond}_2(G_j)=\kappa\) up to floating-point precision, with singular values balanced around one. The levels are

\[
\kappa\in\{1,1.5,2,3,5,7.5,10\}.
\]

The paired design fixes \(Q_{L,j},Q_{R,j}\) across all \(\kappa\) for each of 100 trials. The factors are transformed using

\[
\widetilde B_g=B_gG,\qquad
\widetilde A_g=\operatorname{solve}(G,A_g),
\]

without explicitly computing an inverse.

## Metrics

At fixed \(r=8\), RBLA and compact SP+ are compared with the dense optimal error \(E_r^*\):

\[
\Delta E_{j,\kappa}^{\rm method}
=E_{j,\kappa}^{\rm method}-E_r^*.
\]

The CSV also records target and achieved condition numbers, full-update invariance error, absolute SP+ versus dense-optimum error difference, compact full-rank reconstruction error, and transformed factor norms. Factor norms are diagnostics only because factor coordinates are not identifiable.

## Figure

The PDF uses two panels:

- panel (a): median excess error versus \(\kappa(G)\), with interquartile bands and a dense-optimum zero reference;
- panel (b): maximum full-update invariance error and maximum absolute SP+–dense error difference across trials, on a logarithmic y-axis.

The horizontal axes are logarithmic. Panel (b) includes the \(10^{-10}\) acceptance tolerance. Condition number controls anisotropic scaling, not prefix quality by itself, so no monotonic RBLA trend is assumed.

## Numerical checks

`plot_figure.py` verifies achieved condition numbers, nonsingularity, \(\kappa\le10\), full-update invariance, SP+ equivalence, compact full-rank reconstruction, absence of dense-shaped compact intermediates, and finite values. Float64 tolerance is \(10^{-10}\).

The conclusions are limited to broadcast representation. The experiment does not repair aggregation-stage mismatch or imply that SP+ eliminates LoRA gauge freedom.

## Outputs

- plot_figure.py: one-call experiment, validation, CSV export, and adjustable plotting API;
- plot.ipynb: one code cell with one `plot_condition_number_sensitivity(...)` call;
- condition_number_sensitivity.csv: trial-level results;
- condition_number_sensitivity.pdf: vector figure.
