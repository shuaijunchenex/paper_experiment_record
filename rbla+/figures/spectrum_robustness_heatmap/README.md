# Spectrum and rank-scale robustness heatmaps

## Purpose

This synthetic experiment tests whether gauge-induced RBLA prefix error depends on one particular singular spectrum or on the small latent rank used in the original demonstration. It varies spectral decay and full latent rank while keeping the construction, gauge families, and trial-seed policy fixed.

The experiment concerns broadcast representation only. It does not test training accuracy, repair aggregation-stage semantic mismatch, or claim that SP+ eliminates LoRA gauge freedom.

## Rank-scale design

The sweep uses

\[
R\in\{16,64,160\},\qquad r=1,\ldots,R,
\]

and 1,000 seeded gauges per family for every \(R\). Results are shown against normalized rank \(r/R\), while a second PDF shows the \(R=160\) result against absolute rank, including the practical range from 4 to 160.

The Monte Carlo sweep is evaluated exactly in the latent \(R\times R\) space. With column-orthonormal \(U,V\), multiplication by \(U\) and \(V^\top\) preserves the Frobenius norm, so ambient dimensions do not affect the plotted relative errors as long as \(d_{\rm out},d_{\rm in}\ge R\). Separate end-to-end checks use both \(128\times96,R=16\) and \(256\times192,R=160\) matrices to validate this reduction and the compact SP+ implementation.

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

The tested spectrum parameters are

\[
\alpha\in\{0.5,1,2,3,4,6,8\}.
\]

The row \(\alpha=4\) is scale-equivalent to the main gauge-sensitivity experiment. Normalizing the spectrum does not change its relative-error metric.

## Gauge construction

For every \(R\), the same 1,000 trial seeds are reused across all \(\alpha\) values within each family:

- permutation: a random permutation matrix;
- orthogonal: a seeded Gaussian matrix followed by QR with deterministic sign normalization.

For either invertible gauge \(G\),

\[
\widetilde B_g=B_gG,\qquad
\widetilde A_g=G^{-1}A_g.
\]

The end-to-end implementation evaluates `numpy.linalg.solve(G, A_g)`, never an explicit inverse.

## Broadcast calculations

RBLA uses

\[
\widehat{\Delta W}_{r,j}^{\rm RBLA}
=\widetilde B_g^{(j)}[:,:r]\widetilde A_g^{(j)}[:r,:].
\]

For the orthogonal family, with \(P_r=G[:,:r]G[:,:r]^\top\), the sweep evaluates the exactly equivalent latent expression

\[
\left\|\Sigma-\Sigma^{1/2}P_r\Sigma^{1/2}\right\|_F.
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

The optimal error \(E_r^*\) is the normalized tail energy of the designed singular spectrum. Each heatmap cell reports the median across 1,000 trials. The summary CSV also contains the mean, sample standard deviation, quartiles, minimum, and maximum. The compressed NPZ retains all 3,360,000 trial-cell excess errors and the optimal-error arrays.

## Figures

`spectrum_robustness_heatmap.pdf` contains six heatmaps with one shared color scale:

- rows: permutation and orthogonal gauges;
- columns: \(R=16,64,160\);
- horizontal axis: normalized rank \(r/R\);
- vertical axis: the numerically spaced spectral-decay parameter \(\alpha\);
- color: median RBLA excess error.

`spectrum_robustness_absolute_rank.pdf` shows the two \(R=160\) panels using absolute rank \(r\). A white dashed rectangle marks \(\alpha=4\) in both figures. SP+ has no redundant heatmap because the end-to-end checks verify its excess error against numerical zero.

## Numerical checks

The generator verifies:

- orthogonality for every sampled orthogonal gauge;
- non-negative RBLA excess error within float64 tolerance;
- full-rank zero residual in the latent calculation;
- full-update invariance in ambient matrices;
- compact SP+ equivalence to the optimal spectral tail at representative ranks;
- absence of a dense \(d_{\rm out}\times d_{\rm in}\) intermediate inside compact SP+;
- condition number one for the sampled permutation and orthogonal gauges.

The tolerance is \(10^{-10}\). No monotonicity in \(\alpha\), rank, or excess error is assumed.

## Outputs

- `plot_figure.py`: one-call sweep, checks, artifact export, and fully adjustable plotting API;
- `generate.py`: retained command-line generator;
- `plot.ipynb`: one code cell with one `plot_spectrum_robustness_heatmap(...)` call;
- `spectrum_robustness_heatmap.csv`: cell-level summary statistics;
- `spectrum_robustness_trials.npz`: compressed trial-level arrays;
- `spectrum_robustness_checks.csv`: numerical validation results;
- `spectrum_robustness_heatmap.pdf`: normalized-rank figure;
- `spectrum_robustness_absolute_rank.pdf`: absolute-rank figure.
