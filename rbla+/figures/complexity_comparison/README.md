# Broadcast complexity comparison

This directory contains analytical rank-fixed complexity proxies for square matrices with \(R=r=32\) and \(d\in\{128,256,512,1024,2048,4096,8192\}\). The quantities are deterministic formula evaluations, not measured runtime or hardware-exact FLOPs.

## Figure organization

- `broadcast_complexity.pdf` combines the two absolute quantities:
  - panel (a): rank-fixed server work in analytical work units;
  - panel (b): algorithm-specific auxiliary storage in scalar elements.
- `broadcast_space_complexity.pdf` combines the two relative quantities:
  - panel (a): server-work multiplier relative to compact SP+;
  - panel (b): auxiliary-storage multiplier relative to compact SP+.

The second filename is retained for compatibility with existing manuscript references. In that figure, SP+ is the 1× baseline. RBLA is omitted from the relative panels because its constant-time/index-view proxy is far below 1× and would compress the three computational baselines of interest.

Common factor inputs, serialization/communication, and the final transmitted rank-\(r\) payload are excluded. Library-specific scratch buffers are also excluded from the storage model.

## Work model

\[
C_{\mathrm{RBLA}}=1,
\]

\[
C_{\mathrm{SP+}}=(m+n)R^2+R^3+(m+n)Rr,
\]

\[
C_{\mathrm{DenseTrunc}}=mnR+mnr,
\]

\[
C_{\mathrm{FullSVD}}=mnR+mn\min(m,n).
\]

## Auxiliary-storage model

\[
M_{\mathrm{RBLA}}=1,
\]

\[
M_{\mathrm{SP+}}=2(m+n)R+5R^2+R,
\]

\[
M_{\mathrm{DenseTrunc}}=mn+(m+n)r+r,
\]

\[
M_{\mathrm{FullSVD}}=mn+(m+n)k+k,
\qquad k=\min(m,n).
\]

## Files

- `plot_figure.py`: one-call generator, analytical checks, and fully adjustable plotting API;
- `generate.py`: retained command-line generator;
- `plot.ipynb`: one code cell with one `plot_complexity_comparison(...)` call;
- `broadcast_complexity.csv`: work-unit calculations;
- `broadcast_space_complexity.csv`: auxiliary-storage calculations;
- `broadcast_complexity.pdf`: combined absolute-unit figure;
- `broadcast_space_complexity.pdf`: combined relative-multiplier figure.
