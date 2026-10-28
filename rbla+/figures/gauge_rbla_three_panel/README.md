# Gauge and synthetic RBLA: compact PDF spanning both IEEE text columns

Panel order (left to right, 1 row x 3 columns):

1. **(a) Gauge-induced excess error**, from the existing `gauge_induced_excess_error` figure.
2. **(c) Excess reconstruction error**, the right panel of the synthetic RBLA main figure.
3. **(b) Relative reconstruction error**, the left panel of the synthetic RBLA main figure.

This A-C-B order places the gauge excess-error diagnostic beside the synthetic aggregation excess-error diagnostic, followed by the total reconstruction-error comparison. Panel (a) deliberately applies gauges to a fixed structured update; panels (b,c) use original RBLA-aggregated factors with **no additional gauge transform**. These are different experiments, so the caption states their distinct dimensions, ranks and repeat counts. Panel (a) retains its absolute-rank axis, and panels (b,c) retain normalized-rank axes.

The figure is redrawn from frozen raw data as a vector figure, styled after `fedgra/semantic_coupling_comparison.pdf`. Times New Roman, large axis labels, sparse horizontal ticks and a compact three-panel layout follow the reference. Panel labels are below the axes, and there are no top titles or descriptive footer. Each panel has its own legend above its axes. Median/IQR statistics and the existing controls are retained. No new experiments are run, and the source figures and data are unchanged.

## Outputs

- `gauge_rbla_three_panel.pdf`: the only figure export; a 12 x 5.25 inch vector master with embedded fonts, placed across both IEEE text columns using `width=\textwidth`.
- `caption.txt`: complete English caption distinguishing the two experiments.
- `figure_ieee.tex`: LaTeX `figure*` snippet using `width=\textwidth`.
- `*_plot_summary.csv`: the numerical summaries used for drawing.
- `plot_metadata.json`: layout, sample counts and source validation.
- `sources/`: exact copies of the input CSV/config files and their provenance hashes.
- `qa/`: PDF validation and final visual-review records. Temporary raster previews have been removed.

## Reproduce

From the project root:

```powershell
python "rbla+/figures/gauge_rbla_three_panel/plot_figure.py"
python "rbla+/figures/gauge_rbla_three_panel/verify_figure.py"
```

Requires the same NumPy, pandas, Matplotlib and PyMuPDF environment used for the synthetic experiment. The plotting function accepts figure size, font sizes and spacing. PDF is the only export format. For example:

```powershell
python "rbla+/figures/gauge_rbla_three_panel/plot_figure.py" --width 12 --height 5.25
```

The reference PDF is approximately 11.90 x 4.83 inches and uses 20 pt tick labels, 22 pt horizontal labels, and 25 pt titles. This figure uses a comparable master size with extra space for the required bottom panel labels: 12 x 5.25 inches, 20 pt ticks, 22 pt axis labels, 20 pt bottom labels and 17 pt per-panel legends. At 7.16-inch placement width these become approximately 11.9, 13.1, 11.9 and 10.1 pt, respectively. The `wspace` value is reduced from 0.43 to 0.34 while allowing room for the enlarged vertical labels. The gauge axis labels ranks 1, 4, 8, 12 and 16; the other axes label 0, 0.5 and 1. Every original rank is still plotted. Full experimental definitions remain in `caption.txt`, outside the graphic.

In an IEEE two-column document, use `figure*` and `\includegraphics[width=\textwidth]{...}` as shown in `figure_ieee.tex`. This produces one horizontal figure spanning both text columns. Only PDF is saved; earlier PNG/SVG exports have been removed.
