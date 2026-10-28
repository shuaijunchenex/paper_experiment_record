# Independent gauge and synthetic RBLA figures

Three independent PDFs split from the current A-C-B comparison:

| PDF | Original panel | Contents |
|---|---|---|
| `gauge_excess_error.pdf` | a | Gauge-induced excess error |
| `rbla_excess_error.pdf` | c | Synthetic RBLA normalized excess error |
| `rbla_relative_error.pdf` | b | Synthetic RBLA relative reconstruction error |

Each figure retains its own legend, axis labels, colors, measured curves, and interquartile bands. The bottom panel title and letter are removed; no top title or descriptive footer is added. Typography matches the combined figure: Times New Roman, 20 pt ticks, 22 pt axis labels, and 17 pt legends. Each PDF master is 5.2 x 4.9 inches.

All horizontal axes use normalized rank `r/R`, with ticks at `0, 0.2, 0.4, 0.6, 0.8, 1`. Only measured ranks are plotted: gauge starts at `1/16`, and both synthetic plots start at `1/32`. No sample at rank zero is added. All three legends name the method `SP+`.

Both excess-error plots use the label **Normalized excess error** and the definition `(E_method - E_opt) / ||X||_F`, with unsquared Frobenius norms. They retain separate vertical ranges. The archived gauge generator used `||X||_F + machine_epsilon`; the standalone script converts those archived relative errors to the common denominator using the norm of the recorded target spectrum. This affects only floating-point roundoff and leaves the raw CSV intact. `qa/normalization_audit.json` records the conversion and verifies the synthetic definition. The relative-reconstruction plot retains its distinct metric `E_method / ||X||_F` and label **Relative Frobenius error**.

The synthetic excess-error plot includes a black dashed **Optimum (zero)** reference. It is drawn over the coincident SP+ curve so the reference dashes remain visible. The SP+ curve still uses measured errors, and all sampled ranks are unchanged.

The existing `../gauge_rbla_three_panel/` folder, including its combined PDF, is preserved. The script checks all original file hashes before and after generation. Frozen data copies and drawing/analysis helpers are local to this new folder; no experiments are rerun.

From the project root:

```powershell
python "rbla+/figures/gauge_rbla_individual/plot_figures.py"
```

To regenerate only the excess-error PDF, add `--figure rbla_excess_error`.

Only PDF figures are exported by default. `qa/verification.json` records font, page-boundary, title-removal, and original-file preservation checks. `sources/provenance.json` contains input and original-file hashes. Full experimental interpretation remains documented with the combined figure and original synthetic experiment.
