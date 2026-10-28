# Condition-number sensitivity through 100,000

The PDF shows the controlled FP64 gauge experiment at fixed r=8, R=16, using the same 128 x 96 synthetic update and 100 paired gauge trials at each condition number. Every trial reuses its orthogonal gauge components across levels.

The seven archived levels (1, 1.5, 2, 3, 5, 7.5, 10) are retained unchanged. Twelve newly evaluated levels are added: 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, and 100000. These are computed trials, not interpolated observations. `extended_condition_raw.csv` stores the 2400 added method/trial records; original data remain in `sources/condition_raw.csv`.

The horizontal axis is logarithmic, with decade ticks from 10^0 to 10^5. The vertical axis is symmetric-logarithmic, linear between -1 and 1 (linthresh=1, linscale=0.7), so both zero and large errors remain visible. Lines show medians and bands the 25th--75th percentiles across trials. The 5.2 x 4.9 inch page, Times New Roman, colors, and independent legend are retained.

Normalized excess error is (E_method - E_opt) / ||X||_F using unsquared Frobenius norms. The plot script corrects the frozen generator's epsilon-normalized denominators using the same fixed target. Signed raw values and summary statistics are preserved; only tolerance-sized negative plotted values are displayed at zero. At kappa=100000, RBLA's median is 2067.06 and interquartile range is 1524.61--2633.13. SP+ remains at numerical zero.

## Reproduction

From the project root, run `python -B rbla+/figures/condition_number_individual/extend_experiment.py` to recompute the added levels and kappa=10 continuity anchor. Run `python -B rbla+/figures/condition_number_individual/plot_figure.py` to render recorded data without rerunning trials.

Frozen inputs are verified against sources/provenance.json; added results against extended_config.json. Original figures elsewhere in the repository are preserved. The repeated kappa=10 anchor matches archived broadcast errors within 1.23e-15.

qa/extension_audit.json records all checks. Reconstruction and optimality retain the original 1e-10 tolerance. Condition-number measurement uses relative tolerance 1e-9 for FP64 sensitivity at kappa=1e5; observed maximum mismatch is 2.09e-12. Maximum full-update relative error is 1.72e-12, compact reconstruction error is 3.82e-12, and SP+ optimality gap is 2.78e-17. All checks pass. Normalization, font and page-boundary checks are also recorded in qa/.

Suggested caption:

> Normalized excess reconstruction error versus gauge condition number at fixed r/R=8/16. Curves and shaded bands show medians and interquartile ranges over 100 paired gauge trials for a fixed 128 x 96 synthetic update. The condition number ranges from 1 to 10^5; each trial reuses its orthogonal gauge components across levels. Errors use unsquared Frobenius norms relative to the dense-SVD optimum. The horizontal axis is logarithmic and the vertical axis is symmetric-logarithmic, with a linear region up to an error of 1 to retain the zero reference.
