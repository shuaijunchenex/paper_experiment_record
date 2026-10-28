# Synthetic RBLA aggregation and redistribution experiment

Complete paired reconstruction experiment using generated client LoRA factors. The default design fixes **7 settings x 30 seeds = 210 aggregations**, scanning **7,200 seed-rank pairs**. The main setting is fixed in advance at R=32, eta=0.3. No full federated training is performed.

## Reproduce from the project root

```powershell
python -m pip install -r "rbla+/figures/synthetic_rbla_redistribution/requirements.txt"
python "rbla+/figures/synthetic_rbla_redistribution/run.py"
python "rbla+/figures/synthetic_rbla_redistribution/verify_outputs.py"
```

The runner validates the small dense cases before running the main setting, then the supplementary and sensitivity settings. It generates raw data, summaries, all seven paper figures, full captions, and the report. The output verifier checks saved factors/data, reproduces one prespecified seed per setting, and renders PDFs for visual inspection. No training data, checkpoints, GPU, or network is needed after dependencies are installed. Use Python 3.11; exact executed versions are in `requirements.txt` and `results/run_manifest.json`.

Separate operations:

```powershell
python "rbla+/figures/synthetic_rbla_redistribution/run.py" --validate-only
python "rbla+/figures/synthetic_rbla_redistribution/plot_figure.py"
```

To use another configuration without overwriting the delivered results:

```powershell
python "rbla+/figures/synthetic_rbla_redistribution/run.py" --config "path/to/config.json" --output "path/to/new_results"
python "rbla+/figures/synthetic_rbla_redistribution/verify_outputs.py" --results "path/to/new_results"
```

`config.json` exposes seeds, dimensions, ranks, weights, generator, eta, main setting, and factor saving. Adapter width must be smaller than both matrix dimensions. Keep `effective_lora_scaling=1`, float64, and no post-aggregation gauge. `prepare_config.py` was used only to freeze the initial configuration and existing canonicalization function before any results were computed; it is not needed for reproduction. Running it again would re-snapshot the upstream function.

## Files

| File | Purpose |
|---|---|
| `core.py` | RMS-scaled generation, original per-slot RBLA averaging, stable compact residual measurements |
| `reference_sp_plus.py` | Frozen existing compact QR-SVD SP+ implementation |
| `validate.py` | Independent small dense SVD/residual validation and edge cases |
| `run.py` | Configurable end-to-end experiment and summary tables |
| `plot_figure.py` | English two-panel figures from saved raw data |
| `report.py` | Regenerate the bilingual scientific report |
| `verify_outputs.py` | Data integrity, numerical reproducibility, PDF/PNG/SVG checks |
| `config.json`, `provenance.json`, `source_audit.md` | Prespecified settings/seeds and honest source/scaling audit |
| `results/report.md`, `results/paper_summary_en.txt` | Results and paper-ready English summary |
| `results/raw/*.csv`, `results/raw_all.csv` | Unclipped measurements for every setting/seed/rank |
| `results/rank_summary.csv` | Pointwise seed median, IQR, mean, SD, extrema, valid count |
| `results/client_weighted_by_seed.csv` | Weighted metrics at actual client ranks, within each seed |
| `results/client_weighted_summary.csv` | Across-seed summaries of actual-client weighted metrics |
| `results/all_rank_scan_*.csv` | Separate uniformly weighted diagnostic scan, not client benefit |
| `results/slot_participation.csv` | Per-index participation counts and weight denominators |
| `results/numerical_diagnostics.csv` | Per-aggregation rank, identity, residual and tolerance checks |
| `results/factors/*.npz` | All original global B/A factors and their compact singular values |
| `results/validation/*` | Raw dense checks and validation summary |
| `results/run_manifest.json`, `results/config_frozen.json` | Executed dependencies, BLAS, times, commands and hashes |
| `results/figures/*.{pdf,svg,png}` | Main and six supplementary dual-panel paper figures; PNG at 600 dpi |
| `results/figures/captions.md`, `main_caption.txt` | Complete paper captions |
| `results/qa/*` | Integrity/format checks and PDF-render previews |

## Statistical and numerical conventions

Every row compares **the same** X=B_global A_global, produced by a single original RBLA aggregation. The global factor SHA-256 identifies the paired target. SP+ is decomposed once per aggregation and serves all ranks. Its error is measured from its actual output factors, not assigned from the optimum.

All errors are **unsquared Frobenius norms**. `error_*` are absolute errors; `relative_*` divide by ||X||_F; `excess_*=(error_*-error_opt)/||X||_F`; `gain=(error_pre-error_sp)/||X||_F`; `reduction=(error_pre-error_sp)/error_pre` when the denominator exceeds the recorded numerical tolerance. `absolute_excess_*` and `absolute_gain` are retained. NA is an empty CSV field, never silently replaced with zero.

The figure uses pointwise medians and 25th-75th percentiles. Client summaries first use the real r_i and p_i within a seed and then summarize across seeds. `weighted_total_reduction` is a ratio of weighted errors per seed; `mean_client_reduction_conditional` is the weighted mean of defined per-client reductions, renormalized by `reduction_available_weight`. For the defaults, the full-rank client's reduction is undefined, so the latter has available weight 0.9. Neither is the uniform all-rank scan. Median paired gains need not equal differences between separate method medians.

The normalization floor is `64*eps*||B||F*||A||F`, and the absolute acceptance tolerance is `128*eps*||B||F*||A||F + 1e-12*||X||F`. Raw CSV values are never clipped. Only tiny negative excess values within that tolerance may be displayed at zero. Positive residuals and endpoint floating-point differences remain in the data. Figure axes have no ceiling of one.

Scientific scope: these are synthetic aggregate reconstruction results. Task accuracy, convergence, fairness, aggregation bias, and rank-data mismatch require real checkpoint and training evaluations.
