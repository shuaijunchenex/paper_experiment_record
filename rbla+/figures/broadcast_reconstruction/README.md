# Broadcast reconstruction figure

`plot_figure.py` provides one notebook-facing entry point:

```python
plot_broadcast_reconstruction(data=None, output_base=None, **options)
```

Leave `data=None` to run the entire synthetic experiment, numerical checks, CSV export, and plotting pipeline with one call. Alternatively, pass a raw experiment `DataFrame` or the path to `broadcast_reconstruction_error.csv` to redraw without rerunning the experiment. The function returns `(fig, ax, summary)`, so callers can continue editing the Matplotlib objects after the function returns. It does not mutate input data or global `matplotlib.rcParams`.

The same module also owns the reusable experiment helpers formerly defined in the notebook:

- `orthonormal_columns`;
- `make_synthetic_target`;
- `rotated_lora_factors`;
- `compact_sp_plus`;
- `relative_frobenius_error`;
- `relative_matrix_difference`;
- `analytic_optimal_errors`.

The notebook has one code cell and calls only `plot_broadcast_reconstruction`. All generation, validation, statistics, saving, and plotting helpers remain in `plot_figure.py`. When the experiment is generated, `summary.attrs` contains `source_data`, `diagnostics`, `checks`, `metrics`, `csv_path`, and `figure_paths` for inspection.

## Quick use

```python
from plot_figure import plot_broadcast_reconstruction

fig, ax, summary = plot_broadcast_reconstruction(
    output_base="broadcast_reconstruction_error",
    d_out=128,
    d_in=96,
    full_rank=8,
    rotation_seeds=tuple(range(42, 62)),
    figsize=(3.5, 2.5),
    save_formats=("pdf",),
)
```

## Manual formatting example

```python
fig, ax, summary = plot_broadcast_reconstruction(
    data="broadcast_reconstruction_error.csv",
    output_base="broadcast_reconstruction_error_custom",
    figsize=(4.2, 3.0),
    xlabel_size=10,
    ylabel_size=10,
    x_tick_label_size=8,
    y_tick_label_size=8,
    xlim=(0.10, 1.02),
    ylim=(0.0, 1.05),
    method_labels={
        "RBLA index prefix": "RBLA",
        "SP+ (compact QR)": "SP+",
        "Dense truncated SVD": "Dense optimum",
    },
    method_styles={
        "RBLA index prefix": {"color": "#B2182B", "marker": "o"},
        "SP+ (compact QR)": {"color": "#2166AC", "marker": "s"},
    },
    band_alpha=0.15,
    legend_loc="upper center",
    legend_bbox_to_anchor=(0.5, 1.18),
    legend_ncol=3,
    legend_kwargs={"columnspacing": 1.0, "handlelength": 2.2},
    grid_axis="both",
    grid_kwargs={"linestyle": ":", "linewidth": 0.5, "alpha": 0.5},
    subplot_adjust={"left": 0.17, "right": 0.98, "bottom": 0.18, "top": 0.82},
    save_formats=("pdf", "png"),
    savefig_kwargs={"facecolor": "white"},
)
```

Frequently changed properties have explicit keyword arguments. The mappings `rc_params`, `figure_kwargs`, `plot_kwargs`, `method_styles`, `band_styles`, `fill_between_kwargs`, `xlabel_kwargs`, `ylabel_kwargs`, `title_kwargs`, `tick_params`, `grid_kwargs`, `legend_kwargs`, `tight_layout_kwargs`, and `savefig_kwargs` pass remaining Matplotlib options through to the corresponding object.

Set `ax` to an existing `Axes` to draw the figure as a panel inside a larger layout. Set `show=False` and `close=False` when making additional manual adjustments after the function call.
