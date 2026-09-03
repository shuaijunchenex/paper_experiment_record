# Gauge sensitivity figure

`plot_figure.py` contains the complete synthetic experiment, gauge helpers, compact SP+ implementation, numerical checks, CSV export, and plotting code. The notebook contains one code cell and one call:

```python
fig, ax, summary = plot_gauge_sensitivity(...)
```

Leave `data=None` to regenerate the experiment. Pass `gauge_sensitivity_results.csv` through `data` to adjust the figure without rerunning the experiment. Generated source data, checks, metrics, and output paths are available through `summary.attrs`.

