# Gauge-induced excess-error figure

`plot_figure.py` loads and validates the gauge-sensitivity source data, computes excess-error statistics, writes the derived CSV, and draws the figure. The notebook contains one code cell and one call:

```python
fig, ax, summary = plot_gauge_induced_excess_error(...)
```

All formatting options are keyword arguments on the plotting function. Source data, SP+ statistics, checks, metrics, and output paths are available through `summary.attrs`.
