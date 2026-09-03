"""One-call analysis and plotting API for gauge-induced excess error."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D


TOLERANCE = 1e-10
FIGURES_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_CSV = FIGURES_ROOT / "gauge_sensitivity" / "gauge_sensitivity_results.csv"
DEFAULT_OUTPUT_BASE = Path(__file__).resolve().parent / "gauge_induced_excess_error"
DEFAULT_GAUGE_ORDER = ("permutation", "orthogonal", "diagonal")
DEFAULT_STYLES = {
    "permutation": {"color": "#C44E52", "linestyle": "-", "marker": "o", "label": "RBLA–perm."},
    "orthogonal": {"color": "#DD8452", "linestyle": "--", "marker": "s", "label": "RBLA–orth."},
    "diagonal": {"color": "#7F7F7F", "linestyle": ":", "marker": "^", "label": "RBLA–diag. control"},
}


def prepare_excess_error_data(
    source: pd.DataFrame | str | Path = DEFAULT_SOURCE_CSV,
    *, tolerance: float = TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, bool], dict[str, Any]]:
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    required = {"gauge_type", "trial", "R", "r", "method", "broadcast_error",
                "dense_optimal_error", "full_update_error", "numerical_checks_passed"}
    missing = required.difference(frame.columns)
    if missing: raise ValueError(f"Source data is missing columns: {sorted(missing)}")
    frame["excess_error"] = frame["broadcast_error"] - frame["dense_optimal_error"]
    frame["reference_method"] = "Dense truncated SVD optimum"
    frame["source_results"] = "../gauge_sensitivity/gauge_sensitivity_results.csv"
    ranks = sorted(frame["r"].unique().astype(int)); maximum_rank = int(frame["R"].max())
    rbla = frame[frame["method"].eq("RBLA prefix")]
    sp_plus = frame[frame["method"].eq("SP+ (compact QR)")]
    gauge_stats = (
        rbla.groupby(["gauge_type", "r"])["excess_error"]
        .agg(median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75),
             mean="mean", std="std", minimum="min", maximum="max").reset_index()
    )
    sp_stats = (
        sp_plus.groupby("r")["excess_error"]
        .agg(median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75),
             maximum_absolute=lambda x: np.abs(x).max()).reset_index()
    )
    source_checks = bool((frame["numerical_checks_passed"].astype(str).str.lower() == "true").all())
    diagonal_excess = rbla.loc[rbla["gauge_type"].eq("diagonal"), "excess_error"]
    full_rank_excess = frame.loc[frame["r"].eq(maximum_rank), "excess_error"]
    checks = {
        "source gauge experiment passed its numerical checks": source_checks,
        "all analysis values are finite": bool(np.isfinite(frame["excess_error"]).all()),
        "excess errors are non-negative within tolerance": bool(frame["excess_error"].min() >= -tolerance),
        "SP+ excess error is numerical zero": bool(sp_plus["excess_error"].abs().max() <= tolerance),
        "diagonal RBLA is a zero-excess negative control": bool(diagonal_excess.abs().max() <= tolerance),
        "full-rank excess error is numerical zero": bool(full_rank_excess.abs().max() <= tolerance),
    }
    frame["analysis_checks_passed"] = all(checks.values())
    metrics = {"all_checks_passed": all(checks.values()), "ranks": ranks,
               "maximum_rank": maximum_rank, "maximum_excess": float(rbla["excess_error"].max()),
               "maximum_sp_excess": float(sp_plus["excess_error"].abs().max()),
               "maximum_diagonal_excess": float(diagonal_excess.abs().max())}
    return frame, gauge_stats, sp_stats, checks, metrics


def _save(fig: plt.Figure, base: str | Path | None, formats: Sequence[str], dpi: int,
          kwargs: Mapping[str, Any] | None) -> list[Path]:
    if base is None: return []
    base = Path(base).with_suffix(""); base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in formats:
        path = base.with_suffix("." + suffix.lstrip("."))
        options = {"bbox_inches": "tight", "pad_inches": .02, **dict(kwargs or {})}
        if path.suffix.lower() == ".png": options.setdefault("dpi", dpi)
        fig.savefig(path, **options); paths.append(path)
    return paths


def plot_gauge_induced_excess_error(
    source: pd.DataFrame | str | Path = DEFAULT_SOURCE_CSV,
    output_base: str | Path | None = DEFAULT_OUTPUT_BASE,
    *,
    tolerance: float = TOLERANCE, write_csv: bool = True,
    csv_path: str | Path | None = None, assert_checks: bool = True,
    print_report: bool = True,
    gauge_order: Sequence[str] = DEFAULT_GAUGE_ORDER,
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    figsize: tuple[float, float] = (7.16, 2.70), dpi: int = 600,
    font_family: str | None = None, font_size: float = 8,
    axes_label_size: float = 8, tick_label_size: float = 7,
    legend_fontsize: float = 6.4, line_width: float = 1.45,
    marker_size: float = 4.0, diagonal_marker_size: float = 5.2,
    band_alpha: float = .13, show_bands: bool = True,
    sp_color: str = "#4C72B0", sp_marker: str = "D", sp_marker_size: float = 3.3,
    xlabel: str = r"Client rank $r$", ylabel: str = "Excess error over dense optimum",
    title: str | None = None, xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None, xticks: Sequence[float] | None = None,
    xticklabels: Sequence[str] | None = None,
    show_grid: bool = True, grid_axis: str = "y", grid_kwargs: Mapping[str, Any] | None = None,
    hidden_spines: Sequence[str] = ("top", "right"),
    legend_loc: str = "lower center", legend_bbox_to_anchor: tuple[float, float] | None = (.5, 1.01),
    legend_ncol: int = 5, legend_kwargs: Mapping[str, Any] | None = None,
    figure_kwargs: Mapping[str, Any] | None = None, rc_params: Mapping[str, Any] | None = None,
    plot_kwargs: Mapping[str, Any] | None = None,
    fill_between_kwargs: Mapping[str, Any] | None = None,
    tight_layout: bool = True, tight_layout_kwargs: Mapping[str, Any] | None = None,
    subplot_adjust: Mapping[str, float] | None = None,
    save_formats: Sequence[str] = ("pdf",), savefig_kwargs: Mapping[str, Any] | None = None,
    show: bool = False, close: bool = False,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    frame, gauge_stats, sp_stats, checks, metrics = prepare_excess_error_data(source, tolerance=tolerance)
    if assert_checks and not metrics["all_checks_passed"]:
        raise AssertionError([name for name, passed in checks.items() if not passed])
    resolved_csv = None
    if write_csv:
        resolved_csv = Path(csv_path) if csv_path else Path(output_base).with_suffix(".csv")
        resolved_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(resolved_csv, index=False, float_format="%.16e")
    resolved_styles = deepcopy(DEFAULT_STYLES)
    for name, values in (styles or {}).items(): resolved_styles.setdefault(name, {}).update(values)
    if font_family is None:
        fonts = {f.name for f in font_manager.fontManager.ttflist}; font_family = "Times New Roman" if "Times New Roman" in fonts else "DejaVu Serif"
    rc = {"font.family": font_family, "font.size": font_size, "axes.labelsize": axes_label_size,
          "xtick.labelsize": tick_label_size, "ytick.labelsize": tick_label_size,
          "legend.fontsize": legend_fontsize, "axes.linewidth": .8,
          "pdf.fonttype": 42, "ps.fonttype": 42, **dict(rc_params or {})}
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(**{"figsize": figsize, "dpi": dpi, **dict(figure_kwargs or {})})
        ax.axhline(0, color="black", linestyle="--", linewidth=1, zorder=1)
        for gauge_type in gauge_order:
            group = gauge_stats[gauge_stats["gauge_type"].eq(gauge_type)].sort_values("r")
            style = resolved_styles[gauge_type]; x = group["r"].to_numpy(float)
            if show_bands and gauge_type != "diagonal":
                ax.fill_between(x, group["q25"], group["q75"],
                    **{"color": style["color"], "alpha": band_alpha, "linewidth": 0,
                       "zorder": 2, **dict(fill_between_kwargs or {})})
            ax.plot(x, group["median"], color=style["color"], linestyle=style["linestyle"],
                    marker=style["marker"], markerfacecolor="white" if gauge_type == "diagonal" else style["color"],
                    markeredgecolor=style["color"], markersize=diagonal_marker_size if gauge_type == "diagonal" else marker_size,
                    linewidth=line_width, label=style["label"], zorder=3, **dict(plot_kwargs or {}))
        sp_group = sp_stats.sort_values("r")
        ax.plot(sp_group["r"], sp_group["median"], color=sp_color, linestyle="-", marker=sp_marker,
                markersize=sp_marker_size, markerfacecolor=sp_color, markeredgecolor="white",
                markeredgewidth=.45, linewidth=line_width, label="SP+", zorder=5)
        ranks = metrics["ranks"]; ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if title is not None: ax.set_title(title)
        ticks = ranks if xticks is None else xticks; labels = [str(x) for x in ticks] if xticklabels is None else xticklabels
        ax.set_xticks(ticks); ax.set_xticklabels(labels); ax.set_xlim(*(xlim or (min(ranks) - .45, max(ranks) + .45)))
        ax.set_ylim(*(ylim or (-.025 * metrics["maximum_excess"], 1.08 * metrics["maximum_excess"])))
        ax.grid(show_grid, axis=grid_axis, which="major", **{"color": ".75", "linestyle": ":", "linewidth": .5, "alpha": .55, **dict(grid_kwargs or {})})
        ax.set_axisbelow(True)
        for spine in hidden_spines: ax.spines[spine].set_visible(False)
        ax.tick_params(direction="out", width=.8, length=3)
        handles, labels = ax.get_legend_handles_labels(); handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=1)); labels.append("Dense optimum (zero)")
        options = {"handles": handles, "labels": labels, "loc": legend_loc, "ncol": legend_ncol,
                   "fontsize": legend_fontsize, "handlelength": 1.5, "handletextpad": .35,
                   "columnspacing": .8, "borderaxespad": 0, **dict(legend_kwargs or {})}
        if legend_bbox_to_anchor is not None: options["bbox_to_anchor"] = legend_bbox_to_anchor
        ax.legend(**options)
        if tight_layout: fig.tight_layout(**{"pad": .3, **dict(tight_layout_kwargs or {})})
        if subplot_adjust: fig.subplots_adjust(**dict(subplot_adjust))
        saved = _save(fig, output_base, save_formats, dpi, savefig_kwargs)
        if show: plt.show()
        if close: plt.close(fig)
    gauge_stats.attrs.update({"source_data": frame, "sp_stats": sp_stats, "checks": checks,
                              "metrics": metrics, "csv_path": resolved_csv, "figure_paths": saved})
    if print_report:
        print(f"Loaded {len(frame):,} rows; all analysis checks passed: {metrics['all_checks_passed']}")
        for path in ([resolved_csv] if resolved_csv else []) + saved: print(f"  - {path}")
    return fig, ax, gauge_stats


__all__ = ["plot_gauge_induced_excess_error", "prepare_excess_error_data"]
