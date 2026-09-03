"""One-call experiment and plotting API for gauge sensitivity."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


EPS = np.finfo(np.float64).eps
DEFAULT_OUTPUT_BASE = Path(__file__).resolve().parent / "gauge_sensitivity"
DEFAULT_COLORS = {"RBLA prefix": "#C44E52", "SP+ (compact QR)": "#4C72B0"}


def orthonormal_columns(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((rows, columns)), mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs[None, :]


def make_structured_factors(
    d_out: int, d_in: int, rank: int, seed: int, alpha: float = 4.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = orthonormal_columns(rng, d_out, rank)
    v = orthonormal_columns(rng, d_in, rank)
    singular_values = np.exp(-alpha * np.arange(rank) / max(rank - 1, 1))
    sqrt_s = np.sqrt(singular_values)
    return u * sqrt_s[None, :], sqrt_s[:, None] * v.T, singular_values


def make_gauge(gauge_type: str, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if gauge_type == "permutation":
        return np.eye(rank, dtype=np.float64)[:, rng.permutation(rank)]
    if gauge_type == "orthogonal":
        return orthonormal_columns(rng, rank, rank)
    if gauge_type == "diagonal":
        return np.diag(np.exp(rng.uniform(np.log(.5), np.log(2), size=rank)))
    raise ValueError(f"Unknown gauge type: {gauge_type}")


def apply_gauge(
    b_global: np.ndarray, a_global: np.ndarray, gauge: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return b_global @ gauge, np.linalg.solve(gauge, a_global)


def compact_sp_plus(
    b_global: np.ndarray, a_global: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[int, ...]]]:
    q_b, r_b = np.linalg.qr(b_global, mode="reduced")
    q_a, r_a = np.linalg.qr(a_global.T, mode="reduced")
    core = r_b @ r_a.T
    u_core, singular_values, vh_core = np.linalg.svd(core, full_matrices=False)
    sqrt_s = np.sqrt(np.maximum(singular_values, 0.0))
    b_plus = (q_b @ u_core) * sqrt_s[None, :]
    a_plus = sqrt_s[:, None] * (vh_core @ q_a.T)
    shapes = {
        "q_b": q_b.shape, "r_b": r_b.shape, "q_a": q_a.shape, "r_a": r_a.shape,
        "core": core.shape, "u_core": u_core.shape, "vh_core": vh_core.shape,
        "b_plus": b_plus.shape, "a_plus": a_plus.shape,
    }
    dense_shape = (b_global.shape[0], a_global.shape[1])
    if any(shape == dense_shape for shape in shapes.values()):
        raise AssertionError("Compact path created a dense-shaped intermediate")
    return b_plus, a_plus, singular_values, shapes


def relative_frobenius_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    return float(np.linalg.norm(reference - approximation, ord="fro") /
                 (np.linalg.norm(reference, ord="fro") + EPS))


def generate_gauge_sensitivity_data(
    *,
    d_out: int = 128, d_in: int = 96, full_rank: int = 16,
    ranks: Sequence[int] | None = None,
    gauge_types: Sequence[str] = ("permutation", "orthogonal", "diagonal"),
    trials_per_type: int = 100, seed: int = 42, target_seed: int = 20260903,
    alpha: float = 4.0, tolerance: float = 1e-10, condition_limit: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, bool], dict[str, Any]]:
    resolved_ranks = tuple(range(1, full_rank + 1)) if ranks is None else tuple(sorted(set(map(int, ranks))))
    if not resolved_ranks or min(resolved_ranks) < 1 or max(resolved_ranks) > full_rank:
        raise ValueError("ranks must lie in [1, full_rank]")
    if min(resolved_ranks) >= full_rank or trials_per_type <= 0:
        raise ValueError("ranks needs a value below full_rank and trials_per_type must be positive")
    b_global, a_global, _ = make_structured_factors(
        d_out, d_in, full_rank, target_seed, alpha
    )
    delta_w = b_global @ a_global
    u_dense, s_dense, vh_dense = np.linalg.svd(delta_w, full_matrices=False)
    dense_errors, singular_gaps = {}, {}
    for rank in resolved_ranks:
        dense_update = (u_dense[:, :rank] * s_dense[:rank][None, :]) @ vh_dense[:rank, :]
        dense_errors[rank] = relative_frobenius_error(delta_w, dense_update)
        singular_gaps[rank] = float(s_dense[rank - 1] - s_dense[rank])

    records: list[dict[str, Any]] = []
    for type_index, gauge_type in enumerate(gauge_types):
        for trial in range(trials_per_type):
            gauge_seed = seed + 10_000 * type_index + trial
            gauge = make_gauge(gauge_type, full_rank, gauge_seed)
            condition_number = float(np.linalg.cond(gauge, p=2))
            b_gauge, a_gauge = apply_gauge(b_global, a_global, gauge)
            full_update_error = relative_frobenius_error(delta_w, b_gauge @ a_gauge)
            b_plus, a_plus, _, shapes = compact_sp_plus(b_gauge, a_gauge)
            compact_ok = all(shape != delta_w.shape for shape in shapes.values())
            for rank in resolved_ranks:
                rbla_error = relative_frobenius_error(
                    delta_w, b_gauge[:, :rank] @ a_gauge[:rank, :]
                )
                sp_error = relative_frobenius_error(
                    delta_w, b_plus[:, :rank] @ a_plus[:rank, :]
                )
                common = {
                    "dataset": "synthetic", "model": "structured_low_rank_update",
                    "layer": "aggregate", "checkpoint": "N/A", "seed": seed,
                    "gauge_seed": gauge_seed, "trial": trial, "gauge_type": gauge_type,
                    "condition_number": condition_number, "R": full_rank, "r": rank,
                    "full_update_error": full_update_error,
                    "dense_optimal_error": dense_errors[rank],
                    "sp_plus_dense_error_gap": abs(sp_error - dense_errors[rank]),
                    "singular_value_gap": singular_gaps[rank],
                    "compact_path_no_dense_intermediate": compact_ok,
                    "d_out": d_out, "d_in": d_in,
                    "spectrum": f"exp(-{alpha:g}*i/(R-1))",
                }
                records.extend([
                    {**common, "method": "RBLA prefix", "broadcast_error": rbla_error},
                    {**common, "method": "SP+ (compact QR)", "broadcast_error": sp_error},
                ])

    results = pd.DataFrame.from_records(records)
    numeric = ["condition_number", "full_update_error", "broadcast_error",
               "dense_optimal_error", "sp_plus_dense_error_gap", "singular_value_gap"]
    rbla_nontrivial = results[
        results["method"].eq("RBLA prefix")
        & results["gauge_type"].isin(["permutation", "orthogonal"])
        & results["r"].lt(full_rank)
    ]
    diagonal = results[results["method"].eq("RBLA prefix") & results["gauge_type"].eq("diagonal")]
    rbla_ranges = rbla_nontrivial.groupby("r")["broadcast_error"].agg(np.ptp)
    diagonal_ranges = diagonal.groupby("r")["broadcast_error"].agg(np.ptp)
    checks = {
        "all gauge matrices satisfy the condition-number limit": bool(results["condition_number"].max() <= condition_limit),
        "full update is invariant under every gauge": bool(results["full_update_error"].max() <= tolerance),
        "SP+ error matches the dense optimum": bool(results["sp_plus_dense_error_gap"].max() <= tolerance),
        "compact SP+ has no dense-shaped intermediate": bool(results["compact_path_no_dense_intermediate"].all()),
        "all reported numeric values are finite": bool(np.isfinite(results[numeric]).all().all()),
        "all full-rank broadcast errors are near numerical precision": bool(
            results.loc[results["r"].eq(full_rank), "broadcast_error"].max() <= tolerance
        ),
        "non-diagonal gauges produce prefix sensitivity below full rank": bool(rbla_ranges.max() > 1e-3),
        "diagonal scaling acts as a prefix-invariant negative control": bool(diagonal_ranges.max() <= tolerance),
    }
    results["numerical_checks_passed"] = all(checks.values())
    metrics = {"all_checks_passed": all(checks.values()), "ranks": resolved_ranks,
               "gauge_types": tuple(gauge_types), "trials_per_type": trials_per_type,
               "max_full_update_error": float(results["full_update_error"].max()),
               "max_sp_dense_gap": float(results["sp_plus_dense_error_gap"].max())}
    return results, checks, metrics


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


def plot_gauge_sensitivity(
    data: pd.DataFrame | str | Path | None = None, output_base: str | Path | None = None, *,
    d_out: int = 128, d_in: int = 96, full_rank: int = 16,
    ranks: Sequence[int] | None = None,
    gauge_types: Sequence[str] = ("permutation", "orthogonal", "diagonal"),
    trials_per_type: int = 100, seed: int = 42, target_seed: int = 20260903,
    alpha: float = 4.0, tolerance: float = 1e-10, condition_limit: float = 10.0,
    write_csv: bool = True, csv_path: str | Path | None = None,
    assert_checks: bool = True, print_report: bool = True,
    figsize: tuple[float, float] = (7.16, 2.65), dpi: int = 600,
    font_family: str | None = None, font_size: float = 8, axes_label_size: float = 8,
    tick_label_size: float = 7, legend_fontsize: float = 6.2,
    colors: Mapping[str, str] | None = None, box_offsets: Mapping[str, float] | None = None,
    box_width: float = .24, box_alpha: float = .32, show_fliers: bool = False,
    show_means: bool = True, dense_color: str = "black", dense_linestyle: str = "--",
    dense_marker: str = "x", dense_linewidth: float = 1.25,
    xlabel: str = r"Client rank $r$", ylabel: str = "Relative broadcast reconstruction error",
    title: str | None = None, xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None, xticks: Sequence[float] | None = None,
    xticklabels: Sequence[str] | None = None, show_grid: bool = True,
    grid_kwargs: Mapping[str, Any] | None = None, hidden_spines: Sequence[str] = ("top", "right"),
    legend_loc: str = "lower center", legend_bbox_to_anchor: tuple[float, float] | None = (.5, 1.01),
    legend_ncol: int = 3, legend_kwargs: Mapping[str, Any] | None = None,
    figure_kwargs: Mapping[str, Any] | None = None, rc_params: Mapping[str, Any] | None = None,
    boxplot_kwargs: Mapping[str, Any] | None = None,
    tight_layout: bool = True, tight_layout_kwargs: Mapping[str, Any] | None = None,
    subplot_adjust: Mapping[str, float] | None = None,
    save_formats: Sequence[str] = ("pdf",), savefig_kwargs: Mapping[str, Any] | None = None,
    show: bool = False, close: bool = False,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    generated = data is None; checks = metrics = None; resolved_csv = None
    if generated:
        output_base = DEFAULT_OUTPUT_BASE if output_base is None else output_base
        frame, checks, metrics = generate_gauge_sensitivity_data(
            d_out=d_out, d_in=d_in, full_rank=full_rank, ranks=ranks,
            gauge_types=gauge_types, trials_per_type=trials_per_type, seed=seed,
            target_seed=target_seed, alpha=alpha, tolerance=tolerance,
            condition_limit=condition_limit)
        if assert_checks and not metrics["all_checks_passed"]:
            raise AssertionError([name for name, passed in checks.items() if not passed])
        if write_csv:
            resolved_csv = Path(csv_path) if csv_path else Path(output_base).with_name("gauge_sensitivity_results.csv")
            resolved_csv.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(resolved_csv, index=False, float_format="%.16e")
    else:
        frame = data.copy() if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    resolved_ranks = sorted(frame["r"].unique().astype(int)); rank_centers = np.arange(1, len(resolved_ranks) + 1, dtype=float)
    palette = {**DEFAULT_COLORS, **dict(colors or {})}; offsets = {"RBLA prefix": -.16, "SP+ (compact QR)": .16, **dict(box_offsets or {})}
    dense_errors = frame.groupby("r")["dense_optimal_error"].first().reindex(resolved_ranks)
    if font_family is None:
        fonts = {f.name for f in font_manager.fontManager.ttflist}; font_family = "Times New Roman" if "Times New Roman" in fonts else "DejaVu Serif"
    rc = {"font.family": font_family, "font.size": font_size, "axes.labelsize": axes_label_size,
          "xtick.labelsize": tick_label_size, "ytick.labelsize": tick_label_size,
          "legend.fontsize": legend_fontsize, "axes.linewidth": .8,
          "pdf.fonttype": 42, "ps.fonttype": 42, **dict(rc_params or {})}
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(**{"figsize": figsize, "dpi": dpi, **dict(figure_kwargs or {})})
        ax.plot(rank_centers, dense_errors, color=dense_color, linestyle=dense_linestyle,
                marker=dense_marker, linewidth=dense_linewidth, markersize=4.5, zorder=1)
        for method in ("RBLA prefix", "SP+ (compact QR)"):
            distributions = [frame.loc[frame["method"].eq(method) & frame["r"].eq(rank), "broadcast_error"].to_numpy() for rank in resolved_ranks]
            color = palette[method]
            ax.boxplot(distributions, positions=rank_centers + offsets[method], widths=box_width,
                manage_ticks=False, patch_artist=True, showfliers=show_fliers, showmeans=show_means,
                boxprops={"facecolor": color, "edgecolor": color, "alpha": box_alpha, "linewidth": 1.1},
                whiskerprops={"color": color, "linewidth": 1}, capprops={"color": color, "linewidth": 1},
                medianprops={"color": color, "linewidth": 1.5},
                meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": color,
                           "markeredgewidth": .9, "markersize": 3.2}, zorder=3, **dict(boxplot_kwargs or {}))
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if title is not None: ax.set_title(title)
        ticks = rank_centers if xticks is None else xticks
        labels = [str(rank) for rank in resolved_ranks] if xticklabels is None else xticklabels
        ax.set_xticks(ticks); ax.set_xticklabels(labels)
        ax.set_xlim(*(xlim or (.55, len(resolved_ranks) + .45)))
        ax.set_ylim(*(ylim or (0, float(frame["broadcast_error"].max()) * 1.12)))
        ax.grid(show_grid, axis="y", which="major", **{"color": ".75", "linestyle": ":", "linewidth": .5, "alpha": .55, **dict(grid_kwargs or {})})
        ax.set_axisbelow(True)
        for spine in hidden_spines: ax.spines[spine].set_visible(False)
        ax.tick_params(direction="out", width=.8, length=3)
        handles = [
            Patch(facecolor=palette["RBLA prefix"], edgecolor=palette["RBLA prefix"], alpha=box_alpha, label="RBLA prefix"),
            Patch(facecolor=palette["SP+ (compact QR)"], edgecolor=palette["SP+ (compact QR)"], alpha=box_alpha, label="SP+ (compact QR)"),
            Line2D([0], [0], color=dense_color, linestyle=dense_linestyle, marker=dense_marker,
                   linewidth=dense_linewidth, markersize=4.5, label="Dense optimum"),
        ]
        legend_options = {"handles": handles, "loc": legend_loc, "ncol": legend_ncol,
                          "fontsize": legend_fontsize, "handlelength": 1.4,
                          "handletextpad": .35, "columnspacing": .75,
                          "borderaxespad": 0, **dict(legend_kwargs or {})}
        if legend_bbox_to_anchor is not None: legend_options["bbox_to_anchor"] = legend_bbox_to_anchor
        ax.legend(**legend_options)
        if tight_layout: fig.tight_layout(**{"pad": .3, **dict(tight_layout_kwargs or {})})
        if subplot_adjust: fig.subplots_adjust(**dict(subplot_adjust))
        saved = _save(fig, output_base, save_formats, dpi, savefig_kwargs)
        if show: plt.show()
        if close: plt.close(fig)
    summary = frame.groupby(["r", "method"])["broadcast_error"].agg(["mean", "std", "min", "max"]).reset_index()
    summary.attrs.update({"source_data": frame, "checks": checks, "metrics": metrics,
                          "csv_path": resolved_csv, "figure_paths": saved})
    if generated and print_report:
        print(f"Saved {len(frame):,} rows; all numerical checks passed: {metrics['all_checks_passed']}")
        for path in ([resolved_csv] if resolved_csv else []) + saved: print(f"  - {path}")
    return fig, ax, summary


__all__ = ["plot_gauge_sensitivity", "generate_gauge_sensitivity_data", "orthonormal_columns",
           "make_structured_factors", "make_gauge", "apply_gauge", "compact_sp_plus",
           "relative_frobenius_error"]
