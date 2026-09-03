"""One-call experiment and plotting API for condition-number sensitivity."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


EPS = np.finfo(np.float64).eps
DEFAULT_OUTPUT_BASE = Path(__file__).resolve().parent / "condition_number_sensitivity"
DEFAULT_METHOD_STYLES = {
    "RBLA prefix": {"color": "#C44E52", "marker": "o", "label": "RBLA prefix"},
    "SP+ (compact QR)": {
        "color": "#4C72B0", "marker": "s", "label": "SP+ (compact QR)"
    },
}


def orthonormal_columns(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((rows, columns)), mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs[None, :]


def make_structured_factors(
    d_out: int, d_in: int, rank: int, seed: int, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = orthonormal_columns(rng, d_out, rank)
    v = orthonormal_columns(rng, d_in, rank)
    spectrum = np.exp(-alpha * np.arange(rank) / max(rank - 1, 1))
    spectrum = spectrum / np.linalg.norm(spectrum)
    sqrt_s = np.sqrt(spectrum)
    return u * sqrt_s[None, :], sqrt_s[:, None] * v.T, spectrum


def prescribed_condition_gauge(
    q_left: np.ndarray, q_right: np.ndarray, kappa: float
) -> np.ndarray:
    rank = q_left.shape[1]
    log_scales = np.linspace(-0.5 * np.log(kappa), 0.5 * np.log(kappa), rank)
    return (q_left * np.exp(log_scales)[None, :]) @ q_right.T


def compact_sp_plus(
    b_global: np.ndarray, a_global: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    q_b, r_b = np.linalg.qr(b_global, mode="reduced")
    q_a, r_a = np.linalg.qr(a_global.T, mode="reduced")
    core = r_b @ r_a.T
    u_core, singular_values, vh_core = np.linalg.svd(core, full_matrices=False)
    sqrt_s = np.sqrt(np.maximum(singular_values, 0.0))
    b_plus = (q_b @ u_core) * sqrt_s[None, :]
    a_plus = sqrt_s[:, None] * (vh_core @ q_a.T)
    shapes = [
        q_b.shape, r_b.shape, q_a.shape, r_a.shape, core.shape,
        u_core.shape, vh_core.shape, b_plus.shape, a_plus.shape,
    ]
    dense_shape = (b_global.shape[0], a_global.shape[1])
    if any(shape == dense_shape for shape in shapes):
        raise AssertionError("Compact SP+ created a dense-shaped intermediate")
    return b_plus, a_plus, shapes


def relative_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    return float(
        np.linalg.norm(reference - approximation, ord="fro")
        / (np.linalg.norm(reference, ord="fro") + EPS)
    )


def generate_condition_number_data(
    *,
    d_out: int = 128,
    d_in: int = 96,
    full_rank: int = 16,
    client_rank: int = 8,
    kappas: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0),
    trials: int = 100,
    seed: int = 42,
    target_seed: int = 20260903,
    alpha: float = 4.0,
    tolerance: float = 1e-10,
    condition_rtol: float = 1e-12,
) -> tuple[pd.DataFrame, dict[str, bool], dict[str, Any]]:
    if not 1 <= client_rank <= full_rank <= min(d_out, d_in):
        raise ValueError("Require 1 <= client_rank <= full_rank <= min(d_out, d_in)")
    if trials <= 0 or not kappas or min(kappas) < 1.0:
        raise ValueError("trials must be positive and kappas must be non-empty and >= 1")

    b_global, a_global, _ = make_structured_factors(
        d_out, d_in, full_rank, target_seed, alpha
    )
    delta_w = b_global @ a_global
    dense_singular_values = np.linalg.svd(delta_w, compute_uv=False)
    dense_optimal_error = float(
        np.linalg.norm(dense_singular_values[client_rank:])
        / (np.linalg.norm(dense_singular_values) + EPS)
    )
    records: list[dict[str, Any]] = []
    for trial in range(trials):
        q_left = orthonormal_columns(
            np.random.default_rng(seed + 2 * trial), full_rank, full_rank
        )
        q_right = orthonormal_columns(
            np.random.default_rng(seed + 2 * trial + 1), full_rank, full_rank
        )
        for target_kappa in kappas:
            gauge = prescribed_condition_gauge(q_left, q_right, float(target_kappa))
            actual_kappa = float(np.linalg.cond(gauge, p=2))
            smallest_singular_value = float(np.linalg.svd(gauge, compute_uv=False)[-1])
            b_gauge = b_global @ gauge
            a_gauge = np.linalg.solve(gauge, a_global)
            full_update_error = relative_error(delta_w, b_gauge @ a_gauge)
            b_plus, a_plus, compact_shapes = compact_sp_plus(b_gauge, a_gauge)
            rbla_update = b_gauge[:, :client_rank] @ a_gauge[:client_rank, :]
            sp_update = b_plus[:, :client_rank] @ a_plus[:client_rank, :]
            rbla_error = relative_error(delta_w, rbla_update)
            sp_error = relative_error(delta_w, sp_update)
            common = {
                "dataset": "synthetic", "model": "structured_low_rank_update",
                "layer": "aggregate", "checkpoint": "N/A", "seed": seed,
                "trial": trial, "d_out": d_out, "d_in": d_in,
                "R": full_rank, "r": client_rank, "alpha": alpha,
                "target_condition_number": target_kappa,
                "actual_condition_number": actual_kappa,
                "condition_relative_error": abs(actual_kappa - target_kappa) / target_kappa,
                "smallest_gauge_singular_value": smallest_singular_value,
                "b_gauge_frobenius_norm": float(np.linalg.norm(b_gauge, ord="fro")),
                "a_gauge_frobenius_norm": float(np.linalg.norm(a_gauge, ord="fro")),
                "full_update_error": full_update_error,
                "dense_optimal_error": dense_optimal_error,
                "sp_plus_dense_error_gap": abs(sp_error - dense_optimal_error),
                "compact_full_reconstruction_error": relative_error(
                    delta_w, b_plus @ a_plus
                ),
                "compact_path_no_dense_intermediate": all(
                    shape != delta_w.shape for shape in compact_shapes
                ),
                "gauge_construction": "Q_left D(kappa) Q_right^T",
            }
            records.extend([
                {**common, "method": "RBLA prefix", "broadcast_error": rbla_error,
                 "excess_error": rbla_error - dense_optimal_error},
                {**common, "method": "SP+ (compact QR)", "broadcast_error": sp_error,
                 "excess_error": sp_error - dense_optimal_error},
            ])

    results = pd.DataFrame.from_records(records)
    rbla = results[results["method"].eq("RBLA prefix")]
    numeric_columns = [
        "target_condition_number", "actual_condition_number", "condition_relative_error",
        "smallest_gauge_singular_value", "b_gauge_frobenius_norm",
        "a_gauge_frobenius_norm", "full_update_error", "broadcast_error",
        "dense_optimal_error", "sp_plus_dense_error_gap",
        "compact_full_reconstruction_error", "excess_error",
    ]
    checks = {
        "achieved condition numbers match their targets": bool(
            results["condition_relative_error"].max() <= condition_rtol
        ),
        "all gauges are nonsingular": bool(results["smallest_gauge_singular_value"].min() > 0),
        "all gauges satisfy the condition-number limit": bool(
            results["actual_condition_number"].max() <= max(kappas) * (1 + condition_rtol)
        ),
        "all complete updates are gauge invariant": bool(results["full_update_error"].max() <= tolerance),
        "SP+ matches the dense optimum": bool(results["sp_plus_dense_error_gap"].max() <= tolerance),
        "compact full-rank reconstruction is accurate": bool(
            results["compact_full_reconstruction_error"].max() <= tolerance
        ),
        "compact SP+ has no dense-shaped intermediate": bool(
            results["compact_path_no_dense_intermediate"].all()
        ),
        "all numeric values are finite": bool(np.isfinite(results[numeric_columns]).all().all()),
        "RBLA excess is non-negative within tolerance": bool(rbla["excess_error"].min() >= -tolerance),
    }
    results["numerical_checks_passed"] = all(checks.values())
    metrics = {
        "all_checks_passed": all(checks.values()), "d_out": d_out, "d_in": d_in,
        "full_rank": full_rank, "client_rank": client_rank, "kappas": tuple(kappas),
        "trials": trials, "maximum_excess": float(rbla["excess_error"].max()),
    }
    return results, checks, metrics


def _save(fig: plt.Figure, output_base: str | Path | None, formats: Sequence[str],
          dpi: int, savefig_kwargs: Mapping[str, Any] | None) -> list[Path]:
    if output_base is None:
        return []
    base = Path(output_base).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in formats:
        path = base.with_suffix("." + suffix.lstrip("."))
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02, **dict(savefig_kwargs or {})}
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            kwargs.setdefault("dpi", dpi)
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def plot_condition_number_sensitivity(
    data: pd.DataFrame | str | Path | None = None,
    output_base: str | Path | None = None,
    *,
    d_out: int = 128, d_in: int = 96, full_rank: int = 16, client_rank: int = 8,
    kappas: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0),
    trials: int = 100, seed: int = 42, target_seed: int = 20260903,
    alpha: float = 4.0, tolerance: float = 1e-10, condition_rtol: float = 1e-12,
    write_csv: bool = True, csv_path: str | Path | None = None,
    assert_checks: bool = True, print_report: bool = True,
    figsize: tuple[float, float] = (7.16, 2.70), dpi: int = 600,
    font_family: str | None = None, font_size: float = 8,
    axes_label_size: float = 8, tick_label_size: float = 7,
    title_size: float = 8, legend_fontsize: float = 7,
    method_styles: Mapping[str, Mapping[str, Any]] | None = None,
    line_width: float = 1.45, marker_size: float = 4.2, band_alpha: float = 0.14,
    error_xlabel: str = r"Gauge condition number $\kappa(G)$",
    error_ylabel: str = "Excess error over dense optimum",
    error_title: str = r"(a) Broadcast excess at fixed $r=8$",
    residual_xlabel: str = r"Gauge condition number $\kappa(G)$",
    residual_ylabel: str = "Maximum numerical residual",
    residual_title: str = "(b) Numerical stability",
    error_xlim: tuple[float | None, float | None] | None = None,
    error_ylim: tuple[float | None, float | None] | None = None,
    residual_xlim: tuple[float | None, float | None] | None = None,
    residual_ylim: tuple[float | None, float | None] = (5e-17, 5e-10),
    xscale: str = "log", residual_yscale: str = "log",
    xticks: Sequence[float] | None = None, xticklabels: Sequence[str] | None = None,
    show_grid: bool = True, grid_kwargs: Mapping[str, Any] | None = None,
    error_legend_loc: str = "upper right", residual_legend_loc: str = "upper left",
    error_legend_kwargs: Mapping[str, Any] | None = None,
    residual_legend_kwargs: Mapping[str, Any] | None = None,
    figure_kwargs: Mapping[str, Any] | None = None, rc_params: Mapping[str, Any] | None = None,
    plot_kwargs: Mapping[str, Any] | None = None,
    fill_between_kwargs: Mapping[str, Any] | None = None,
    tight_layout: bool = True, tight_layout_kwargs: Mapping[str, Any] | None = None,
    subplot_adjust: Mapping[str, float] | None = None,
    save_formats: Sequence[str] = ("pdf",), savefig_kwargs: Mapping[str, Any] | None = None,
    show: bool = False, close: bool = False,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes], dict[str, pd.DataFrame]]:
    generated = data is None
    checks = metrics = None
    resolved_csv = None
    if generated:
        output_base = DEFAULT_OUTPUT_BASE if output_base is None else output_base
        frame, checks, metrics = generate_condition_number_data(
            d_out=d_out, d_in=d_in, full_rank=full_rank, client_rank=client_rank,
            kappas=kappas, trials=trials, seed=seed, target_seed=target_seed,
            alpha=alpha, tolerance=tolerance, condition_rtol=condition_rtol,
        )
        if assert_checks and not metrics["all_checks_passed"]:
            raise AssertionError([name for name, passed in checks.items() if not passed])
        if write_csv:
            resolved_csv = Path(csv_path) if csv_path else Path(output_base).with_suffix(".csv")
            resolved_csv.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(resolved_csv, index=False, float_format="%.16e")
    else:
        frame = data.copy() if isinstance(data, pd.DataFrame) else pd.read_csv(data)

    method_stats = (
        frame.groupby(["target_condition_number", "method"])["excess_error"]
        .agg(median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
        .reset_index()
    )
    residual_stats = (
        frame.groupby("target_condition_number")
        .agg(max_full_update_error=("full_update_error", "max"),
             max_sp_dense_gap=("sp_plus_dense_error_gap", "max"))
        .reset_index()
    )
    styles = deepcopy(DEFAULT_METHOD_STYLES)
    for name, values in (method_styles or {}).items():
        styles.setdefault(name, {}).update(values)
    if font_family is None:
        fonts = {font.name for font in font_manager.fontManager.ttflist}
        font_family = "Times New Roman" if "Times New Roman" in fonts else "DejaVu Serif"
    rc = {"font.family": font_family, "font.size": font_size,
          "axes.labelsize": axes_label_size, "axes.titlesize": title_size,
          "xtick.labelsize": tick_label_size, "ytick.labelsize": tick_label_size,
          "legend.fontsize": legend_fontsize, "axes.linewidth": .8,
          "pdf.fonttype": 42, "ps.fonttype": 42, **dict(rc_params or {})}
    with mpl.rc_context(rc):
        fig, (ax_error, ax_residual) = plt.subplots(
            1, 2, **{"figsize": figsize, "dpi": dpi, **dict(figure_kwargs or {})}
        )
        ax_error.axhline(0, color="black", linestyle="--", linewidth=1,
                         label="Dense optimum (zero)", zorder=1)
        for method, style in styles.items():
            group = method_stats[method_stats["method"].eq(method)].sort_values("target_condition_number")
            x = group["target_condition_number"].to_numpy(float)
            ax_error.fill_between(x, group["q25"], group["q75"],
                **{"color": style["color"], "alpha": band_alpha, "linewidth": 0,
                   **dict(fill_between_kwargs or {})})
            ax_error.plot(x, group["median"],
                **{"color": style["color"], "marker": style["marker"],
                   "label": style["label"], "linewidth": line_width,
                   "markersize": marker_size, "zorder": 3, **dict(plot_kwargs or {})})
        maximum_excess = float(frame.loc[frame["method"].eq("RBLA prefix"), "excess_error"].max())
        if error_ylim is None:
            error_ylim = (-.025 * maximum_excess, 1.08 * maximum_excess)
        x = residual_stats["target_condition_number"].to_numpy(float)
        ax_residual.plot(x, np.maximum(residual_stats["max_full_update_error"], EPS),
            color="#555555", marker="o", label="Full-update invariance",
            linewidth=line_width, markersize=marker_size)
        ax_residual.plot(x, np.maximum(residual_stats["max_sp_dense_gap"], EPS),
            color="#4C72B0", marker="s", label="SP+ vs. dense optimum",
            linewidth=line_width, markersize=marker_size)
        tolerance_exponent = int(np.round(np.log10(tolerance)))
        tolerance_label = (
            rf"Tolerance $10^{{{tolerance_exponent}}}$"
            if np.isclose(tolerance, 10.0 ** tolerance_exponent)
            else rf"Tolerance ${tolerance:.1e}$"
        )
        ax_residual.axhline(tolerance, color="#C44E52", linestyle=":", linewidth=1.2,
                            label=tolerance_label)
        ticks = list(kappas) if xticks is None else list(xticks)
        labels = [f"{value:g}" for value in ticks] if xticklabels is None else list(xticklabels)
        for ax, xlabel, ylabel, title, limits in (
            (ax_error, error_xlabel, error_ylabel, error_title, (error_xlim, error_ylim)),
            (ax_residual, residual_xlabel, residual_ylabel, residual_title,
             (residual_xlim, residual_ylim)),
        ):
            ax.set_xscale(xscale); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, pad=3)
            ax.set_xticks(ticks); ax.set_xticklabels(labels)
            if limits[0] is not None: ax.set_xlim(*limits[0])
            if limits[1] is not None: ax.set_ylim(*limits[1])
            ax.grid(show_grid, which="major", **{"color": ".75", "linestyle": ":",
                    "linewidth": .5, "alpha": .55, **dict(grid_kwargs or {})})
            ax.set_axisbelow(True); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out", width=.8, length=3)
        ax_residual.set_yscale(residual_yscale)
        ax_error.legend(loc=error_legend_loc, handlelength=2,
                        **dict(error_legend_kwargs or {}))
        ax_residual.legend(loc=residual_legend_loc, handlelength=2,
                           **dict(residual_legend_kwargs or {}))
        if tight_layout: fig.tight_layout(**{"pad": .3, "w_pad": 1.0, **dict(tight_layout_kwargs or {})})
        if subplot_adjust: fig.subplots_adjust(**dict(subplot_adjust))
        saved = _save(fig, output_base, save_formats, dpi, savefig_kwargs)
        if show: plt.show()
        if close: plt.close(fig)
    stats = {"method": method_stats, "residual": residual_stats}
    stats["method"].attrs.update({"source_data": frame, "checks": checks, "metrics": metrics,
                                  "csv_path": resolved_csv, "figure_paths": saved})
    if generated and print_report:
        print(f"Saved {len(frame):,} rows; all numerical checks passed: {metrics['all_checks_passed']}")
        for path in ([resolved_csv] if resolved_csv else []) + saved: print(f"  - {path}")
    return fig, (ax_error, ax_residual), stats


__all__ = ["plot_condition_number_sensitivity", "generate_condition_number_data",
           "orthonormal_columns", "make_structured_factors", "prescribed_condition_gauge",
           "compact_sp_plus", "relative_error"]
