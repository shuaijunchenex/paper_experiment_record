"""One-call experiment and plotting API for broadcast reconstruction.

``plot_broadcast_reconstruction`` is the only function a notebook needs to
call.  With ``data=None`` it generates and validates the synthetic experiment,
writes the CSV, and draws the figure.  Supplying a DataFrame or CSV path skips
the experiment and redraws existing data.  Plot formatting remains local to
the call, with no global ``matplotlib.rcParams`` side effects.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure


FLOAT64_EPS = np.finfo(np.float64).eps

DEFAULT_METHOD_ORDER = (
    "RBLA index prefix",
    "SP+ (compact QR)",
    "Dense truncated SVD",
)

DEFAULT_METHOD_STYLES: dict[str, dict[str, Any]] = {
    "RBLA index prefix": {
        "color": "#C44E52",
        "marker": "o",
        "linestyle": "-",
        "zorder": 2,
    },
    "SP+ (compact QR)": {
        "color": "#4C72B0",
        "marker": "s",
        "linestyle": "-",
        "zorder": 3,
    },
    "Dense truncated SVD": {
        "color": "black",
        "marker": "^",
        "markerfacecolor": "white",
        "markeredgewidth": 0.8,
        "linestyle": "--",
        "dashes": (4, 2),
        "zorder": 4,
    },
}

DEFAULT_OUTPUT_BASE = Path(__file__).resolve().parent / "broadcast_reconstruction_error"


def orthonormal_columns(
    rng: np.random.Generator, rows: int, columns: int
) -> np.ndarray:
    """Return a deterministic thin orthonormal basis for a seeded RNG."""
    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    if columns > rows:
        raise ValueError("columns must not exceed rows")
    q, r = np.linalg.qr(rng.standard_normal((rows, columns)), mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs[None, :]


def make_synthetic_target(
    d_out: int,
    d_in: int,
    rank: int,
    seed: int,
    *,
    spectral_decay: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create orthonormal bases and an exponential singular spectrum."""
    if rank <= 0:
        raise ValueError("rank must be positive")
    if rank > min(d_out, d_in):
        raise ValueError("rank must not exceed min(d_out, d_in)")
    if spectral_decay < 0:
        raise ValueError("spectral_decay must be non-negative")
    rng = np.random.default_rng(seed)
    u = orthonormal_columns(rng, d_out, rank)
    v = orthonormal_columns(rng, d_in, rank)
    singular_values = np.exp(
        -spectral_decay * np.arange(rank, dtype=np.float64) / max(rank - 1, 1)
    )
    return u, singular_values, v


def rotated_lora_factors(
    u: np.ndarray,
    singular_values: np.ndarray,
    v: np.ndarray,
    rotation_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Represent the same target in a seeded orthogonally mixed rank basis."""
    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.ndim != 1 or singular_values.size == 0:
        raise ValueError("singular_values must be a non-empty vector")
    if np.any(singular_values < 0.0):
        raise ValueError("singular_values must be non-negative")
    rank = singular_values.size
    if u.ndim != 2 or v.ndim != 2 or u.shape[1] != rank or v.shape[1] != rank:
        raise ValueError("u and v must have rank columns matching singular_values")
    rng = np.random.default_rng(rotation_seed)
    rotation = orthonormal_columns(rng, rank, rank)
    sqrt_s = np.sqrt(singular_values)
    b_global = (u * sqrt_s[None, :]) @ rotation
    a_global = rotation.T @ (sqrt_s[:, None] * v.T)
    return b_global, a_global


def compact_sp_plus(
    b_global: np.ndarray, a_global: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[int, ...]]]:
    """Canonicalize LoRA factors using two thin QRs and an R x R core SVD."""
    if b_global.ndim != 2 or a_global.ndim != 2:
        raise ValueError("LoRA factors must be matrices")
    if b_global.shape[1] != a_global.shape[0]:
        raise ValueError("incompatible LoRA factor ranks")

    q_b, r_b = np.linalg.qr(b_global, mode="reduced")
    q_a, r_a = np.linalg.qr(a_global.T, mode="reduced")
    core = r_b @ r_a.T
    u_core, singular_values, vh_core = np.linalg.svd(core, full_matrices=False)
    sqrt_s = np.sqrt(np.maximum(singular_values, 0.0))
    b_plus = (q_b @ u_core) * sqrt_s[None, :]
    a_plus = sqrt_s[:, None] * (vh_core @ q_a.T)

    intermediate_shapes = {
        "q_b": q_b.shape,
        "r_b": r_b.shape,
        "q_a": q_a.shape,
        "r_a": r_a.shape,
        "core": core.shape,
        "u_core": u_core.shape,
        "vh_core": vh_core.shape,
        "b_plus": b_plus.shape,
        "a_plus": a_plus.shape,
    }
    dense_shape = (b_global.shape[0], a_global.shape[1])
    if any(shape == dense_shape for shape in intermediate_shapes.values()):
        raise AssertionError(f"compact path created a dense-shaped intermediate: {dense_shape}")
    return b_plus, a_plus, singular_values, intermediate_shapes


def relative_frobenius_error(
    reference: np.ndarray,
    approximation: np.ndarray,
    *,
    epsilon: float = FLOAT64_EPS,
) -> float:
    """Return ||reference - approximation||_F / (||reference||_F + epsilon)."""
    if reference.shape != approximation.shape:
        raise ValueError("reference and approximation must have the same shape")
    return float(
        np.linalg.norm(reference - approximation, ord="fro")
        / (np.linalg.norm(reference, ord="fro") + epsilon)
    )


def relative_matrix_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float = FLOAT64_EPS,
) -> float:
    """Return ||left - right||_F / (||right||_F + epsilon)."""
    if left.shape != right.shape:
        raise ValueError("left and right must have the same shape")
    return float(
        np.linalg.norm(left - right, ord="fro")
        / (np.linalg.norm(right, ord="fro") + epsilon)
    )


def analytic_optimal_errors(
    singular_values: np.ndarray,
    ranks: Sequence[int] | None = None,
) -> dict[int, float]:
    """Return optimal relative Frobenius rank-r errors from spectral tails."""
    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.ndim != 1 or singular_values.size == 0:
        raise ValueError("singular_values must be a non-empty vector")
    denominator = float(np.linalg.norm(singular_values))
    if denominator == 0.0:
        raise ValueError("singular_values must not be all zero")
    if ranks is None:
        ranks = range(1, singular_values.size + 1)
    resolved: dict[int, float] = {}
    for rank in ranks:
        rank = int(rank)
        if rank < 1 or rank > singular_values.size:
            raise ValueError(f"rank {rank} is outside [1, {singular_values.size}]")
        resolved[rank] = float(np.linalg.norm(singular_values[rank:]) / denominator)
    return resolved


def generate_broadcast_reconstruction_data(
    *,
    d_out: int = 128,
    d_in: int = 96,
    full_rank: int = 8,
    ranks: Sequence[int] | None = None,
    rotation_seeds: Sequence[int] = tuple(range(42, 62)),
    base_seed: int = 20260903,
    spectral_decay: float = 4.0,
    tolerance_reconstruction: float = 1e-12,
    tolerance_equivalence: float = 1e-11,
    tolerance_monotonicity: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool], dict[str, Any]]:
    """Generate the complete synthetic experiment and numerical diagnostics."""
    if full_rank <= 0:
        raise ValueError("full_rank must be positive")
    if full_rank > min(d_out, d_in):
        raise ValueError("full_rank must not exceed min(d_out, d_in)")
    resolved_ranks = np.asarray(
        list(range(1, full_rank + 1)) if ranks is None else list(ranks),
        dtype=int,
    )
    if resolved_ranks.ndim != 1 or resolved_ranks.size == 0:
        raise ValueError("ranks must be a non-empty one-dimensional sequence")
    if len(np.unique(resolved_ranks)) != len(resolved_ranks):
        raise ValueError("ranks must not contain duplicates")
    if np.any(resolved_ranks < 1) or np.any(resolved_ranks > full_rank):
        raise ValueError(f"ranks must lie in [1, {full_rank}]")
    resolved_ranks = np.sort(resolved_ranks)
    resolved_seeds = tuple(int(seed) for seed in rotation_seeds)
    if not resolved_seeds:
        raise ValueError("rotation_seeds must not be empty")

    u_target, target_singular_values, v_target = make_synthetic_target(
        d_out,
        d_in,
        full_rank,
        base_seed,
        spectral_decay=spectral_decay,
    )
    analytic_errors = analytic_optimal_errors(target_singular_values, resolved_ranks)
    spectrum_label = f"exp(-{spectral_decay:g}*i/(R-1))"

    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for seed in resolved_seeds:
        b_global, a_global = rotated_lora_factors(
            u_target, target_singular_values, v_target, seed
        )
        b_plus, a_plus, compact_singular_values, compact_shapes = compact_sp_plus(
            b_global, a_global
        )

        # Dense matrices are deliberately confined to this offline evaluator.
        delta_w = b_global @ a_global
        u_dense, s_dense, vh_dense = np.linalg.svd(delta_w, full_matrices=False)
        full_compact = b_plus @ a_plus
        full_reconstruction_error = relative_frobenius_error(delta_w, full_compact)

        for client_rank in resolved_ranks:
            client_rank = int(client_rank)
            rbla_update = (
                b_global[:, :client_rank] @ a_global[:client_rank, :]
            )
            sp_update = b_plus[:, :client_rank] @ a_plus[:client_rank, :]
            dense_update = (
                u_dense[:, :client_rank] * s_dense[:client_rank][None, :]
            ) @ vh_dense[:client_rank, :]

            rbla_error = relative_frobenius_error(delta_w, rbla_update)
            sp_error = relative_frobenius_error(delta_w, sp_update)
            dense_error = relative_frobenius_error(delta_w, dense_update)
            sp_dense_difference = relative_matrix_difference(sp_update, dense_update)
            common = {
                "dataset": "synthetic",
                "model": "synthetic_low_rank_update",
                "layer_or_aggregate": "aggregate",
                "seed": seed,
                "round": -1,
                "R": full_rank,
                "r": client_rank,
                "normalized_rank": client_rank / full_rank,
                "sp_plus_dense_difference": sp_dense_difference,
                "gap": rbla_error - sp_error,
                "analytic_optimal_error": analytic_errors[client_rank],
                "experiment_type": "structured_synthetic_rotation",
                "checkpoint": "N/A",
                "d_out": d_out,
                "d_in": d_in,
                "spectrum": spectrum_label,
            }
            for method, error in (
                ("RBLA index prefix", rbla_error),
                ("SP+ (compact QR)", sp_error),
                ("Dense truncated SVD", dense_error),
            ):
                records.append(
                    {**common, "method": method, "relative_error": error}
                )

        diagnostics.append(
            {
                "seed": seed,
                "full_reconstruction_error": full_reconstruction_error,
                "compact_spectrum_difference": float(
                    np.max(
                        np.abs(compact_singular_values - target_singular_values)
                    )
                ),
                "compact_path_has_no_dense_intermediate": all(
                    shape != (d_out, d_in) for shape in compact_shapes.values()
                ),
            }
        )

    results = pd.DataFrame.from_records(records)
    diagnostics_frame = pd.DataFrame.from_records(diagnostics)
    pivot = results.pivot_table(
        index=["seed", "r"], columns="method", values="relative_error"
    )
    max_error_difference = float(
        np.max(
            np.abs(
                pivot["SP+ (compact QR)"] - pivot["Dense truncated SVD"]
            )
        )
    )
    max_matrix_difference = float(results["sp_plus_dense_difference"].max())
    analytic_by_row = np.array(
        [analytic_errors[int(rank)] for _, rank in pivot.index], dtype=float
    )
    max_analytic_difference = float(
        np.max(
            np.abs(pivot["SP+ (compact QR)"].to_numpy() - analytic_by_row)
        )
    )

    max_increase: dict[str, float] = {}
    for method in results["method"].unique():
        method_max = 0.0
        for _, group in results[results["method"].eq(method)].groupby("seed"):
            increases = np.diff(
                group.sort_values("r")["relative_error"].to_numpy()
            )
            method_max = max(method_max, float(np.max(increases, initial=0.0)))
        max_increase[method] = method_max

    checks = {
        "compact path has no dense-shaped intermediate": bool(
            diagnostics_frame["compact_path_has_no_dense_intermediate"].all()
        ),
        "full-rank compact reconstruction": bool(
            diagnostics_frame["full_reconstruction_error"].max()
            <= tolerance_reconstruction
        ),
        "SP+ error matches dense SVD": (
            max_error_difference <= tolerance_equivalence
        ),
        "SP+ matrix matches dense SVD": (
            max_matrix_difference <= tolerance_equivalence
        ),
        "SP+ matches analytic spectral tail": (
            max_analytic_difference <= tolerance_equivalence
        ),
        "SP+ errors are non-increasing": (
            max_increase["SP+ (compact QR)"] <= tolerance_monotonicity
        ),
        "dense SVD errors are non-increasing": (
            max_increase["Dense truncated SVD"] <= tolerance_monotonicity
        ),
        "RBLA errors are non-increasing for this construction": (
            max_increase["RBLA index prefix"] <= tolerance_monotonicity
        ),
        "SP+ is never worse than RBLA prefix": bool(
            np.all(
                pivot["SP+ (compact QR)"]
                <= pivot["RBLA index prefix"] + tolerance_equivalence
            )
        ),
    }
    all_checks_passed = all(checks.values())
    results["numerical_checks_passed"] = all_checks_passed
    metrics = {
        "d_out": d_out,
        "d_in": d_in,
        "full_rank": full_rank,
        "ranks": resolved_ranks.tolist(),
        "rotation_seeds": resolved_seeds,
        "base_seed": base_seed,
        "spectral_decay": spectral_decay,
        "max_error_difference": max_error_difference,
        "max_matrix_difference": max_matrix_difference,
        "max_analytic_difference": max_analytic_difference,
        "max_increase": max_increase,
        "all_checks_passed": all_checks_passed,
    }
    return results, diagnostics_frame, checks, metrics


def _load_data(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    path = Path(data)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _prepare_summary(
    data: pd.DataFrame,
    *,
    method_order: Sequence[str],
    method_column: str,
    x_column: str,
    y_column: str,
    center_stat: str,
    spread_stat: str | None,
    spread_scale: float,
    band_clip_lower: float | None,
) -> pd.DataFrame:
    required = {method_column, x_column, y_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Plot data is missing columns: {sorted(missing)}")
    absent_methods = [method for method in method_order if method not in set(data[method_column])]
    if absent_methods:
        raise ValueError(f"Plot data is missing methods: {absent_methods}")
    if center_stat not in {"mean", "median"}:
        raise ValueError("center_stat must be 'mean' or 'median'")
    if spread_stat not in {None, "std", "iqr"}:
        raise ValueError("spread_stat must be None, 'std', or 'iqr'")
    if spread_scale < 0:
        raise ValueError("spread_scale must be non-negative")

    selected = data[data[method_column].isin(method_order)].copy()
    selected[y_column] = pd.to_numeric(selected[y_column], errors="raise")
    selected[x_column] = pd.to_numeric(selected[x_column], errors="raise")
    grouped = selected.groupby([method_column, x_column], as_index=False)[y_column]
    center = grouped.agg(center_stat).rename(columns={y_column: "center"})

    if spread_stat is None:
        center["spread"] = 0.0
        center["lower"] = center["center"]
        center["upper"] = center["center"]
        return center

    if spread_stat == "std":
        spread = grouped.std().rename(columns={y_column: "spread"})
        summary = center.merge(spread, on=[method_column, x_column], validate="one_to_one")
        summary["spread"] = summary["spread"].fillna(0.0)
        summary["lower"] = summary["center"] - spread_scale * summary["spread"]
        summary["upper"] = summary["center"] + spread_scale * summary["spread"]
    else:
        quantiles = (
            selected.groupby([method_column, x_column])[y_column]
            .quantile([0.25, 0.75])
            .unstack()
            .rename(columns={0.25: "q25", 0.75: "q75"})
            .reset_index()
        )
        summary = center.merge(
            quantiles, on=[method_column, x_column], validate="one_to_one"
        )
        summary["spread"] = summary["q75"] - summary["q25"]
        summary["lower"] = summary["center"] - spread_scale * (
            summary["center"] - summary["q25"]
        )
        summary["upper"] = summary["center"] + spread_scale * (
            summary["q75"] - summary["center"]
        )

    if band_clip_lower is not None:
        summary["lower"] = summary["lower"].clip(lower=band_clip_lower)
    return summary


def _resolve_method_styles(
    method_order: Sequence[str],
    method_styles: Mapping[str, Mapping[str, Any]] | None,
    *,
    line_width: float,
    marker_size: float,
    plot_kwargs: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    resolved = deepcopy(DEFAULT_METHOD_STYLES)
    overrides = method_styles or {}
    common = {"linewidth": line_width, "markersize": marker_size, **(plot_kwargs or {})}
    for method in method_order:
        style = {**common, **resolved.get(method, {}), **dict(overrides.get(method, {}))}
        resolved[method] = style
    return resolved


def _set_limits(ax: Axes, *, xlim: tuple[float | None, float | None] | None,
                ylim: tuple[float | None, float | None] | None) -> None:
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])


def _configure_axis(
    ax: Axes,
    *,
    xlabel: str | None,
    ylabel: str | None,
    title: str | None,
    axes_label_size: float,
    xlabel_size: float | None,
    ylabel_size: float | None,
    title_size: float,
    tick_label_size: float,
    x_tick_label_size: float | None,
    y_tick_label_size: float | None,
    xlim: tuple[float | None, float | None] | None,
    ylim: tuple[float | None, float | None] | None,
    xticks: Sequence[float] | None,
    xticklabels: Sequence[str] | None,
    yticks: Sequence[float] | None,
    yticklabels: Sequence[str] | None,
    xscale: str,
    yscale: str,
    xlabel_kwargs: Mapping[str, Any] | None,
    ylabel_kwargs: Mapping[str, Any] | None,
    title_kwargs: Mapping[str, Any] | None,
    tick_params: Mapping[str, Any] | None,
    show_grid: bool,
    grid_axis: str,
    grid_which: str,
    grid_kwargs: Mapping[str, Any] | None,
    hidden_spines: Sequence[str],
    axis_below: bool,
    aspect: str | float | None,
) -> None:
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    _set_limits(ax, xlim=xlim, ylim=ylim)
    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            **{
                "fontsize": axes_label_size if xlabel_size is None else xlabel_size,
                **dict(xlabel_kwargs or {}),
            },
        )
    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            **{
                "fontsize": axes_label_size if ylabel_size is None else ylabel_size,
                **dict(ylabel_kwargs or {}),
            },
        )
    if title is not None:
        ax.set_title(title, **{"fontsize": title_size, **dict(title_kwargs or {})})
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    ax.tick_params(
        **{
            "labelsize": tick_label_size,
            "direction": "out",
            "width": 0.8,
            "length": 3,
            **dict(tick_params or {}),
        }
    )
    if x_tick_label_size is not None:
        ax.tick_params(axis="x", labelsize=x_tick_label_size)
    if y_tick_label_size is not None:
        ax.tick_params(axis="y", labelsize=y_tick_label_size)
    ax.grid(
        **{
            "visible": show_grid,
            "axis": grid_axis,
            "which": grid_which,
            "color": "0.75",
            "linestyle": ":",
            "linewidth": 0.5,
            "alpha": 0.55,
            **dict(grid_kwargs or {}),
        },
    )
    ax.set_axisbelow(axis_below)
    for spine in hidden_spines:
        if spine not in ax.spines:
            raise ValueError(f"Unknown spine: {spine}")
        ax.spines[spine].set_visible(False)
    if aspect is not None:
        ax.set_aspect(aspect)


def _save_figure(
    fig: Figure,
    *,
    output_base: str | Path | None,
    save_formats: Sequence[str],
    dpi: int,
    bbox_inches: str | None,
    pad_inches: float,
    transparent: bool,
    savefig_kwargs: Mapping[str, Any] | None,
) -> list[Path]:
    if output_base is None:
        return []
    base = Path(output_base)
    if base.suffix:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in save_formats:
        suffix = suffix.lower().lstrip(".")
        path = base.with_suffix(f".{suffix}")
        kwargs = {
            "bbox_inches": bbox_inches,
            "pad_inches": pad_inches,
            "transparent": transparent,
            **dict(savefig_kwargs or {}),
        }
        if suffix in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs.setdefault("dpi", dpi)
        fig.savefig(path, **kwargs)
        saved.append(path)
    return saved


def _print_experiment_report(
    data: pd.DataFrame,
    diagnostics: pd.DataFrame,
    checks: Mapping[str, bool],
    metrics: Mapping[str, Any],
    csv_path: Path | None,
    figure_paths: Sequence[Path],
) -> None:
    mean_errors = data.groupby("method")["relative_error"].mean()
    mean_by_rank = (
        data.groupby(["r", "method"])["relative_error"].mean().unstack()
    )
    reductions = (
        mean_by_rank["RBLA index prefix"] - mean_by_rank["SP+ (compact QR)"]
    )
    max_reduction_rank = int(reductions.idxmax())
    seeds = metrics["rotation_seeds"]

    print("Synthetic experiment summary")
    print("  source: structured synthetic LoRA factors; checkpoint: N/A")
    print(
        f"  shape: {metrics['d_out']} x {metrics['d_in']}; "
        f"R={metrics['full_rank']}; ranks={metrics['ranks']}"
    )
    print(
        f"  factor rotations: {len(seeds)} seeds "
        f"({min(seeds)}-{max(seeds)})"
    )
    print(f"  mean RBLA error: {mean_errors['RBLA index prefix']:.6e}")
    print(f"  mean SP+ error: {mean_errors['SP+ (compact QR)']:.6e}")
    print(
        "  mean dense-SVD error: "
        f"{mean_errors['Dense truncated SVD']:.6e}"
    )
    print(
        f"  maximum mean error reduction: {reductions.max():.6e} "
        f"at r={max_reduction_rank}"
    )
    print(
        "  maximum SP+/dense matrix difference: "
        f"{metrics['max_matrix_difference']:.6e}"
    )
    print(
        "  maximum full-rank compact reconstruction error: "
        f"{diagnostics['full_reconstruction_error'].max():.6e}"
    )
    print(f"  all numerical checks passed: {metrics['all_checks_passed']}")
    for name, passed in checks.items():
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
    output_paths = ([csv_path] if csv_path is not None else []) + list(figure_paths)
    if output_paths:
        print("  outputs:")
        for path in output_paths:
            print(f"    - {path}")


def plot_broadcast_reconstruction(
    data: pd.DataFrame | str | Path | None = None,
    output_base: str | Path | None = None,
    *,
    d_out: int = 128,
    d_in: int = 96,
    full_rank: int = 8,
    ranks: Sequence[int] | None = None,
    rotation_seeds: Sequence[int] = tuple(range(42, 62)),
    base_seed: int = 20260903,
    spectral_decay: float = 4.0,
    tolerance_reconstruction: float = 1e-12,
    tolerance_equivalence: float = 1e-11,
    tolerance_monotonicity: float = 1e-12,
    write_csv: bool = True,
    csv_path: str | Path | None = None,
    assert_numerical_checks: bool = True,
    print_experiment_report: bool = True,
    method_order: Sequence[str] = DEFAULT_METHOD_ORDER,
    method_labels: Mapping[str, str] | None = None,
    method_column: str = "method",
    x_column: str = "normalized_rank",
    y_column: str = "relative_error",
    center_stat: str = "mean",
    spread_stat: str | None = "std",
    spread_scale: float = 1.0,
    show_error_band: bool = True,
    band_clip_lower: float | None = 0.0,
    figsize: tuple[float, float] = (3.5, 2.5),
    dpi: int = 600,
    figure_kwargs: Mapping[str, Any] | None = None,
    rc_params: Mapping[str, Any] | None = None,
    subplot_adjust: Mapping[str, float] | None = None,
    tight_layout: bool = True,
    tight_layout_kwargs: Mapping[str, Any] | None = None,
    font_family: str | None = None,
    font_size: float = 8,
    axes_label_size: float = 8,
    xlabel_size: float | None = None,
    ylabel_size: float | None = None,
    tick_label_size: float = 7,
    x_tick_label_size: float | None = None,
    y_tick_label_size: float | None = None,
    title_size: float = 8,
    xlabel: str | None = r"Normalized client rank $r/R$",
    ylabel: str | None = "Relative reconstruction error",
    title: str | None = None,
    xlim: tuple[float | None, float | None] | None = (0.1, 1.02),
    ylim: tuple[float | None, float | None] | None = (0.0, None),
    xticks: Sequence[float] | None = None,
    xticklabels: Sequence[str] | None = None,
    yticks: Sequence[float] | None = None,
    yticklabels: Sequence[str] | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    xlabel_kwargs: Mapping[str, Any] | None = None,
    ylabel_kwargs: Mapping[str, Any] | None = None,
    title_kwargs: Mapping[str, Any] | None = None,
    tick_params: Mapping[str, Any] | None = None,
    method_styles: Mapping[str, Mapping[str, Any]] | None = None,
    line_width: float = 1.45,
    marker_size: float = 4.2,
    plot_kwargs: Mapping[str, Any] | None = None,
    band_alpha: float = 0.12,
    band_styles: Mapping[str, Mapping[str, Any]] | None = None,
    fill_between_kwargs: Mapping[str, Any] | None = None,
    show_legend: bool = True,
    legend_loc: str = "upper right",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    legend_ncol: int = 1,
    legend_fontsize: float = 7,
    legend_frameon: bool = False,
    legend_kwargs: Mapping[str, Any] | None = None,
    show_grid: bool = True,
    grid_axis: str = "both",
    grid_which: str = "major",
    grid_kwargs: Mapping[str, Any] | None = None,
    hidden_spines: Sequence[str] = ("top", "right"),
    axis_below: bool = True,
    aspect: str | float | None = None,
    axes_position: Sequence[float] | None = None,
    save_formats: Sequence[str] = ("pdf",),
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.02,
    transparent: bool = False,
    savefig_kwargs: Mapping[str, Any] | None = None,
    show: bool = False,
    close: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Generate (when needed), validate, save, and plot reconstruction error.

    Leave ``data`` as ``None`` for the notebook's one-call workflow. In that
    mode the experiment is generated by :func:`generate_broadcast_reconstruction_data`,
    saved next to ``output_base``, numerically checked, and then plotted. Pass a
    DataFrame or CSV path to redraw existing results without running the
    experiment. The returned summary stores generated raw data and diagnostics
    in ``summary.attrs``.
    """
    generated = data is None
    diagnostics: pd.DataFrame | None = None
    checks: dict[str, bool] | None = None
    metrics: dict[str, Any] | None = None
    resolved_csv_path: Path | None = None
    if generated:
        if output_base is None:
            output_base = DEFAULT_OUTPUT_BASE
        frame, diagnostics, checks, metrics = generate_broadcast_reconstruction_data(
            d_out=d_out,
            d_in=d_in,
            full_rank=full_rank,
            ranks=ranks,
            rotation_seeds=rotation_seeds,
            base_seed=base_seed,
            spectral_decay=spectral_decay,
            tolerance_reconstruction=tolerance_reconstruction,
            tolerance_equivalence=tolerance_equivalence,
            tolerance_monotonicity=tolerance_monotonicity,
        )
        if assert_numerical_checks and not metrics["all_checks_passed"]:
            failed = [name for name, passed in checks.items() if not passed]
            raise AssertionError(f"Numerical checks failed: {failed}")
        if write_csv:
            if csv_path is None:
                resolved_csv_path = Path(output_base).with_suffix(".csv")
            else:
                resolved_csv_path = Path(csv_path)
            resolved_csv_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(resolved_csv_path, index=False, float_format="%.16e")
    else:
        frame = _load_data(data)

    summary = _prepare_summary(
        frame,
        method_order=method_order,
        method_column=method_column,
        x_column=x_column,
        y_column=y_column,
        center_stat=center_stat,
        spread_stat=spread_stat,
        spread_scale=spread_scale,
        band_clip_lower=band_clip_lower,
    )
    styles = _resolve_method_styles(
        method_order,
        method_styles,
        line_width=line_width,
        marker_size=marker_size,
        plot_kwargs=plot_kwargs,
    )

    if font_family is None:
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        font_family = (
            "Times New Roman" if "Times New Roman" in available_fonts else "DejaVu Serif"
        )
    rc = {
        "font.family": font_family,
        "font.size": font_size,
        "axes.labelsize": axes_label_size,
        "xtick.labelsize": tick_label_size,
        "ytick.labelsize": tick_label_size,
        "legend.fontsize": legend_fontsize,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        **dict(rc_params or {}),
    }

    with mpl.rc_context(rc):
        if ax is None:
            fig_options = {"figsize": figsize, "dpi": dpi, **dict(figure_kwargs or {})}
            fig, axis = plt.subplots(**fig_options)
        else:
            axis = ax
            fig = axis.figure

        label_map = method_labels or {}
        band_overrides = band_styles or {}
        for method in method_order:
            group = summary[summary[method_column].eq(method)].sort_values(x_column)
            x = group[x_column].to_numpy(dtype=float)
            center = group["center"].to_numpy(dtype=float)
            if show_error_band and spread_stat is not None:
                band_style = {
                    "color": styles[method].get("color"),
                    "alpha": band_alpha,
                    "linewidth": 0,
                    "zorder": 1,
                    **dict(fill_between_kwargs or {}),
                    **dict(band_overrides.get(method, {})),
                }
                axis.fill_between(
                    x,
                    group["lower"].to_numpy(dtype=float),
                    group["upper"].to_numpy(dtype=float),
                    **band_style,
                )
            line_style = {
                "label": label_map.get(method, method),
                **styles[method],
            }
            axis.plot(x, center, **line_style)

        if xticks is None:
            xticks = sorted(summary[x_column].unique().astype(float).tolist())
        if xticklabels is None and xticks is not None:
            xticklabels = [f"{value:.3g}" for value in xticks]
        _configure_axis(
            axis,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            axes_label_size=axes_label_size,
            xlabel_size=xlabel_size,
            ylabel_size=ylabel_size,
            title_size=title_size,
            tick_label_size=tick_label_size,
            x_tick_label_size=x_tick_label_size,
            y_tick_label_size=y_tick_label_size,
            xlim=xlim,
            ylim=ylim,
            xticks=xticks,
            xticklabels=xticklabels,
            yticks=yticks,
            yticklabels=yticklabels,
            xscale=xscale,
            yscale=yscale,
            xlabel_kwargs=xlabel_kwargs,
            ylabel_kwargs=ylabel_kwargs,
            title_kwargs=title_kwargs,
            tick_params=tick_params,
            show_grid=show_grid,
            grid_axis=grid_axis,
            grid_which=grid_which,
            grid_kwargs=grid_kwargs,
            hidden_spines=hidden_spines,
            axis_below=axis_below,
            aspect=aspect,
        )
        if show_legend:
            legend_options = {
                "loc": legend_loc,
                "ncol": legend_ncol,
                "fontsize": legend_fontsize,
                "frameon": legend_frameon,
                "handlelength": 2.4,
                "borderaxespad": 0.3,
                **dict(legend_kwargs or {}),
            }
            if legend_bbox_to_anchor is not None:
                legend_options["bbox_to_anchor"] = legend_bbox_to_anchor
            axis.legend(**legend_options)
        if tight_layout:
            fig.tight_layout(**{"pad": 0.25, **dict(tight_layout_kwargs or {})})
        if subplot_adjust:
            fig.subplots_adjust(**dict(subplot_adjust))
        if axes_position is not None:
            axis.set_position(axes_position)
        saved_paths = _save_figure(
            fig,
            output_base=output_base,
            save_formats=save_formats,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            transparent=transparent,
            savefig_kwargs=savefig_kwargs,
        )
        if show:
            plt.show()
        if close:
            plt.close(fig)

    summary.attrs["generated_experiment"] = generated
    summary.attrs["source_data"] = frame
    summary.attrs["diagnostics"] = diagnostics
    summary.attrs["checks"] = checks
    summary.attrs["metrics"] = metrics
    summary.attrs["csv_path"] = resolved_csv_path
    summary.attrs["figure_paths"] = saved_paths
    if generated and print_experiment_report:
        _print_experiment_report(
            frame,
            diagnostics,
            checks,
            metrics,
            resolved_csv_path,
            saved_paths,
        )
    return fig, axis, summary


__all__ = [
    "FLOAT64_EPS",
    "DEFAULT_OUTPUT_BASE",
    "DEFAULT_METHOD_ORDER",
    "DEFAULT_METHOD_STYLES",
    "analytic_optimal_errors",
    "compact_sp_plus",
    "generate_broadcast_reconstruction_data",
    "make_synthetic_target",
    "orthonormal_columns",
    "plot_broadcast_reconstruction",
    "relative_frobenius_error",
    "relative_matrix_difference",
    "rotated_lora_factors",
]
