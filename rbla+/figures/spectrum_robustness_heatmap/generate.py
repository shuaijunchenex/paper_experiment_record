"""Generate spectrum- and rank-scale robustness results for RBLA prefixes.

The Monte Carlo sweep is evaluated in the latent R x R space. For the
orthonormal U,V used by the synthetic construction this gives exactly the same
Frobenius relative error as materialising the ambient matrix U Sigma V^T.
Small end-to-end checks at R=16 and R=160 separately validate that equivalence
and the compact SP+ implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle


R_VALUES = (16, 64, 160)
ALPHAS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
GAUGE_TYPES = ("permutation", "orthogonal")
TRIALS_PER_TYPE = 1_000
SEED = 42
TARGET_SEED = 20260903
TOL = 1e-10
EPS = np.finfo(np.float64).eps
SPECTRUM_LABEL = "normalized exp(-alpha*i/(R-1))"
END_TO_END_CONFIGS = ((128, 96, 16), (256, 192, 160))


def default_output_dir() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "spectrum_robustness_heatmap":
        return cwd
    return cwd / "rbla+" / "figures" / "spectrum_robustness_heatmap"


def orthonormal_columns(
    rng: np.random.Generator, rows: int, columns: int
) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((rows, columns)), mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs[None, :]


def make_spectrum(rank: int, alpha: float) -> np.ndarray:
    spectrum = np.exp(-alpha * np.arange(rank, dtype=np.float64) / (rank - 1))
    return spectrum / np.linalg.norm(spectrum)


def make_gauge(gauge_type: str, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if gauge_type == "permutation":
        return np.eye(rank)[:, rng.permutation(rank)]
    if gauge_type == "orthogonal":
        return orthonormal_columns(rng, rank, rank)
    raise ValueError(gauge_type)


def optimal_errors(spectrum: np.ndarray) -> np.ndarray:
    tail_energy = np.cumsum((spectrum[::-1] ** 2))[::-1]
    errors = np.zeros(spectrum.size, dtype=np.float64)
    errors[:-1] = np.sqrt(tail_energy[1:])
    return errors


def permutation_prefix_errors(spectrum: np.ndarray, gauge: np.ndarray) -> np.ndarray:
    permutation = np.argmax(gauge, axis=0)
    retained_energy = np.cumsum(spectrum[permutation] ** 2)
    errors = np.sqrt(np.maximum(1.0 - retained_energy, 0.0))
    errors[-1] = 0.0
    return errors


def orthogonal_prefix_errors(
    spectrum: np.ndarray, gauge: np.ndarray
) -> tuple[np.ndarray, float]:
    # For P_r = G[:, :r]G[:, :r]^T, evaluate
    # ||Sigma - Sigma^(1/2) P_r Sigma^(1/2)||_F without ambient matrices.
    linear_terms = np.cumsum((gauge * gauge).T @ (spectrum * spectrum))
    weighted_gram = gauge.T @ (spectrum[:, None] * gauge)
    gram_squared = weighted_gram * weighted_gram
    quadratic_terms = np.diag(np.cumsum(np.cumsum(gram_squared, axis=0), axis=1))
    error_squared = 1.0 - 2.0 * linear_terms + quadratic_terms
    raw_full_rank_residual = float(abs(error_squared[-1]))
    errors = np.sqrt(np.maximum(error_squared, 0.0))
    errors[-1] = 0.0
    return errors, raw_full_rank_residual


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
        q_b.shape,
        r_b.shape,
        q_a.shape,
        r_a.shape,
        core.shape,
        u_core.shape,
        vh_core.shape,
        b_plus.shape,
        a_plus.shape,
    ]
    return b_plus, a_plus, shapes


def relative_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    return float(
        np.linalg.norm(reference - approximation, ord="fro")
        / (np.linalg.norm(reference, ord="fro") + EPS)
    )


def end_to_end_checks(trials_per_type: int) -> dict[str, float | bool | int]:
    max_full_update_error = 0.0
    max_sp_optimal_gap = 0.0
    max_gauge_condition_number = 0.0
    compact_path_ok = True
    cases = 0

    for d_out, d_in, rank in END_TO_END_CONFIGS:
        basis_rng = np.random.default_rng(TARGET_SEED + rank)
        u = orthonormal_columns(basis_rng, d_out, rank)
        v = orthonormal_columns(basis_rng, d_in, rank)
        rank_probes = sorted({1, max(1, rank // 4), rank // 2, rank})

        for alpha in (0.5, 4.0, 8.0):
            spectrum = make_spectrum(rank, alpha)
            sqrt_s = np.sqrt(spectrum)
            b_global = u * sqrt_s[None, :]
            a_global = sqrt_s[:, None] * v.T
            delta_w = b_global @ a_global
            optimum = optimal_errors(spectrum)

            for type_index, gauge_type in enumerate(GAUGE_TYPES):
                for trial in sorted({0, trials_per_type - 1}):
                    gauge_seed = SEED + 10_000 * type_index + trial
                    gauge = make_gauge(gauge_type, rank, gauge_seed)
                    b_gauge = b_global @ gauge
                    a_gauge = np.linalg.solve(gauge, a_global)
                    max_gauge_condition_number = max(
                        max_gauge_condition_number, float(np.linalg.cond(gauge, p=2))
                    )
                    max_full_update_error = max(
                        max_full_update_error,
                        relative_error(delta_w, b_gauge @ a_gauge),
                    )
                    b_plus, a_plus, shapes = compact_sp_plus(b_gauge, a_gauge)
                    compact_path_ok &= all(shape != delta_w.shape for shape in shapes)
                    for r in rank_probes:
                        sp_error = relative_error(
                            delta_w, b_plus[:, :r] @ a_plus[:r, :]
                        )
                        max_sp_optimal_gap = max(
                            max_sp_optimal_gap, abs(sp_error - optimum[r - 1])
                        )
                    cases += 1

    return {
        "validation_cases": cases,
        "maximum_full_update_error": max_full_update_error,
        "maximum_sp_plus_optimal_gap": max_sp_optimal_gap,
        "maximum_gauge_condition_number": max_gauge_condition_number,
        "compact_path_no_dense_intermediate": bool(compact_path_ok),
    }


def run_sweep(
    trials_per_type: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, float | bool | int]]:
    summary_records: list[dict[str, float | int | str]] = []
    raw_arrays: dict[str, np.ndarray] = {
        "R_values": np.asarray(R_VALUES, dtype=np.int64),
        "alpha_values": np.asarray(ALPHAS, dtype=np.float64),
        "trial_ids": np.arange(trials_per_type, dtype=np.int64),
        "base_seed": np.asarray(SEED, dtype=np.int64),
        "gauge_seed_offsets": np.asarray((0, 10_000), dtype=np.int64),
    }
    max_orthogonality_error = 0.0
    max_raw_full_rank_residual = 0.0
    minimum_raw_excess = np.inf

    for rank in R_VALUES:
        spectra = np.stack([make_spectrum(rank, alpha) for alpha in ALPHAS])
        optimum = np.stack([optimal_errors(spectrum) for spectrum in spectra])
        raw_arrays[f"optimal_error_R{rank}"] = optimum

        for type_index, gauge_type in enumerate(GAUGE_TYPES):
            excess_trials = np.empty(
                (trials_per_type, len(ALPHAS), rank), dtype=np.float64
            )
            for trial in range(trials_per_type):
                gauge_seed = SEED + 10_000 * type_index + trial
                gauge = make_gauge(gauge_type, rank, gauge_seed)
                if gauge_type == "orthogonal":
                    identity_error = np.max(
                        np.abs(gauge.T @ gauge - np.eye(rank, dtype=np.float64))
                    )
                    max_orthogonality_error = max(
                        max_orthogonality_error, float(identity_error)
                    )

                for alpha_index, spectrum in enumerate(spectra):
                    if gauge_type == "permutation":
                        rbla_error = permutation_prefix_errors(spectrum, gauge)
                    else:
                        rbla_error, full_rank_residual = orthogonal_prefix_errors(
                            spectrum, gauge
                        )
                        max_raw_full_rank_residual = max(
                            max_raw_full_rank_residual, full_rank_residual
                        )
                    raw_excess = rbla_error - optimum[alpha_index]
                    minimum_raw_excess = min(minimum_raw_excess, float(raw_excess.min()))
                    if raw_excess.min() < -TOL:
                        raise AssertionError(
                            f"RBLA beat the dense optimum beyond tolerance: {raw_excess.min()}"
                        )
                    excess_trials[trial, alpha_index] = np.maximum(raw_excess, 0.0)

            raw_arrays[f"excess_{gauge_type}_R{rank}"] = excess_trials
            means = excess_trials.mean(axis=0)
            stds = excess_trials.std(axis=0, ddof=1)
            q25, medians, q75 = np.quantile(
                excess_trials, (0.25, 0.5, 0.75), axis=0
            )
            minima = excess_trials.min(axis=0)
            maxima = excess_trials.max(axis=0)

            for alpha_index, alpha in enumerate(ALPHAS):
                for r in range(1, rank + 1):
                    column = r - 1
                    summary_records.append(
                        {
                            "dataset": "synthetic",
                            "experiment": "latent_rank_scale_sweep",
                            "seed": SEED,
                            "gauge_type": gauge_type,
                            "R": rank,
                            "r": r,
                            "normalized_rank": r / rank,
                            "alpha": alpha,
                            "spectrum": SPECTRUM_LABEL,
                            "trials": trials_per_type,
                            "dense_optimal_error": optimum[alpha_index, column],
                            "mean_excess_error": means[alpha_index, column],
                            "std_excess_error": stds[alpha_index, column],
                            "q25_excess_error": q25[alpha_index, column],
                            "median_excess_error": medians[alpha_index, column],
                            "q75_excess_error": q75[alpha_index, column],
                            "min_excess_error": minima[alpha_index, column],
                            "max_excess_error": maxima[alpha_index, column],
                        }
                    )

    checks: dict[str, float | bool | int] = {
        "trials_per_gauge_type": trials_per_type,
        "raw_trial_cells": int(
            len(GAUGE_TYPES) * len(ALPHAS) * trials_per_type * sum(R_VALUES)
        ),
        "maximum_orthogonality_entrywise_error": max_orthogonality_error,
        "maximum_raw_full_rank_squared_residual": max_raw_full_rank_residual,
        "minimum_raw_excess_error": float(minimum_raw_excess),
    }
    checks.update(end_to_end_checks(trials_per_type))
    checks["all_numerical_checks_passed"] = bool(
        max_orthogonality_error <= TOL
        and max_raw_full_rank_residual <= TOL
        and minimum_raw_excess >= -TOL
        and float(checks["maximum_full_update_error"]) <= TOL
        and float(checks["maximum_sp_plus_optimal_gap"]) <= TOL
        and float(checks["maximum_gauge_condition_number"]) <= 1.0 + TOL
        and bool(checks["compact_path_no_dense_intermediate"])
    )
    return pd.DataFrame.from_records(summary_records), raw_arrays, checks


def alpha_edges() -> np.ndarray:
    centers = np.asarray(ALPHAS, dtype=np.float64)
    interior = (centers[:-1] + centers[1:]) / 2.0
    first = max(0.0, centers[0] - (interior[0] - centers[0]))
    last = centers[-1] + (centers[-1] - interior[-1])
    return np.concatenate(([first], interior, [last]))


def median_matrix(summary: pd.DataFrame, gauge_type: str, rank: int) -> np.ndarray:
    subset = summary[
        summary["gauge_type"].eq(gauge_type) & summary["R"].eq(rank)
    ]
    return (
        subset.pivot(index="alpha", columns="r", values="median_excess_error")
        .reindex(index=ALPHAS, columns=range(1, rank + 1))
        .to_numpy()
    )


def configure_matplotlib() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    paper_font = "Times New Roman" if "Times New Roman" in available_fonts else "DejaVu Serif"
    mpl.rcParams.update(
        {
            "font.family": paper_font,
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.titlesize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def add_alpha_four_box(ax: plt.Axes, x_min: float, x_max: float) -> None:
    edges = alpha_edges()
    row = ALPHAS.index(4.0)
    ax.add_patch(
        Rectangle(
            (x_min, edges[row]),
            x_max - x_min,
            edges[row + 1] - edges[row],
            fill=False,
            edgecolor="white",
            linewidth=0.9,
            linestyle="--",
        )
    )


def plot_normalized_heatmaps(summary: pd.DataFrame, output_dir: Path) -> Path:
    configure_matplotlib()
    matrices = {
        (gauge_type, rank): median_matrix(summary, gauge_type, rank)
        for gauge_type in GAUGE_TYPES
        for rank in R_VALUES
    }
    color_max = max(float(matrix.max()) for matrix in matrices.values())
    y_edges = alpha_edges()
    fig, axes = plt.subplots(
        2, 3, figsize=(7.16, 4.65), sharex=True, sharey=True, constrained_layout=True
    )
    panel_index = 0
    image = None
    for row, gauge_type in enumerate(GAUGE_TYPES):
        for column, rank in enumerate(R_VALUES):
            ax = axes[row, column]
            x_edges = np.arange(rank + 1, dtype=np.float64) / rank
            image = ax.pcolormesh(
                x_edges,
                y_edges,
                matrices[(gauge_type, rank)],
                cmap="viridis",
                vmin=0.0,
                vmax=color_max,
                shading="flat",
                edgecolors="none",
                linewidth=0.0,
                antialiased=False,
                rasterized=True,
            )
            label = chr(ord("a") + panel_index)
            gauge_label = "Permutation" if gauge_type == "permutation" else "Orthogonal"
            ax.set_title(f"({label}) {gauge_label}, $R={rank}$", pad=3)
            ax.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
            ax.set_yticks(ALPHAS)
            ax.set_yticklabels([f"{alpha:g}" for alpha in ALPHAS])
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(y_edges[0], y_edges[-1])
            add_alpha_four_box(ax, 0.0, 1.0)
            ax.tick_params(direction="out", width=0.8, length=3)
            panel_index += 1
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Normalized client rank $r/R$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Spectral decay $\alpha$")
    if image is None:
        raise AssertionError("No heatmap was created")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Median excess error")
    path = output_dir / "spectrum_robustness_heatmap.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_absolute_rank_heatmaps(summary: pd.DataFrame, output_dir: Path) -> Path:
    configure_matplotlib()
    rank = max(R_VALUES)
    matrices = {
        gauge_type: median_matrix(summary, gauge_type, rank)
        for gauge_type in GAUGE_TYPES
    }
    global_color_max = float(summary["median_excess_error"].max())
    y_edges = alpha_edges()
    x_edges = np.arange(rank + 1, dtype=np.float64)
    fig, axes = plt.subplots(
        1, 2, figsize=(7.16, 2.75), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    for panel, (ax, gauge_type) in enumerate(zip(axes, GAUGE_TYPES)):
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            matrices[gauge_type],
            cmap="viridis",
            vmin=0.0,
            vmax=global_color_max,
            shading="flat",
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
            rasterized=True,
        )
        gauge_label = "Permutation" if gauge_type == "permutation" else "Orthogonal"
        ax.set_title(f"({chr(ord('a') + panel)}) {gauge_label}, $R=160$", pad=3)
        ax.set_xticks((4, 32, 64, 96, 128, 160))
        ax.set_yticks(ALPHAS)
        ax.set_yticklabels([f"{alpha:g}" for alpha in ALPHAS])
        ax.set_xlim(0, rank)
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_xlabel(r"Absolute client rank $r$")
        add_alpha_four_box(ax, 0.0, float(rank))
        ax.tick_params(direction="out", width=0.8, length=3)
    axes[0].set_ylabel(r"Spectral decay $\alpha$")
    if image is None:
        raise AssertionError("No heatmap was created")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.025)
    colorbar.set_label("Median excess error")
    path = output_dir / "spectrum_robustness_absolute_rank.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=TRIALS_PER_TYPE)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Running rank-scale sweep: R={R_VALUES}, alphas={ALPHAS}, "
        f"trials/type/configuration={args.trials:,}"
    )

    summary, raw_arrays, checks = run_sweep(args.trials)
    summary_path = output_dir / "spectrum_robustness_heatmap.csv"
    raw_path = output_dir / "spectrum_robustness_trials.npz"
    checks_path = output_dir / "spectrum_robustness_checks.csv"
    summary.to_csv(summary_path, index=False, float_format="%.16e")
    np.savez_compressed(raw_path, **raw_arrays)
    pd.DataFrame(
        {"check": list(checks.keys()), "value": list(checks.values())}
    ).to_csv(checks_path, index=False)
    normalized_pdf = plot_normalized_heatmaps(summary, output_dir)
    absolute_pdf = plot_absolute_rank_heatmaps(summary, output_dir)

    print(f"Saved {len(summary):,} summary cells to {summary_path.name}")
    print(f"Saved {checks['raw_trial_cells']:,} trial-cell values to {raw_path.name}")
    print(f"Saved numerical checks to {checks_path.name}")
    print(f"Saved figures to {normalized_pdf.name} and {absolute_pdf.name}")
    print(f"Maximum median excess error: {summary['median_excess_error'].max():.6f}")
    print(f"All numerical checks passed: {checks['all_numerical_checks_passed']}")
    if not checks["all_numerical_checks_passed"]:
        raise AssertionError(checks)


if __name__ == "__main__":
    main()
