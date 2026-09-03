"""One-call data and plotting API for the combined complexity comparison."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


R = 32
CLIENT_RANK = 32
DIMENSIONS = np.array([128, 256, 512, 1024, 2048, 4096, 8192], dtype=int)
METHODS = (
    "RBLA index prefix",
    "SP+ (compact QR)",
    "Dense truncated SVD",
    "Full dense SVD",
)
RATIO_METHODS = ("SP+ (compact QR)", "Dense truncated SVD", "Full dense SVD")

WORK_EXPRESSIONS = {
    "RBLA index prefix": "1 (constant-time prefix view)",
    "SP+ (compact QR)": "(m+n)R^2 + R^3 + (m+n)Rr",
    "Dense truncated SVD": "mnR + mnr",
    "Full dense SVD": "mnR + mn min(m,n)",
}
WORK_ORDERS = {
    "RBLA index prefix": "O(1)",
    "SP+ (compact QR)": "O(m+n) for fixed R,r",
    "Dense truncated SVD": "O(mn) for fixed R,r",
    "Full dense SVD": "O(mn min(m,n))",
}
SPACE_EXPRESSIONS = {
    "RBLA index prefix": "1 (constant-size prefix view)",
    "SP+ (compact QR)": "2(m+n)R + 5R^2 + R",
    "Dense truncated SVD": "mn + (m+n)r + r",
    "Full dense SVD": "mn + (m+n)k + k",
}
SPACE_ORDERS = {
    "RBLA index prefix": "O(1)",
    "SP+ (compact QR)": "O(m+n) for fixed R",
    "Dense truncated SVD": "O(mn) for fixed r",
    "Full dense SVD": "O(mn + (m+n)min(m,n))",
}
STYLES = {
    "RBLA index prefix": dict(color="#C44E52", marker="o", linestyle="-", zorder=2),
    "SP+ (compact QR)": dict(color="#4C72B0", marker="s", linestyle="-", zorder=3),
    "Dense truncated SVD": dict(
        color="black",
        marker="^",
        markerfacecolor="white",
        markeredgewidth=0.8,
        linestyle="--",
        dashes=(4, 2),
        zorder=4,
    ),
    "Full dense SVD": dict(
        color="#8172B2",
        marker="D",
        markerfacecolor="white",
        markeredgewidth=0.8,
        linestyle="-.",
        zorder=5,
    ),
}
X_TICK_LABELS = ("128", "256", "512", "1k", "2k", "4k", "8k")


def output_dir() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "complexity_comparison":
        return cwd
    return cwd / "rbla+" / "figures" / "complexity_comparison"


def configure_matplotlib() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    paper_font = "Times New Roman" if "Times New Roman" in available_fonts else "DejaVu Serif"
    mpl.rcParams.update(
        {
            "font.family": paper_font,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.2,
            "grid.color": "0.75",
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.55,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def build_work_complexity(
    full_rank: int = R,
    client_rank: int = CLIENT_RANK,
    dimensions: np.ndarray | tuple[int, ...] = DIMENSIONS,
) -> tuple[pd.DataFrame, dict[str, bool], dict[str, float]]:
    records = []
    for dimension in dimensions:
        m = n = int(dimension)
        common_payload = (m + n) * client_rank
        costs = {
            "RBLA index prefix": 1,
            "SP+ (compact QR)": (m + n) * full_rank**2 + full_rank**3 + (m + n) * full_rank * client_rank,
            "Dense truncated SVD": m * n * full_rank + m * n * client_rank,
            "Full dense SVD": m * n * full_rank + m * n * min(m, n),
        }
        for method in METHODS:
            records.append(
                {
                    "matrix_dimension": dimension,
                    "d_out": m,
                    "d_in": n,
                    "R": full_rank,
                    "r": client_rank,
                    "method": method,
                    "work_units": int(costs[method]),
                    "common_payload_scalars_excluded": int(common_payload),
                    "materializes_dense_update": method
                    in {"Dense truncated SVD", "Full dense SVD"},
                    "complexity_expression": WORK_EXPRESSIONS[method],
                    "asymptotic_order": WORK_ORDERS[method],
                    "complexity_type": "rank_fixed_server_time_proxy",
                    "dense_svd_model": (
                        "full SVD after dense materialization"
                        if method == "Full dense SVD"
                        else "rank-r truncated SVD after dense materialization"
                        if method == "Dense truncated SVD"
                        else "not applicable"
                    ),
                }
            )

    frame = pd.DataFrame.from_records(records)
    wide = frame.pivot(index="matrix_dimension", columns="method", values="work_units")
    sp_by_dimension = wide["SP+ (compact QR)"].to_dict()
    frame["relative_to_sp_plus"] = [
        row.work_units / sp_by_dimension[row.matrix_dimension]
        for row in frame.itertuples(index=False)
    ]
    slopes = {
        method: fitted_log_slope(wide, method)
        for method in METHODS
    }
    checks = {
        "RBLA prefix-view cost is constant": bool((wide["RBLA index prefix"] == 1).all()),
        "compact SP+ cost increases with dimension": bool(
            (np.diff(wide["SP+ (compact QR)"]) > 0).all()
        ),
        "dense truncated-SVD cost increases with dimension": bool(
            (np.diff(wide["Dense truncated SVD"]) > 0).all()
        ),
        "large-d SP+ log-log slope is approximately one": 0.95
        <= slopes["SP+ (compact QR)"]
        <= 1.05,
        "dense log-log slope is two": abs(slopes["Dense truncated SVD"] - 2.0)
        <= 1e-12,
        "large-d full-SVD log-log slope is approximately three": 2.95
        <= slopes["Full dense SVD"]
        <= 3.05,
        "dense reference is always more expensive than compact SP+": bool(
            (wide["Dense truncated SVD"] > wide["SP+ (compact QR)"]).all()
        ),
        "full SVD is always more expensive than truncated SVD": bool(
            (wide["Full dense SVD"] > wide["Dense truncated SVD"]).all()
        ),
    }
    frame["analytical_checks_passed"] = all(checks.values())
    return frame, checks, slopes


def build_space_complexity(
    full_rank: int = R,
    client_rank: int = CLIENT_RANK,
    dimensions: np.ndarray | tuple[int, ...] = DIMENSIONS,
) -> tuple[pd.DataFrame, dict[str, bool], dict[str, float]]:
    records = []
    for dimension in dimensions:
        m = n = int(dimension)
        k = min(m, n)
        common_payload = (m + n) * client_rank
        storage = {
            "RBLA index prefix": 1,
            "SP+ (compact QR)": 2 * (m + n) * full_rank + 5 * full_rank**2 + full_rank,
            "Dense truncated SVD": m * n + (m + n) * client_rank + client_rank,
            "Full dense SVD": m * n + (m + n) * k + k,
        }
        for method in METHODS:
            records.append(
                {
                    "matrix_dimension": dimension,
                    "d_out": m,
                    "d_in": n,
                    "R": full_rank,
                    "r": client_rank,
                    "method": method,
                    "auxiliary_scalar_elements": int(storage[method]),
                    "common_payload_scalars_excluded": int(common_payload),
                    "space_expression": SPACE_EXPRESSIONS[method],
                    "asymptotic_space_order": SPACE_ORDERS[method],
                    "space_scope": "algorithm-specific auxiliary tensors",
                }
            )

    frame = pd.DataFrame.from_records(records)
    wide = frame.pivot(
        index="matrix_dimension", columns="method", values="auxiliary_scalar_elements"
    )
    sp_by_dimension = wide["SP+ (compact QR)"].to_dict()
    frame["relative_to_sp_plus"] = [
        row.auxiliary_scalar_elements / sp_by_dimension[row.matrix_dimension]
        for row in frame.itertuples(index=False)
    ]
    slopes = {
        method: fitted_log_slope(wide, method)
        for method in METHODS
    }
    checks = {
        "RBLA auxiliary space is constant": bool((wide["RBLA index prefix"] == 1).all()),
        "large-d SP+ space slope is approximately one": 0.95
        <= slopes["SP+ (compact QR)"]
        <= 1.05,
        "large-d truncated-SVD space slope is approximately two": 1.95
        <= slopes["Dense truncated SVD"]
        <= 2.05,
        "large-d full-SVD space slope is approximately two": 1.95
        <= slopes["Full dense SVD"]
        <= 2.05,
        "full SVD storage exceeds truncated-SVD storage": bool(
            (wide["Full dense SVD"] > wide["Dense truncated SVD"]).all()
        ),
    }
    frame["analytical_checks_passed"] = all(checks.values())
    return frame, checks, slopes


def fitted_log_slope(wide: pd.DataFrame, method: str, tail: int = 4) -> float:
    x = wide.index.to_numpy(dtype=float)[-tail:]
    y = wide[method].to_numpy(dtype=float)[-tail:]
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlabel(r"Square matrix dimension $d$ ($m=n=d$)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(DIMENSIONS[0] / 1.12, DIMENSIONS[-1] * 1.16)
    ax.set_xticks(DIMENSIONS)
    ax.set_xticklabels(X_TICK_LABELS)
    ax.grid(True, which="major")
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.8, length=3)


def plot_absolute_units(
    work: pd.DataFrame, space: pd.DataFrame, path: Path
) -> None:
    fig, (ax_work, ax_space) = plt.subplots(1, 2, figsize=(7.16, 2.65))
    for method in METHODS:
        work_group = work[work["method"].eq(method)].sort_values("matrix_dimension")
        space_group = space[space["method"].eq(method)].sort_values("matrix_dimension")
        ax_work.plot(
            work_group["matrix_dimension"], work_group["work_units"],
            label=method, **STYLES[method]
        )
        ax_space.plot(
            space_group["matrix_dimension"], space_group["auxiliary_scalar_elements"],
            label=method, **STYLES[method]
        )

    for ax in (ax_work, ax_space):
        style_axis(ax)
    ax_work.set_ylabel("Rank-fixed server work (units)")
    ax_space.set_ylabel("Auxiliary storage (scalar elements)")
    ax_work.set_ylim(0.5, 1e13)
    ax_space.set_ylim(0.5, 1e10)
    ax_work.set_title("(a) Server work", loc="left", pad=3)
    ax_space.set_title("(b) Auxiliary storage", loc="left", pad=3)
    handles, labels = ax_work.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        handlelength=2.2,
        columnspacing=1.25,
    )
    fig.text(
        0.5,
        0.01,
        r"Fixed $R=r=32$; analytical units; common factor inputs and transmitted payload excluded",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="0.3",
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.25, top=0.78, wspace=0.33)
    fig.savefig(path)
    plt.close(fig)


def annotate_endpoint(ax: plt.Axes, group: pd.DataFrame, method: str) -> None:
    endpoint = group.iloc[-1]
    value = float(endpoint.relative_to_sp_plus)
    label = "1×" if np.isclose(value, 1.0) else f"{value:,.0f}×"
    vertical_offset = {
        "SP+ (compact QR)": -8,
        "Dense truncated SVD": 3,
        "Full dense SVD": 3,
    }[method]
    ax.annotate(
        label,
        (endpoint.matrix_dimension, value),
        xytext=(-6, vertical_offset),
        textcoords="offset points",
        ha="right",
        va="bottom" if vertical_offset >= 0 else "top",
        fontsize=6.5,
        color=STYLES[method]["color"],
    )


def plot_relative_multipliers(
    work: pd.DataFrame, space: pd.DataFrame, path: Path
) -> None:
    fig, (ax_work, ax_space) = plt.subplots(1, 2, figsize=(7.16, 2.65))
    for method in RATIO_METHODS:
        work_group = work[work["method"].eq(method)].sort_values("matrix_dimension")
        space_group = space[space["method"].eq(method)].sort_values("matrix_dimension")
        ax_work.plot(
            work_group["matrix_dimension"], work_group["relative_to_sp_plus"],
            label=method, **STYLES[method]
        )
        ax_space.plot(
            space_group["matrix_dimension"], space_group["relative_to_sp_plus"],
            label=method, **STYLES[method]
        )
        annotate_endpoint(ax_work, work_group, method)
        annotate_endpoint(ax_space, space_group, method)

    for ax in (ax_work, ax_space):
        style_axis(ax)
        ax.axhline(1.0, color="0.45", linewidth=0.7, linestyle=":", zorder=1)
    ax_work.set_ylabel("Relative server work vs. SP+")
    ax_space.set_ylabel("Relative auxiliary storage vs. SP+")
    ax_work.set_ylim(0.7, 1e5)
    ax_space.set_ylim(0.7, 1e3)
    ax_work.set_title("(a) Server work multiplier", loc="left", pad=3)
    ax_space.set_title("(b) Auxiliary storage multiplier", loc="left", pad=3)
    handles, labels = ax_work.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        handlelength=2.2,
        columnspacing=1.5,
    )
    fig.text(
        0.5,
        0.01,
        r"SP+ is the 1× baseline; fixed $R=r=32$; common factor inputs and transmitted payload excluded",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="0.3",
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.25, top=0.78, wspace=0.33)
    fig.savefig(path)
    plt.close(fig)


def _save_formats(
    fig: plt.Figure,
    output_base: str | Path | None,
    formats: Sequence[str],
    dpi: int,
    savefig_kwargs: Mapping[str, Any] | None,
) -> list[Path]:
    if output_base is None:
        return []
    base = Path(output_base).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in formats:
        path = base.with_suffix("." + suffix.lstrip("."))
        options = {
            "bbox_inches": "tight",
            "pad_inches": 0.02,
            **dict(savefig_kwargs or {}),
        }
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            options.setdefault("dpi", dpi)
        fig.savefig(path, **options)
        saved.append(path)
    return saved


def plot_complexity_comparison(
    work_data: pd.DataFrame | str | Path | None = None,
    space_data: pd.DataFrame | str | Path | None = None,
    output_directory: str | Path | None = None,
    *,
    full_rank: int = R,
    client_rank: int = CLIENT_RANK,
    dimensions: Sequence[int] = tuple(DIMENSIONS.tolist()),
    write_csv: bool = True,
    assert_checks: bool = True,
    print_report: bool = True,
    absolute_output_name: str = "broadcast_complexity",
    relative_output_name: str = "broadcast_space_complexity",
    figsize: tuple[float, float] = (7.16, 2.65),
    dpi: int = 600,
    font_family: str | None = None,
    font_size: float = 8,
    axes_label_size: float = 8,
    tick_label_size: float = 7,
    title_size: float = 8,
    legend_fontsize: float = 7,
    method_styles: Mapping[str, Mapping[str, Any]] | None = None,
    line_width: float = 1.45,
    marker_size: float = 4.2,
    xlabel: str = r"Square matrix dimension $d$ ($m=n=d$)",
    xscale: str = "log",
    xscale_base: float = 2,
    yscale: str = "log",
    xlim: tuple[float, float] | None = None,
    xticks: Sequence[float] | None = None,
    xticklabels: Sequence[str] | None = None,
    absolute_work_ylabel: str = "Rank-fixed server work (units)",
    absolute_space_ylabel: str = "Auxiliary storage (scalar elements)",
    absolute_work_title: str = "(a) Server work",
    absolute_space_title: str = "(b) Auxiliary storage",
    absolute_work_ylim: tuple[float, float] = (0.5, 1e13),
    absolute_space_ylim: tuple[float, float] = (0.5, 1e10),
    relative_work_ylabel: str = "Relative server work vs. SP+",
    relative_space_ylabel: str = "Relative auxiliary storage vs. SP+",
    relative_work_title: str = "(a) Server work multiplier",
    relative_space_title: str = "(b) Auxiliary storage multiplier",
    relative_work_ylim: tuple[float, float] = (0.7, 1e5),
    relative_space_ylim: tuple[float, float] = (0.7, 1e3),
    title_loc: str = "left",
    title_pad: float = 3,
    show_grid: bool = True,
    grid_kwargs: Mapping[str, Any] | None = None,
    hidden_spines: Sequence[str] = ("top", "right"),
    absolute_legend_loc: str = "upper center",
    absolute_legend_bbox_to_anchor: tuple[float, float] | None = (0.5, 1.01),
    absolute_legend_ncol: int = 4,
    relative_legend_loc: str = "upper center",
    relative_legend_bbox_to_anchor: tuple[float, float] | None = (0.5, 1.01),
    relative_legend_ncol: int = 3,
    legend_kwargs: Mapping[str, Any] | None = None,
    show_endpoint_annotations: bool = True,
    annotation_fontsize: float = 6.5,
    annotation_kwargs: Mapping[str, Any] | None = None,
    show_footnotes: bool = True,
    absolute_footnote: str | None = None,
    relative_footnote: str | None = None,
    footnote_y: float = 0.01,
    footnote_fontsize: float = 6.5,
    footnote_kwargs: Mapping[str, Any] | None = None,
    figure_kwargs: Mapping[str, Any] | None = None,
    rc_params: Mapping[str, Any] | None = None,
    plot_kwargs: Mapping[str, Any] | None = None,
    absolute_subplot_adjust: Mapping[str, float] | None = None,
    relative_subplot_adjust: Mapping[str, float] | None = None,
    save_formats: Sequence[str] = ("pdf",),
    savefig_kwargs: Mapping[str, Any] | None = None,
    show: bool = False,
    close: bool = False,
) -> tuple[
    tuple[plt.Figure, plt.Figure],
    tuple[tuple[plt.Axes, plt.Axes], tuple[plt.Axes, plt.Axes]],
    dict[str, Any],
]:
    """Build both combined figures through one fully adjustable call."""
    resolved_dimensions = np.asarray(dimensions, dtype=int)
    if resolved_dimensions.ndim != 1 or resolved_dimensions.size < 4:
        raise ValueError("dimensions must contain at least four values")
    if np.any(resolved_dimensions <= 0) or np.any(np.diff(resolved_dimensions) <= 0):
        raise ValueError("dimensions must be positive and strictly increasing")
    generated = work_data is None and space_data is None
    if (work_data is None) != (space_data is None):
        raise ValueError("work_data and space_data must either both be supplied or both omitted")
    if generated:
        work, work_checks, work_slopes = build_work_complexity(
            full_rank, client_rank, resolved_dimensions
        )
        space, space_checks, space_slopes = build_space_complexity(
            full_rank, client_rank, resolved_dimensions
        )
    else:
        work = work_data.copy() if isinstance(work_data, pd.DataFrame) else pd.read_csv(work_data)
        space = space_data.copy() if isinstance(space_data, pd.DataFrame) else pd.read_csv(space_data)
        work_checks = space_checks = None
        work_slopes = space_slopes = None
        resolved_dimensions = np.sort(work["matrix_dimension"].unique().astype(int))
    if assert_checks and generated:
        failed = [name for name, passed in {**work_checks, **space_checks}.items() if not passed]
        if failed:
            raise AssertionError(failed)

    target = output_dir() if output_directory is None else Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    csv_paths: list[Path] = []
    if write_csv:
        work_path = target / "broadcast_complexity.csv"
        space_path = target / "broadcast_space_complexity.csv"
        work.to_csv(work_path, index=False)
        space.to_csv(space_path, index=False)
        csv_paths = [work_path, space_path]

    styles = deepcopy(STYLES)
    for name, values in (method_styles or {}).items():
        styles.setdefault(name, {}).update(values)
    if font_family is None:
        fonts = {font.name for font in font_manager.fontManager.ttflist}
        font_family = "Times New Roman" if "Times New Roman" in fonts else "DejaVu Serif"
    rc = {
        "font.family": font_family, "font.size": font_size,
        "axes.labelsize": axes_label_size, "axes.titlesize": title_size,
        "xtick.labelsize": tick_label_size, "ytick.labelsize": tick_label_size,
        "legend.fontsize": legend_fontsize, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, **dict(rc_params or {}),
    }
    labels = list(xticklabels) if xticklabels is not None else [
        f"{value // 1024}k" if value >= 1024 else str(value)
        for value in resolved_dimensions
    ]
    ticks = list(xticks) if xticks is not None else resolved_dimensions.tolist()
    limits = xlim or (resolved_dimensions[0] / 1.12, resolved_dimensions[-1] * 1.16)
    absolute_footnote = absolute_footnote or (
        rf"Fixed $R={full_rank}, r={client_rank}$; analytical units; common factor "
        "inputs and transmitted payload excluded"
    )
    relative_footnote = relative_footnote or (
        rf"SP+ is the 1× baseline; fixed $R={full_rank}, r={client_rank}$; common "
        "factor inputs and transmitted payload excluded"
    )

    def style_axis_local(axis: plt.Axes) -> None:
        axis.set_xlabel(xlabel)
        if xscale == "log":
            axis.set_xscale(xscale, base=xscale_base)
        else:
            axis.set_xscale(xscale)
        axis.set_yscale(yscale)
        axis.set_xlim(*limits)
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels)
        axis.grid(show_grid, which="major", **{
            "color": "0.75", "linestyle": ":", "linewidth": 0.5,
            "alpha": 0.55, **dict(grid_kwargs or {}),
        })
        axis.grid(False, which="minor")
        axis.set_axisbelow(True)
        for spine in hidden_spines:
            axis.spines[spine].set_visible(False)
        axis.tick_params(direction="out", width=0.8, length=3)

    with mpl.rc_context(rc):
        fig_abs, (ax_aw, ax_as) = plt.subplots(
            1, 2, **{"figsize": figsize, "dpi": dpi, **dict(figure_kwargs or {})}
        )
        for method in METHODS:
            for axis, frame, y_column in (
                (ax_aw, work, "work_units"),
                (ax_as, space, "auxiliary_scalar_elements"),
            ):
                group = frame[frame["method"].eq(method)].sort_values("matrix_dimension")
                axis.plot(group["matrix_dimension"], group[y_column], label=method,
                          **{"linewidth": line_width, "markersize": marker_size,
                             **styles[method], **dict(plot_kwargs or {})})
        for axis in (ax_aw, ax_as):
            style_axis_local(axis)
        ax_aw.set_ylabel(absolute_work_ylabel); ax_as.set_ylabel(absolute_space_ylabel)
        ax_aw.set_ylim(*absolute_work_ylim); ax_as.set_ylim(*absolute_space_ylim)
        ax_aw.set_title(absolute_work_title, loc=title_loc, pad=title_pad)
        ax_as.set_title(absolute_space_title, loc=title_loc, pad=title_pad)
        handles, legend_labels = ax_aw.get_legend_handles_labels()
        abs_legend = {"handles": handles, "labels": legend_labels,
                      "loc": absolute_legend_loc, "ncol": absolute_legend_ncol,
                      "fontsize": legend_fontsize, "handlelength": 2.2,
                      "columnspacing": 1.25, **dict(legend_kwargs or {})}
        if absolute_legend_bbox_to_anchor is not None:
            abs_legend["bbox_to_anchor"] = absolute_legend_bbox_to_anchor
        fig_abs.legend(**abs_legend)
        if show_footnotes:
            fig_abs.text(0.5, footnote_y, absolute_footnote, ha="center", va="bottom",
                         fontsize=footnote_fontsize, color="0.3", **dict(footnote_kwargs or {}))
        fig_abs.subplots_adjust(**{"left": .09, "right": .995, "bottom": .25,
                                   "top": .78, "wspace": .33,
                                   **dict(absolute_subplot_adjust or {})})

        fig_rel, (ax_rw, ax_rs) = plt.subplots(
            1, 2, **{"figsize": figsize, "dpi": dpi, **dict(figure_kwargs or {})}
        )
        for method in RATIO_METHODS:
            for axis, frame in ((ax_rw, work), (ax_rs, space)):
                group = frame[frame["method"].eq(method)].sort_values("matrix_dimension")
                axis.plot(group["matrix_dimension"], group["relative_to_sp_plus"],
                          label=method, **{"linewidth": line_width,
                          "markersize": marker_size, **styles[method], **dict(plot_kwargs or {})})
                if show_endpoint_annotations:
                    endpoint = group.iloc[-1]
                    value = float(endpoint["relative_to_sp_plus"])
                    text = "1×" if np.isclose(value, 1) else f"{value:,.0f}×"
                    offset = -8 if method == "SP+ (compact QR)" else 3
                    axis.annotate(text, (endpoint["matrix_dimension"], value),
                        xytext=(-6, offset), textcoords="offset points", ha="right",
                        va="bottom" if offset >= 0 else "top", fontsize=annotation_fontsize,
                        color=styles[method]["color"], **dict(annotation_kwargs or {}))
        for axis in (ax_rw, ax_rs):
            style_axis_local(axis)
            axis.axhline(1, color="0.45", linewidth=.7, linestyle=":", zorder=1)
        ax_rw.set_ylabel(relative_work_ylabel); ax_rs.set_ylabel(relative_space_ylabel)
        ax_rw.set_ylim(*relative_work_ylim); ax_rs.set_ylim(*relative_space_ylim)
        ax_rw.set_title(relative_work_title, loc=title_loc, pad=title_pad)
        ax_rs.set_title(relative_space_title, loc=title_loc, pad=title_pad)
        handles, legend_labels = ax_rw.get_legend_handles_labels()
        rel_legend = {"handles": handles, "labels": legend_labels,
                      "loc": relative_legend_loc, "ncol": relative_legend_ncol,
                      "fontsize": legend_fontsize, "handlelength": 2.2,
                      "columnspacing": 1.5, **dict(legend_kwargs or {})}
        if relative_legend_bbox_to_anchor is not None:
            rel_legend["bbox_to_anchor"] = relative_legend_bbox_to_anchor
        fig_rel.legend(**rel_legend)
        if show_footnotes:
            fig_rel.text(0.5, footnote_y, relative_footnote, ha="center", va="bottom",
                         fontsize=footnote_fontsize, color="0.3", **dict(footnote_kwargs or {}))
        fig_rel.subplots_adjust(**{"left": .09, "right": .995, "bottom": .25,
                                   "top": .78, "wspace": .33,
                                   **dict(relative_subplot_adjust or {})})
        abs_paths = _save_formats(fig_abs, target / absolute_output_name, save_formats, dpi, savefig_kwargs)
        rel_paths = _save_formats(fig_rel, target / relative_output_name, save_formats, dpi, savefig_kwargs)
        if show:
            plt.show()
        if close:
            plt.close(fig_abs); plt.close(fig_rel)

    results = {
        "work": work, "space": space, "work_checks": work_checks,
        "space_checks": space_checks, "work_slopes": work_slopes,
        "space_slopes": space_slopes, "csv_paths": csv_paths,
        "absolute_figure_paths": abs_paths, "relative_figure_paths": rel_paths,
    }
    if print_report:
        checks_passed = True if not generated else all(work_checks.values()) and all(space_checks.values())
        print(f"Combined complexity comparison; analytical checks passed: {checks_passed}")
        for path in csv_paths + abs_paths + rel_paths:
            print(f"  - {path}")
    return (fig_abs, fig_rel), ((ax_aw, ax_as), (ax_rw, ax_rs)), results


def main() -> None:
    target = output_dir()
    target.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    work, work_checks, work_slopes = build_work_complexity()
    space, space_checks, space_slopes = build_space_complexity()
    if not all(work_checks.values()):
        raise AssertionError([name for name, passed in work_checks.items() if not passed])
    if not all(space_checks.values()):
        raise AssertionError([name for name, passed in space_checks.items() if not passed])

    work.to_csv(target / "broadcast_complexity.csv", index=False)
    space.to_csv(target / "broadcast_space_complexity.csv", index=False)
    plot_absolute_units(work, space, target / "broadcast_complexity.pdf")
    plot_relative_multipliers(work, space, target / "broadcast_space_complexity.pdf")

    work_end = work[work["matrix_dimension"].eq(DIMENSIONS[-1])].set_index("method")
    space_end = space[space["matrix_dimension"].eq(DIMENSIONS[-1])].set_index("method")
    print("Combined complexity comparison")
    print(f"  fixed ranks: R={R}, r={CLIENT_RANK}")
    print(f"  square dimensions: {DIMENSIONS.tolist()}")
    print(f"  work slopes: {work_slopes}")
    print(f"  space slopes: {space_slopes}")
    print(
        "  d=8192 work multipliers: "
        f"dense={work_end.loc['Dense truncated SVD', 'relative_to_sp_plus']:.2f}x, "
        f"full={work_end.loc['Full dense SVD', 'relative_to_sp_plus']:.2f}x"
    )
    print(
        "  d=8192 storage multipliers: "
        f"dense={space_end.loc['Dense truncated SVD', 'relative_to_sp_plus']:.2f}x, "
        f"full={space_end.loc['Full dense SVD', 'relative_to_sp_plus']:.2f}x"
    )
    print("  all analytical checks passed: True")
    print("  broadcast_complexity.pdf: absolute work and storage units")
    print("  broadcast_space_complexity.pdf: work and storage multipliers vs. SP+")


if __name__ == "__main__":
    main()
