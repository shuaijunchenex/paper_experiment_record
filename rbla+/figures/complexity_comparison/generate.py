"""Generate combined absolute-unit and relative-multiplier complexity figures."""

from __future__ import annotations

from pathlib import Path

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


def build_work_complexity() -> tuple[pd.DataFrame, dict[str, bool], dict[str, float]]:
    records = []
    for dimension in DIMENSIONS:
        m = n = int(dimension)
        common_payload = (m + n) * CLIENT_RANK
        costs = {
            "RBLA index prefix": 1,
            "SP+ (compact QR)": (m + n) * R**2 + R**3 + (m + n) * R * CLIENT_RANK,
            "Dense truncated SVD": m * n * R + m * n * CLIENT_RANK,
            "Full dense SVD": m * n * R + m * n * min(m, n),
        }
        for method in METHODS:
            records.append(
                {
                    "matrix_dimension": dimension,
                    "d_out": m,
                    "d_in": n,
                    "R": R,
                    "r": CLIENT_RANK,
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


def build_space_complexity() -> tuple[pd.DataFrame, dict[str, bool], dict[str, float]]:
    records = []
    for dimension in DIMENSIONS:
        m = n = int(dimension)
        k = min(m, n)
        common_payload = (m + n) * CLIENT_RANK
        storage = {
            "RBLA index prefix": 1,
            "SP+ (compact QR)": 2 * (m + n) * R + 5 * R**2 + R,
            "Dense truncated SVD": m * n + (m + n) * CLIENT_RANK + CLIENT_RANK,
            "Full dense SVD": m * n + (m + n) * k + k,
        }
        for method in METHODS:
            records.append(
                {
                    "matrix_dimension": dimension,
                    "d_out": m,
                    "d_in": n,
                    "R": R,
                    "r": CLIENT_RANK,
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
