"""Prefix excess error across real training checkpoints; no seed inference."""
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from plot_style import (plt, RC, RED, BLUE, DATASETS, verify_sources,
                        make_panels, above_legend, save_figure)

ROOT = Path(__file__).resolve().parent
STEM = "prefix_reconstruction_quality"


def main():
    provenance = verify_sources(ROOT)
    data = pd.read_csv(ROOT / "sources/reconstruction.csv", float_precision="round_trip")
    selected = data[data.dataset.isin(DATASETS) & (data.method == "raw_prefix") & (~data.is_full_width)
                    & (data["round"].isin([9, 49, 100]))].copy()
    if not (selected.input_kind == "training_checkpoint").all():
        raise ValueError("Synthetic records are not allowed")
    if selected.zero_reference.any() or not np.isfinite(selected.normalized_excess_error).all():
        raise ValueError("Undefined relative error")
    if (selected.normalized_excess_error < -1e-12).any():
        raise ValueError("Negative excess exceeds numerical tolerance")
    # Quantiles use the signed recorded values. Only tolerance-sized negative
    # summary values are placed at zero for drawing; the CSV remains unmodified.
    summaries = []
    for (ds, driver, logged_round), group in selected.groupby(["dataset", "driver_method", "round"], sort=True):
        q25, median, q75 = group.normalized_excess_error.quantile([.25, .5, .75])
        summaries.append({"dataset": ds, "driver_method": driver, "round": int(logged_round),
                          "update_index": int(logged_round + 1), "q25": q25,
                          "median": median, "q75": q75, "n_layer_width_pairs": len(group),
                          "layer_count": group.layer.nunique(),
                          "nonpositive_raw_count": int((group.normalized_excess_error <= 0).sum())})
    summary = pd.DataFrame(summaries)
    if len(summary) != 12:
        raise ValueError("Expected 2 datasets x 2 paths x 3 checkpoints")
    with plt.rc_context(RC):
        fig, axes = make_panels()
        for ax, ds in zip(axes, DATASETS):
            for driver, label, color, marker in [
                ("rbla", "RBLA training", RED, "o"),
                ("rbla_plus_rank_scale", "RBLA+ training", BLUE, "s"),
            ]:
                group = summary[(summary.dataset == ds) & (summary.driver_method == driver)].sort_values("round")
                low, center, high = (np.maximum(group[col].to_numpy(), 0.) for col in ["q25", "median", "q75"])
                ax.errorbar(group["round"], center, yerr=np.array([center-low, high-center]),
                            color=color, marker=marker, mfc="white" if driver != "rbla" else color,
                            mew=.6, ms=2.6, lw=1., elinewidth=.65, capsize=2,
                            label=label, zorder=3, clip_on=False)
            ax.set_yscale("symlog", linthresh=1e-5, linscale=.7)
            ax.set(xlim=(0, 100), ylim=(-2e-6, 1.5), xticks=[0, 50, 100],
                   xlabel="Round", ylabel="Normalized excess error" if ds == DATASETS[0] else "")
            if ds != DATASETS[0]:
                ax.tick_params(labelleft=False)
            ax.yaxis.set_major_locator(FixedLocator([0, 1e-4, 1e-2, 1]))
            ax.yaxis.set_major_formatter(FixedFormatter(["0", r"$10^{-4}$", r"$10^{-2}$", r"$10^0$"]))
            ax.yaxis.set_minor_locator(NullLocator())
            above_legend(ax, anchor=1.31)
        save_figure(ROOT, fig, STEM, provenance=provenance, metadata={
            "title": "Training path and raw-prefix reconstruction quality",
            "metric": "(||X-X_prefix,q||_F^2 - ||X-X_dense,q||_F^2) / ||X||_F^2",
            "axis_label_note": "Excess uses squared Frobenius errors; the full definition is in caption.txt and figure_ieee.tex",
            "raw_pairs": len(selected), "groups": len(summary),
            "center": "median", "error_bars": "25th to 75th percentiles across layer/unique-target-width pairs within each checkpoint",
            "not_confidence_intervals": True, "single_training_seed": 42,
            "weighting": "one observation per layer and distinct non-full target width; no duplication by client requests",
            "full_width_excluded": True, "all_layers_included": True,
            "yaxis": {"scale": "symlog", "linthresh": 1e-5, "linscale": .7},
            "display_only_adjustment": "summary values in [-1e-12,0) placed at zero; raw values and signed quantiles retained",
            "xaxis": "logged round; measured checkpoints 9, 49, 100",
            "xlim": [0, 100], "xticks": [0, 50, 100],
            "interpretation": "separate training trajectories have different aggregate reference matrices",
        })
    summary.to_csv(ROOT / "plot_summary.csv", index=False, float_format="%.17g")
    selected.to_csv(ROOT / "plot_data.csv", index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
