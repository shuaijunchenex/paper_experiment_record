"""Final-checkpoint reconstruction using the structural middle LoRA layer."""
from pathlib import Path
import numpy as np
import pandas as pd
from plot_style import (plt, RC, RED, BLUE, DATASETS, verify_sources,
                        make_panels, above_legend, save_figure)

ROOT = Path(__file__).resolve().parent
STEM = "checkpoint_reconstruction"


def main():
    provenance = verify_sources(ROOT)
    data = pd.read_csv(ROOT / "sources/reconstruction.csv", float_precision="round_trip")
    selected, selections = [], []
    with plt.rc_context(RC):
        fig, axes = make_panels()
        for ax, ds in zip(axes, DATASETS):
            subset = data[(data.dataset == ds) & (data.driver_method == "rbla")
                          & (data["round"] == 100)].copy()
            layers = subset[["layer_index", "layer"]].drop_duplicates().sort_values("layer_index")
            layer = layers.iloc[len(layers) // 2]["layer"]
            rows = subset[subset.layer == layer].copy()
            if set(rows.method) != {"raw_prefix", "compact", "dense_svd"}:
                raise ValueError("Incomplete method coverage")
            if not (rows.compute_dtype == "float64").all() or not (rows.input_kind == "training_checkpoint").all():
                raise ValueError("Expected real FP64 reconstruction records")
            if not rows[rows.method == "compact"].optimal_error_check_pass.all():
                raise ValueError("Compact optimality check failed")
            rows["normalized_width"] = rows.target_width / rows.aggregate_width
            selected.append(rows)
            selections.append({"dataset": ds, "layer": layer,
                               "selection_rule": "floor(number_of_layers / 2) in layer_index order",
                               "layer_count": len(layers), "round": 100, "update_index": 101,
                               "matrix_shape": [int(rows.iloc[0].matrix_out), int(rows.iloc[0].matrix_in)],
                               "aggregate_width": int(rows.iloc[0].aggregate_width),
                               "target_widths": sorted(int(v) for v in rows.target_width.unique())})
            for method, label, color, marker, ls, width in [
                ("raw_prefix", "RBLA", RED, "o", "-", 1.0),
                ("compact", "SP+", BLUE, "s", "-", 1.15),
                ("dense_svd", "Dense SVD", "black", None, (0, (4, 3)), .8),
            ]:
                curve = rows[rows.method == method].sort_values("target_width")
                ax.plot(curve.normalized_width, curve.relative_error, color=color,
                        marker=marker, ls=ls, lw=width, ms=2.4,
                        mfc="white" if method == "compact" else color, mew=.5,
                        label=label, zorder=5 if method == "dense_svd" else 3)
            ax.set(xlim=(0, 1.03), ylim=(-.035, 1.05), xticks=[0, .5, 1.],
                   yticks=[0, .5, 1.],
                   xlabel=r"Width $q/Q$", ylabel="Relative error" if ds == DATASETS[0] else "")
            if ds != DATASETS[0]:
                ax.tick_params(labelleft=False)
            above_legend(ax)
        save_figure(ROOT, fig, STEM, provenance=provenance, metadata={
            "title": "Reconstruction of real pre-spectral RBLA aggregates",
            "selections": selections,
            "metric": "||X - X_q||_F / ||X||_F, X = B A",
            "dtype": "FP64 factors and metric evaluation",
            "uncertainty_band": None, "gauge_transformation": None,
            "dense_reference": "economy full dense SVD followed by truncation",
            "internal_width_note": "CNN q is factor width, not logical convolution rank; q = 3r for the selected conv2b layer",
        })
    result = pd.concat(selected, ignore_index=True)
    result.to_csv(ROOT / "plot_data.csv", index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
