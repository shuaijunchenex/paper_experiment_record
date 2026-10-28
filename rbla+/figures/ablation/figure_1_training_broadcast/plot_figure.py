"""Four broadcast-accuracy curves per dataset, from frozen native CSV logs."""
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
from plot_style import (plt, RC, RED, BLUE, DATASETS, verify_sources,
                        make_panels, above_legend, save_figure)

ROOT = Path(__file__).resolve().parent
STEM = "training_broadcast_accuracy"


def main():
    provenance = verify_sources(ROOT)
    frames = []
    for path in sorted((ROOT / "sources").glob("*-broadcast-accuracy.csv")):
        raw = path.read_text(encoding="utf-8-sig")
        frame = pd.read_csv(StringIO(raw.split("\n\n", 1)[1]), float_precision="round_trip")
        if frame["update_index"].tolist() != list(range(1, 102)):
            raise ValueError(f"Unexpected update sequence: {path.name}")
        if frame["round"].tolist() != list(range(101)):
            raise ValueError(f"Unexpected logged round sequence: {path.name}")
        if not (frame["client_count"] == 10).all():
            raise ValueError("Expected an unweighted mean over ten clients")
        frame["dataset"] = path.name.split("_")[0]
        frame["source_file"] = path.name
        frame["accuracy_percent"] = 100 * frame["accuracy"]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data[data.dataset.isin(DATASETS)].copy()
    combinations = [("RBLA", "RBLA"), ("RBLA", "RBLA+"),
                    ("RBLA+", "RBLA"), ("RBLA+", "RBLA+")]
    limits = {"cifar10": (5, 80), "mnli": (30, 70), "qqp": (65, 85)}
    ticks = {"cifar10": [20, 40, 60, 80], "mnli": [30, 40, 50, 60, 70],
             "qqp": [65, 70, 75, 80, 85]}
    with plt.rc_context(RC):
        fig, axes = make_panels()
        for ax, ds in zip(axes, DATASETS):
            for driver, broadcast in combinations:
                rows = data[(data.dataset == ds) & (data.driver_method == driver)
                            & (data.evaluated_method == broadcast)].sort_values("update_index")
                if len(rows) != 101 or not np.isfinite(rows.accuracy_percent).all():
                    raise ValueError(f"Incomplete curve: {ds}, {driver}, {broadcast}")
                color = RED if driver == "RBLA" else BLUE
                is_plus = broadcast == "RBLA+"
                ax.plot(rows["round"], rows.accuracy_percent, color=color,
                        ls=(0, (4, 2.5)) if is_plus else "-", lw=.85,
                        marker="s" if is_plus else "o", ms=1.8, markevery=10,
                        mfc="white" if is_plus else color, mew=.45,
                        label=f"{driver} / {broadcast}", zorder=3 if is_plus else 2)
            ax.set(xlim=(0, 100), ylim=limits[ds], xlabel="Round",
                   ylabel="Accuracy (%)" if ds == DATASETS[0] else "", xticks=[0, 50, 100], yticks=ticks[ds])
            above_legend(ax, ncol=1, fontsize=15, anchor=1.395)
        save_figure(ROOT, fig, STEM, provenance=provenance, metadata={
            "title": "Training path and broadcast accuracy",
            "data_rows": len(data), "curves": 8, "updates_per_curve": 101,
            "metric": "unweighted mean accuracy of ten client broadcast models on the full held-out set",
            "legend_semantics": "training path / evaluated broadcast",
            "single_training_seed": 42, "smoothing": None, "uncertainty_band": None,
            "xaxis": "logged round, 0--100; 101 recorded evaluations",
            "xlim": [0, 100], "xticks": [0, 50, 100],
            "curve_order": combinations,
        })
    data[["dataset", "driver_method", "evaluated_method", "round", "update_index",
          "accuracy_percent", "average_loss", "client_count", "source_file"]].to_csv(
              ROOT / "plot_data.csv", index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
