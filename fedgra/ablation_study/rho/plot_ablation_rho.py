import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RHO_COLORS = {
    0.1: "red",      # AdaFL color in experiment_figures.py/notebook
    0.3: "green",    # AFL
    0.5: "blue",     # FedGRA
    0.7: "orange",   # FedSDR
    0.9: "purple",   # Oort
}


def _extract_rho(path):
    match = re.search(r"rho_?(\d+)", path.stem.lower())
    if match is None:
        return None

    token = match.group(1)
    if token.startswith("0") or len(token) == 2:
        return int(token) / 10
    return float(token)


def _read_training_csv(path):
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline()

    skiprows = 2 if first_line.startswith("Config,") else 0
    return pd.read_csv(path, skiprows=skiprows)


def load_rho_runs(base_dir=None, dataset="fmnist", metric="accuracy"):
    """Load rho ablation runs for one dataset.

    Returns a dict: {rho_value: dataframe}. The dataframes are sorted by round.
    """
    base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    dataset_dir = base_dir / dataset

    runs = {}
    for path in sorted(dataset_dir.glob("*.csv")):
        rho = _extract_rho(path)
        if rho is None:
            continue

        df = _read_training_csv(path)
        if "round" not in df.columns or metric not in df.columns:
            missing = {"round", metric} - set(df.columns)
            raise ValueError(f"{path} missing columns: {sorted(missing)}")

        df = df.copy()
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df.dropna(subset=["round", metric]).sort_values("round")
        runs[rho] = df

    if not runs:
        raise FileNotFoundError(f"No rho CSV files found in {dataset_dir}")

    return dict(sorted(runs.items()))


def plot_rho_ablation_mnist_fmnist(
    base_dir=None,
    datasets=("mnist", "fmnist"),
    metric="accuracy",
    smooth_window=None,
    plot_raw=False,
    figsize=(10, 5),
    y_lims=((0.0, 0.8), (0.0, 0.7)),
    output_prefix=None,
    dpi=800,
    shared_legend=True,
    legend_loc="upper center",
    legend_bbox_to_anchor=(0.5, 1.17),
    legend_ncol=None,
    line_width=2,
    line_alpha=1,
    line_zorder=3,
    raw_line_width=0.8,
    raw_alpha=0.35,
    raw_linestyle="--",
    raw_zorder=1,
    title_fontsize=22,
    title_pad=18,
    label_fontsize=24,
    tick_fontsize=20,
    legend_fontsize=20,
    show=True,
):
    """Plot MNIST and FMNIST rho ablation results in one row with two subplots."""
    base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    output_prefix = Path(output_prefix) if output_prefix is not None else base_dir / "ablation_rho_mnist_fmnist"
    y_lims = [None] * len(datasets) if y_lims is None else y_lims

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 14,
            "axes.labelsize": label_fontsize,
            "axes.titlesize": title_fontsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_fontsize,
            "ytick.labelsize": tick_fontsize,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, len(datasets), figsize=figsize, sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    all_runs = {}
    legend_handles = []
    legend_labels = []
    title_map = {"fmnist": "FMNIST", "mnist": "MNIST"}
    y_label = "Test Accuracy" if metric == "accuracy" else metric.replace("_", " ").title()

    for ax, dataset, y_lim in zip(axes, datasets, y_lims):
        runs = load_rho_runs(base_dir=base_dir, dataset=dataset, metric=metric)
        all_runs[dataset] = runs

        for rho, df in runs.items():
            color = RHO_COLORS.get(rho)
            x = df["round"]
            y = df[metric]
            label = rf"$\rho={rho:g}$"

            if plot_raw:
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=raw_line_width,
                    alpha=raw_alpha,
                    linestyle=raw_linestyle,
                    zorder=raw_zorder,
                )

            if smooth_window and smooth_window > 1:
                y_plot = y.rolling(smooth_window, min_periods=1).mean()
            else:
                y_plot = y

            (line,) = ax.plot(
                x,
                y_plot,
                color=color,
                linewidth=line_width,
                alpha=line_alpha,
                zorder=line_zorder,
                label=label,
            )
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        ax.set_title(title_map.get(dataset, dataset.upper()), fontsize=title_fontsize, pad=title_pad)
        ax.set_xlabel("Communication rounds", fontsize=label_fontsize)
        ax.set_xlim([0, 100])
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.set_ylabel(y_label, fontsize=label_fontsize)
        if not shared_legend:
            ax.legend(frameon=False, ncol=1, loc="lower right")

    if shared_legend:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            ncol=legend_ncol or len(legend_labels),
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            fontsize=legend_fontsize,
        )

    fig.tight_layout()

    if output_prefix:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_prefix.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
        fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, all_runs


def main():
    plot_rho_ablation_mnist_fmnist(show=False)


if __name__ == "__main__":
    main()
