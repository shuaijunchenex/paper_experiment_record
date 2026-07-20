import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {"FedGRA": "#2a7de1", "High-Loss": "#e53e3e"}

LOG_FILES = {
    "MNIST": {
        "FedGRA": Path("../node_participation/mnist/fedgra_mnist_min100_-train-20260617_192203-c076d59e_selection_log.csv"),
        "High-Loss": Path("../node_participation/mnist/high_loss_mnist_-train-20260617_182132-31823659_selection_log.csv"),
    },
    "FMNIST": {
        "FedGRA": Path("../node_participation/fmnist/fedgra_fmnist_min100_-train-20260617_191039-12dbf5cd_selection_log.csv"),
        "High-Loss": Path("../node_participation/fmnist/high_loss_fmnist_-train-20260617_181138-d62f9a9e_selection_log.csv"),
    },
}


def parse_selected_clients(selected_clients_text):
    """Parse 'client.x@group-1;client.y@group-1' into a set of client ids."""
    if pd.isna(selected_clients_text):
        return set()

    return {
        int(match.group(1))
        for match in re.finditer(r"client\.(\d+)@", str(selected_clients_text))
    }


def load_selection_log(log_path):
    """Load one selection log and attach parsed selected-client sets."""
    df = pd.read_csv(log_path)
    if "round" not in df.columns or "selected_clients" not in df.columns:
        missing = {"round", "selected_clients"} - set(df.columns)
        raise ValueError(f"{log_path} missing columns: {sorted(missing)}")

    df = df.copy()
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df.dropna(subset=["round"]).sort_values("round")
    df["client_set"] = df["selected_clients"].apply(parse_selected_clients)
    return df


def window_client_overlap(log_path, round=5):
    """Return repeated client selections in each sliding window.

    For a window of R rounds, overlap is:
        total selected-client occurrences - number of unique selected clients

    When each round selects 2 clients, the maximum is 2 * (R - 1).
    """
    df = load_selection_log(log_path)
    round = _resolve_round_window(round=round)

    records = []
    rows = list(df[["round", "client_set"]].itertuples(index=False, name=None))

    for idx in range(round - 1, len(rows)):
        window_rows = rows[idx - round + 1 : idx + 1]
        window_sets = [client_set for _, client_set in window_rows]
        total_selected = sum(len(client_set) for client_set in window_sets)
        unique_clients = set().union(*window_sets) if window_sets else set()

        records.append(
            {
                "round": int(window_rows[-1][0]),
                "window_start_round": int(window_rows[0][0]),
                "window_rounds": round,
                "total_selected": total_selected,
                "unique_clients": len(unique_clients),
                "overlap": total_selected - len(unique_clients),
            }
        )

    return pd.DataFrame(records)


def _resolve_round_window(round=None, lookback_rounds=None):
    """Resolve the user-facing round parameter into a sliding-window size."""
    value = lookback_rounds if round is None else round
    value = int(value)
    if value < 1:
        raise ValueError("round must be >= 1")
    return value


def load_recent_overlap_data(base_dir=None, log_files=None, round=None, lookback_rounds=5):
    """Load sliding-window client-overlap curves for all configured datasets."""
    base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    log_files = LOG_FILES if log_files is None else log_files
    round = _resolve_round_window(round=round, lookback_rounds=lookback_rounds)

    overlap_data = {}
    for dataset_name, methods in log_files.items():
        overlap_data[dataset_name] = {}
        for method_name, rel_path in methods.items():
            log_path = (base_dir / rel_path).resolve()
            if not log_path.exists():
                raise FileNotFoundError(log_path)
            overlap_data[dataset_name][method_name] = window_client_overlap(
                log_path,
                round=round,
            )

    return overlap_data


def _draw_overlap_panel(
    ax,
    data_dict,
    title,
    round=5,
    y_max=None,
    x_ticks=(0, 25, 50, 75, 100),
    show_ylabel=True,
):
    for method_name, df in data_dict.items():
        color = COLORS[method_name]
        ax.plot(
            df["round"],
            df["overlap"],
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3,
            alpha=0.95,
            label=method_name,
        )

    ax.set_title(title, fontsize=22)
    ax.set_xlabel("Communication Round", fontsize=20)
    if show_ylabel:
        ax.set_ylabel(f"# Repeated Clients\n{round}-Round Window", fontsize=20)
    ax.set_xlim(0, 100)
    ax.set_xticks(x_ticks)
    if y_max is not None:
        ax.set_ylim(-0.1, y_max + 0.1)
        ax.set_yticks(range(0, y_max + 1))
    ax.grid(True, alpha=0.3)


def plot_adjacent_client_overlap(
    base_dir=None,
    dataset_names=("MNIST", "FMNIST"),
    round=None,
    lookback_rounds=5,
    save_prefix=None,
    figsize=(10, 5),
    x_ticks=(0, 25, 50, 75, 100),
    legend_ncol=2,
    legend_loc="upper center",
    legend_bbox_to_anchor=(0.5, 1.17),
    dpi=800,
    show=True,
):
    """Plot overlap counts against clients selected in recent previous rounds."""
    base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    round = _resolve_round_window(round=round, lookback_rounds=lookback_rounds)
    save_prefix = (
        Path(save_prefix)
        if save_prefix is not None
        else base_dir / f"mnist_fmnist_window{round}_client_overlap"
    )

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    overlap_data = load_recent_overlap_data(
        base_dir=base_dir,
        round=round,
    )
    y_max = max(
        int(df["overlap"].max())
        for dataset_name in dataset_names
        for df in overlap_data[dataset_name].values()
    )

    fig, axes = plt.subplots(1, len(dataset_names), figsize=figsize, sharey=True)
    if len(dataset_names) == 1:
        axes = [axes]

    handles = labels = None
    for idx, (ax, dataset_name) in enumerate(zip(axes, dataset_names)):
        _draw_overlap_panel(
            ax,
            overlap_data[dataset_name],
            title=dataset_name,
            round=round,
            y_max=y_max,
            x_ticks=x_ticks,
            show_ylabel=(idx == 0),
        )
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
        fontsize=15,
        frameon=False,
    )

    plt.tight_layout()
    if save_prefix:
        save_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_prefix.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
        fig.savefig(save_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, overlap_data


def adjacent_client_overlap(log_path):
    """Backward-compatible helper for adjacent-round overlap."""
    return window_client_overlap(log_path, round=2)


def load_adjacent_overlap_data(base_dir=None, log_files=None):
    """Backward-compatible helper for adjacent-round overlap data."""
    return load_recent_overlap_data(base_dir=base_dir, log_files=log_files, round=2)


def main():
    plot_adjacent_client_overlap(show=False)


if __name__ == "__main__":
    main()
