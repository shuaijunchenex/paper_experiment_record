import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_selected_client_labels(selected_clients_text):
    """Parse 'client.x@group-1;client.y@group-1' -> [x-1, y-1]."""
    if pd.isna(selected_clients_text) or not str(selected_clients_text).strip():
        return []

    labels = []
    for token in str(selected_clients_text).split(";"):
        m = re.search(r"client\.(\d+)@", token)
        if m:
            cid = int(m.group(1))
            labels.append(cid - 1)  # client 1-10 -> label 0-9
    return labels


def deviation_curve_from_sequence(label_seq, target_grid, n_labels=10):
    """For each deviation target, return # sampled clients needed to reach it."""
    if len(label_seq) == 0:
        return np.full_like(target_grid, np.nan, dtype=float)

    counts = np.zeros(n_labels, dtype=float)
    dev_hist = []

    for label in label_seq:
        if 0 <= label < n_labels:
            counts[label] += 1
        p = counts / counts.sum()
        # Total variation distance to uniform distribution
        dev = 0.5 * np.abs(p - 1.0 / n_labels).sum()
        dev_hist.append(dev)

    dev_hist = np.array(dev_hist)
    needed = []
    for t in target_grid:
        hit_idx = np.where(dev_hist <= t)[0]
        needed.append((hit_idx[0] + 1) if len(hit_idx) else len(dev_hist))
    return np.array(needed)


def bootstrap_band(label_seq, target_grid, n_boot=1000, seed=42):
    """Bootstrap min/max envelope over resampled client sequences."""
    rng = np.random.default_rng(seed)
    if len(label_seq) == 0:
        arr = np.full((n_boot, len(target_grid)), np.nan)
        return np.nanmedian(arr, axis=0), np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)

    label_seq = np.asarray(label_seq)
    curves = []
    for _ in range(n_boot):
        sample = rng.choice(label_seq, size=len(label_seq), replace=True)
        curves.append(deviation_curve_from_sequence(sample, target_grid, n_labels=10))
    curves = np.array(curves)

    mid = np.nanmedian(curves, axis=0)
    low = np.nanmin(curves, axis=0)
    high = np.nanmax(curves, axis=0)
    return mid, low, high


def plot_ablation_deviation(
    base_dir="./ablation_study/deviation",
    target_max=0.9,
    n_boot=1000,
    figsize=(8.2, 4.2),
    save_pdf="ablation_deviation_target_curve.pdf",
    save_png="ablation_deviation_target_curve.png",
    dpi_pdf=800,
    dpi_png=300,
    show=True,
):
    """Plot deviation-target curves for FedGRA and High-Loss from selection logs.

    client 1-10 are mapped to label 0-9. The shaded region is bootstrap min/max.
    """
    log_files = sorted(glob.glob(os.path.join(base_dir, "*_selection_log.csv"))) + \
                sorted(glob.glob(os.path.join(base_dir, "*_selected_clients.csv")))

    method_to_sequences = {"fedgra": [], "high_loss": []}

    for fpath in log_files:
        fname = os.path.basename(fpath).lower()
        if fname.startswith("fedgra"):
            method = "fedgra"
        elif fname.startswith("high_loss"):
            method = "high_loss"
        else:
            continue

        df = pd.read_csv(fpath)
        if "selected_clients" not in df.columns:
            continue

        seq = []
        for txt in df["selected_clients"].tolist():
            seq.extend(parse_selected_client_labels(txt))

        if seq:
            method_to_sequences[method].append(seq)

    target_grid = np.linspace(0.0, float(target_max), int(float(target_max) * 100) + 1)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    plot_specs = [
        ("fedgra", "FedGRA", "#6f2cff"),
        ("high_loss", "High-Loss", "#d62728"),
    ]

    for ax, (method_key, title, color) in zip(axes, plot_specs):
        seqs = method_to_sequences.get(method_key, [])
        if not seqs:
            ax.set_title(f"{title} (no log)")
            continue

        mids, lows, highs = [], [], []
        for i, seq in enumerate(seqs):
            mid, low, high = bootstrap_band(seq, target_grid, n_boot=n_boot, seed=42 + i)
            mids.append(mid)
            lows.append(low)
            highs.append(high)

        mid_curve = np.nanmedian(np.array(mids), axis=0)
        low_curve = np.nanmin(np.array(lows), axis=0)
        high_curve = np.nanmax(np.array(highs), axis=0)

        ax.fill_between(target_grid, low_curve, high_curve, color="gray", alpha=0.35, label="Empirical Dev.")
        ax.plot(target_grid, mid_curve, color=color, linewidth=2.0, label=title)

        ax.set_title(title, fontsize=14, pad=6)
        ax.set_xlabel("Deviation Target", fontsize=12)
        ax.set_yscale("log")
        ax.set_xlim(0.0, float(target_max))
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    axes[0].set_ylabel("# of Sampled Clients", fontsize=12)

    plt.tight_layout()
    if save_pdf:
        plt.savefig(save_pdf, dpi=dpi_pdf, bbox_inches="tight")
    if save_png:
        plt.savefig(save_png, dpi=dpi_png, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes, method_to_sequences


def plot_ablation_deviation_combined(
    base_dir="./ablation_study/deviation",
    target_max=0.9,
    n_boot=1000,
    figsize=(6.0, 4.6),
    save_pdf="ablation_deviation_target_curve_combined.pdf",
    save_png="ablation_deviation_target_curve_combined.png",
    dpi_pdf=800,
    dpi_png=300,
    show=True,
):
    """Plot FedGRA and High-Loss curves in one axis for direct comparison."""
    log_files = sorted(glob.glob(os.path.join(base_dir, "*_selection_log.csv"))) + \
                sorted(glob.glob(os.path.join(base_dir, "*_selected_clients.csv")))

    method_to_sequences = {"fedgra": [], "high_loss": []}

    for fpath in log_files:
        fname = os.path.basename(fpath).lower()
        if fname.startswith("fedgra"):
            method = "fedgra"
        elif fname.startswith("high_loss"):
            method = "high_loss"
        else:
            continue

        df = pd.read_csv(fpath)
        if "selected_clients" not in df.columns:
            continue

        seq = []
        for txt in df["selected_clients"].tolist():
            seq.extend(parse_selected_client_labels(txt))

        if seq:
            method_to_sequences[method].append(seq)

    target_grid = np.linspace(0.0, float(target_max), int(float(target_max) * 100) + 1)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_specs = [
        ("fedgra", "FedGRA", "#6f2cff"),
        ("high_loss", "High-Loss", "#d62728"),
    ]

    for method_key, title, color in plot_specs:
        seqs = method_to_sequences.get(method_key, [])
        if not seqs:
            continue

        mids, lows, highs = [], [], []
        for i, seq in enumerate(seqs):
            mid, low, high = bootstrap_band(seq, target_grid, n_boot=n_boot, seed=42 + i)
            mids.append(mid)
            lows.append(low)
            highs.append(high)

        mid_curve = np.nanmedian(np.array(mids), axis=0)
        low_curve = np.nanmin(np.array(lows), axis=0)
        high_curve = np.nanmax(np.array(highs), axis=0)

        ax.fill_between(target_grid, low_curve, high_curve, color=color, alpha=0.12)
        ax.plot(target_grid, mid_curve, color=color, linewidth=2.3, label=title)

    ax.set_xlabel("Deviation Target", fontsize=12)
    ax.set_ylabel("# of Sampled Clients", fontsize=12)
    ax.set_yscale("log")
    ax.set_xlim(0.0, float(target_max))
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    plt.tight_layout()
    if save_pdf:
        plt.savefig(save_pdf, dpi=dpi_pdf, bbox_inches="tight")
    if save_png:
        plt.savefig(save_png, dpi=dpi_png, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax, method_to_sequences


def main():
    plot_ablation_deviation(show=False)
    plot_ablation_deviation_combined(show=False)


if __name__ == "__main__":
    main()
