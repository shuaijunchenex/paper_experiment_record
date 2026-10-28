"""Compact single-column style informed by RBLA+ complexity and gauge figures."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
import numpy as np
import pandas as pd

DATASETS = ("cifar10", "mnli")
LABELS = {"cifar10": "CIFAR10", "mnli": "MNLI", "qqp": "QQP"}
RED, BLUE = "#C44E52", "#4C72B0"
FONT = "Times New Roman"
RC = {
    "font.family": FONT, "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": .6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelpad": 2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix",
    "legend.frameon": False, "savefig.dpi": 600,
}


def verify_sources(root: Path) -> dict:
    provenance = json.loads((root / "sources/provenance.json").read_text(encoding="utf-8"))
    for item in provenance["files"]:
        actual = hashlib.sha256((root / "sources" / item["name"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {item['name']}")
    font_manager.findfont(FONT, fallback_to_default=False)
    return provenance


def make_panels():
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0))
    fig.subplots_adjust(left=.14, right=.97, bottom=.28, top=.82, wspace=.29)
    for i, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        ax.grid(axis="y", ls=":", alpha=.35, lw=.5)
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", width=.6, length=2.5, pad=2)
    return fig, axes


def above_legend(ax, *, ncol=1, fontsize=17, anchor=1.35):
    """One figure-level legend, shared by both panels."""
    fig = ax.figure
    if fig.legends:
        return fig.legends[0]
    handles, labels = ax.get_legend_handles_labels()
    # One-row legends use the extra space for taller plotting panels.
    fig.subplots_adjust(top=.82 if len(labels) == 4 else .90)
    return fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.54, 1.0),
                      ncol=2 if len(labels) == 4 else len(labels), frameon=False, fontsize=7.5,
                      handlelength=1.5, columnspacing=.8, handletextpad=.35,
                      borderaxespad=0, labelspacing=.15)


def save_figure(root, fig, stem, *, provenance, metadata):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Place each panel label four points below its own x-axis label.
    for i, (ax, dataset) in enumerate(zip(fig.axes, DATASETS)):
        xlabel_box = ax.xaxis.label.get_window_extent(renderer)
        y = (xlabel_box.y0 - 4 * fig.dpi / 72) / fig.bbox.height
        position = ax.get_position()
        fig.text(position.x0 + position.width / 2, y,
                 f"({chr(97+i)}) {LABELS[dataset]}",
                 ha="center", va="top", fontsize=8)
    fig.canvas.draw()
    page = fig.bbox
    outside = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        # Matplotlib retains off-axis ticks; they are not drawn on the page.
        if artist.axes is not None and artist in artist.axes.get_xticklabels() + artist.axes.get_yticklabels():
            continue
        b = artist.get_window_extent(renderer)
        if b.width and b.height and (b.x0 < -1 or b.y0 < -1 or b.x1 > page.x1 + 1 or b.y1 > page.y1 + 1):
            outside.append(artist.get_text())
    if outside:
        raise ValueError(f"Text outside figure: {outside}")
    fig.savefig(root / f"{stem}.pdf", bbox_inches=None, metadata={
        "Title": metadata["title"], "Subject": "Recorded RBLA+ ablation results; no new experiments"})
    metadata.update({
        "source_batch": provenance["batch"],
        "source_hashes_verified": True,
        "source_files": provenance["files"],
        "new_experiments_run": False,
        "displayed_datasets": list(DATASETS), "omitted_from_figure": ["qqp"],
        "layout": {"figsize_inches": [3.5, 2.0], "font_family": FONT,
                   "axis_font_pt": 8, "tick_font_pt": 7.5,
                   "panel_font_pt": 8, "panel_labels": "below x-axis labels", "panel_label_gap_pt": 4, "pdf_fonttype": 42,
                   "legend": "one shared figure legend", "column_layout": "single column",
                   "reference": ["rbla+/figures/complexity_comparison/broadcast_space_complexity.pdf",
                                 "rbla+/figures/gauge_induced_suboptimality/gauge_induced_suboptimality.pdf"],
                   "axes_positions": [list(ax.get_position().bounds) for ax in fig.axes]},
        "software": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    })
    (root / "plot_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plt.close(fig)
    print(root / f"{stem}.pdf")
