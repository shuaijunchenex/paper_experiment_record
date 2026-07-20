from typing import List, Union, Dict, Optional, Sequence
import json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_LEGEND_MAP = {
    "adafl": "AdaFL",
    "afl": "AFL",
    "fedgra": "FedGRA",
    "fedsdr": "FedSDR",
    "high_loss": "High-Loss",
    "high_weight_divergence": "High-Weight-Div",
    "oort": "Oort",
    "powd": "Pow-D",
    "pow-d": "Pow-D",
    "pyramidfl": "PyramidFL",
    "repufl": "RepuFL",
    "random": "Random",
    "all_participate": "All-Participate",
}

DEFAULT_COLOR_MAP = {
    "FedGRA": "blue",
    "AdaFL": "red",
    "AFL": "green",
    "FedSDR": "orange",
    "Oort": "purple",
    "Pow-D": "brown",
    "PyramidFL": "magenta",
    "Random": "gray",
    "RepuFL": "cyan",
    "High-Loss": "pink",
    "High-Weight-Div": "olive",
    "All-Participate": "teal",
}


def _save_current_figure(save_path: Union[str, Path], dpi: int = 800) -> Path:
    """Save the current matplotlib figure, defaulting to PDF when no suffix is given."""
    output_path = Path(save_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".pdf")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def read_training_csv(
    files: Union[str, Path, List[Union[str, Path]]],
    columns: List[str] | None = None,
    concat: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Read one or multiple training CSV files.

    Args:
        files: single file path or list of file paths
        columns: list of column names to read (None = read all)
        concat: whether to concatenate multiple files
        **kwargs: additional arguments for pd.read_csv

    Returns:
        pandas.DataFrame
    """
    if isinstance(files, (str, Path)):
        files = [files]

    dfs = []
    for f in files:
        df = pd.read_csv(f, **kwargs)

        if columns is not None:
            missing = set(columns) - set(df.columns)
            for col in missing:
                df[col] = pd.NA
            df = df[columns]

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    if concat and len(dfs) > 1:
        return pd.concat(dfs, ignore_index=True)

    return dfs[0]


def plot_learning_curves(
    data_dict: Union[pd.DataFrame, Dict[str, pd.Series]],
    x_range: Optional[int] = None,
    column: Optional[str] = None,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    legend_map: Optional[Dict[str, str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    title: str = "Learning Curves",
    x_label: str = "Communication Rounds",
    y_label: str = "Test Accuracy",
    y_lim: tuple = (0, 1),
    window_size: int = 10,
    figsize: tuple = (5, 5),
    is_legend: bool = True,
    save_path: Optional[str] = None,
    plot_raw: bool = True,
    raw_alpha: float = 0.35,
    raw_linewidth: float = 0.7,
    smooth_linewidth: float = 2,
    dpi: int = 800,
    fontsize_title: int = 22,
    fontsize_label: int = 24,
    fontsize_legend: int = 18,
    fontsize_tick: int = 20,
) -> None:
    """
    Plot learning curves with optional rolling average smoothing.
    
    Args:
        data_dict: DataFrame with columns to plot, or dict mapping keys to Series
        x_range: number of x-axis points to display (if None, use all data)
        column: specific column to plot from DataFrame (if data_dict is DataFrame)
        colors: list of colors for each curve (default: red, green, blue, etc.)
        labels: list of labels for each curve (default: uses series names or generic names)
        legend_map: dict mapping key (dataset name) to legend label (overrides labels if provided)
        title: plot title
        x_label: x-axis label
        y_label: y-axis label
        y_lim: tuple of (min, max) for y-axis
        window_size: window size for rolling average
        figsize: (width, height) of the figure
        is_legend: whether to show legend
        save_path: if provided, save figure to this path (without extension)
        plot_raw: whether to plot raw data with dashed lines
        raw_alpha: alpha transparency for raw data lines
        raw_linewidth: linewidth for raw data lines
        smooth_linewidth: linewidth for smoothed lines
        dpi: dpi for saving
        fontsize_title: font size for title
        fontsize_label: font size for axis labels
        fontsize_legend: font size for legend
        fontsize_tick: font size for tick labels
    """
    plt.figure(figsize=figsize)
    
    # Convert DataFrame to dictionary if needed
    if isinstance(data_dict, pd.DataFrame):
        if column is not None:
            # Single column from DataFrame
            data_dict = {column: data_dict[column]}
        else:
            # All numeric columns
            data_dict = {col: data_dict[col] for col in data_dict.select_dtypes(include=[np.number]).columns}
    
    # Set default colors — cycle if there are more datasets than colors
    default_colors = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta", "yellow", "black"]
    if colors is None:
        colors = [default_colors[i % len(default_colors)] for i in range(len(data_dict))]
    elif len(colors) < len(data_dict):
        colors = [colors[i % len(colors)] for i in range(len(data_dict))]
    
    # Set default labels
    if legend_map is not None:
        # Use legend_map to override labels for each key
        labels = [legend_map.get(key, str(key)) for key in data_dict.keys()]
    elif labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(data_dict))]
    elif len(labels) < len(data_dict):
        # If not enough labels provided, use generic names
        labels = labels + [f"Dataset {i+1}" for i in range(len(labels), len(data_dict))]

    # Apply color_map to fix specific colors for certain labels
    if color_map is not None:
        for i, label in enumerate(labels):
            if label in color_map:
                colors[i] = color_map[label]
        # 避免 mapped colors 与其他默认颜色冲突：将冲突的非目标颜色替换掉
        mapped_colors = set(color_map.values())
        unmapped_palette = [c for c in default_colors if c not in mapped_colors]
        for i, label in enumerate(labels):
            if label not in color_map and colors[i] in mapped_colors:
                colors[i] = unmapped_palette[i % len(unmapped_palette)]
    
    # Process each data series
    for idx, (key, series) in enumerate(data_dict.items()):
        # Extract data up to x_range
        if x_range is not None:
            data = series.iloc[:x_range].values
        else:
            data = series.values
            if x_range is None:
                x_range = len(data)
        
        x = np.arange(len(data))
        
        # Plot raw data
        if plot_raw:
            plt.plot(
                x,
                data,
                marker='',
                linestyle='dashed',
                markersize=1,
                linewidth=raw_linewidth,
                alpha=raw_alpha,
                color=colors[idx]
            )
        
        # Calculate and plot rolling average
        if len(data) >= window_size:
            rolling_avg = []
            for i in range(len(data) - window_size + 1):
                window_data = data[i:i+window_size]
                rolling_avg.append(np.mean(window_data))
            
            x_rolling = np.arange(window_size - 1, len(data))
            
            plt.plot(
                x_rolling,
                rolling_avg,
                color=colors[idx],
                linewidth=smooth_linewidth,
                label=labels[idx]
            )
    
    # Set plot properties
    plt.ylim(y_lim)
    plt.xlim([0, x_range])
    plt.title(title, fontsize=fontsize_title)
    plt.xlabel(x_label, fontsize=fontsize_label)
    plt.ylabel(y_label, fontsize=fontsize_label)
    plt.tick_params(axis='x', labelsize=fontsize_tick)
    plt.tick_params(axis='y', labelsize=fontsize_tick)
    
    if is_legend:
        plt.legend(fontsize=fontsize_legend)
    
    if save_path:
        _save_current_figure(save_path, dpi=dpi)
    
    plt.show()


def _extract_method_from_config(csv_file: Union[str, Path]) -> Optional[str]:
    """
    Extract client selection method from the first config line in FedGRA csv.

    Expected first line format:
        Config,{...json...}
    """
    csv_file = Path(csv_file)
    if not csv_file.exists():
        return None

    try:
        with csv_file.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
    except OSError:
        return None

    if not first_line:
        return None

    if "," not in first_line:
        return None

    left, payload = first_line.split(",", 1)
    if left.strip().lower() != "config":
        return None

    try:
        cfg = json.loads(payload)
    except json.JSONDecodeError:
        return None

    method = cfg.get("client_selection", {}).get("method")
    if method is None:
        return None

    return str(method)


def build_fedgra_metric_dict(
    files: Union[str, Path, List[Union[str, Path]]],
    metric: str = "accuracy",
    header: int = 1,
    auto_legend: bool = True,
    labels: Optional[List[str]] = None,
    legend_map: Optional[Dict[str, str]] = None,
    sort_files: bool = True,
    **read_csv_kwargs,
) -> Dict[str, pd.Series]:
    """
    Build a plotting dictionary for FedGRA csv files.

    Args:
        files:
            - csv directory path (e.g. fedgra/mnist)
            - a single csv file path
            - list of csv file paths
        metric: metric column to read from CSV (default: accuracy)
        header: row index of CSV column header (default: 1)
        auto_legend: whether to auto parse method from config line as legend
        labels: manual legend labels; has higher priority than auto legend
        legend_map: map raw legend name -> custom legend name
        sort_files: whether to sort csv paths
        **read_csv_kwargs: extra kwargs passed to read_training_csv

    Returns:
        Dict[str, pd.Series], directly consumable by plot_learning_curves.
    """
    # Normalize input into a list of csv files
    if isinstance(files, (str, Path)):
        files = Path(files)
        if files.is_dir():
            csv_files = list(files.glob("*.csv"))
        else:
            csv_files = [files]
    else:
        csv_files = [Path(f) for f in files]

    csv_files = [f for f in csv_files if f.suffix.lower() == ".csv"]
    if sort_files:
        csv_files = sorted(csv_files)

    if not csv_files:
        raise ValueError("No csv files found from the provided input.")

    data_dict: Dict[str, pd.Series] = {}
    for idx, csv_file in enumerate(csv_files):
        df = read_training_csv(
            [csv_file],
            columns=[metric],
            header=header,
            **read_csv_kwargs,
        )

        if metric not in df.columns:
            raise ValueError(f"Column '{metric}' not found in file: {csv_file}")

        if labels is not None and idx < len(labels):
            legend_name = labels[idx]
        elif auto_legend:
            method = _extract_method_from_config(csv_file)
            if method:
                legend_name = method
            else:
                legend_name = f"Dataset {idx + 1}"
        else:
            legend_name = f"Dataset {idx + 1}"

        if legend_map is not None:
            legend_name = legend_map.get(legend_name, legend_name)

        # Avoid overwriting when methods repeat (e.g. multiple random runs)
        if legend_name in data_dict:
            legend_name = f"{legend_name}_{idx + 1}"

        data_dict[legend_name] = df[metric]

    return data_dict


def _auto_csv_header(csv_file: Union[str, Path]) -> int:
    """Return 1 for FedGRA config-prefixed csv files, otherwise 0."""
    try:
        with Path(csv_file).open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
    except OSError:
        return 0

    return 1 if first_line.lower().startswith("config,") else 0


def _normalize_method_name(method: str) -> str:
    return method.lower().replace("-", "_").strip()


def _method_from_filename(csv_file: Union[str, Path]) -> str:
    stem = Path(csv_file).stem
    if stem.startswith("aws_fmnist_result_"):
        return stem.replace("aws_fmnist_result_", "", 1)
    if stem.endswith("_experiment_result"):
        return stem.replace("_experiment_result", "")

    # Common local filename pattern: fedgra_mnist_-train-...
    prefix = stem.split("-train-", 1)[0].strip("_")
    if "_" in prefix:
        return prefix.split("_", 1)[0]
    return prefix or stem


def _method_label(
    raw_method: str,
    legend_map: Optional[Dict[str, str]] = None,
) -> str:
    if legend_map is None:
        return raw_method

    normalized = _normalize_method_name(raw_method)
    return (
        legend_map.get(raw_method)
        or legend_map.get(normalized)
        or legend_map.get(normalized.replace("_", "-"))
        or raw_method
    )


def _unique_label(label: str, existing: Dict[str, pd.Series]) -> str:
    if label not in existing:
        return label

    index = 2
    candidate = f"{label}_{index}"
    while candidate in existing:
        index += 1
        candidate = f"{label}_{index}"
    return candidate


def read_metric_folder(
    directory: Union[str, Path],
    metric: str = "accuracy",
    header: Union[int, str] = "auto",
    legend_map: Optional[Dict[str, str]] = DEFAULT_LEGEND_MAP,
    method_order: Optional[Sequence[str]] = None,
    sort_files: bool = True,
    **read_csv_kwargs,
) -> Dict[str, pd.Series]:
    """
    Read one metric from all CSV files in a folder.

    This supports both ordinary CSV files (header on row 0) and FedGRA CSV files
    whose first row is a JSON config and whose header is on row 1.
    """
    directory = Path(directory)
    csv_files = list(directory.glob("*.csv")) if directory.is_dir() else [directory]
    csv_files = [csv_file for csv_file in csv_files if csv_file.suffix.lower() == ".csv"]
    if sort_files:
        csv_files = sorted(csv_files)

    if not csv_files:
        raise ValueError(f"No csv files found in: {directory}")

    raw_data: Dict[str, pd.Series] = {}
    for csv_file in csv_files:
        csv_header = _auto_csv_header(csv_file) if header == "auto" else header
        df = pd.read_csv(csv_file, header=csv_header, **read_csv_kwargs)
        if metric not in df.columns:
            raise ValueError(f"Column '{metric}' not found in file: {csv_file}")

        raw_method = _extract_method_from_config(csv_file) or _method_from_filename(csv_file)
        raw_data[_unique_label(raw_method, raw_data)] = df[metric]

    ordered_items = list(raw_data.items())
    if method_order is not None:
        order_index = {
            _normalize_method_name(method): idx for idx, method in enumerate(method_order)
        }
        ordered_items = sorted(
            ordered_items,
            key=lambda item: order_index.get(_normalize_method_name(item[0]), len(order_index)),
        )

    data_dict: Dict[str, pd.Series] = {}
    for raw_method, series in ordered_items:
        label = _method_label(raw_method, legend_map=legend_map)
        data_dict[_unique_label(label, data_dict)] = series

    return data_dict


def _apply_default_style() -> None:
    try:
        from style import MatplotlibStyle
    except ImportError:
        return

    MatplotlibStyle().apply()


def plot_metric_folders(
    directories: Union[str, Path, Sequence[Union[str, Path]]],
    save_path: Optional[Union[str, Path]] = None,
    titles: Optional[Sequence[str]] = None,
    metric: str = "accuracy",
    header: Union[int, str] = "auto",
    legend_map: Optional[Dict[str, str]] = DEFAULT_LEGEND_MAP,
    color_map: Optional[Dict[str, str]] = DEFAULT_COLOR_MAP,
    method_order: Optional[Sequence[str]] = None,
    figsize: Optional[tuple] = None,
    n_cols: Optional[int] = None,
    x_range: Optional[int] = 100,
    x_ticks: Optional[Sequence[float]] = None,
    y_lim: Union[tuple, Sequence[tuple]] = (0, 0.8),
    window_size: int = 5,
    plot_raw: bool = True,
    plot_smooth: bool = True,
    raw_alpha: float = 0.35,
    raw_linewidth: float = 0.7,
    smooth_linewidth: float = 2,
    title_fontsize: int = 22,
    label_fontsize: int = 24,
    tick_fontsize: int = 20,
    legend_fontsize: int = 20,
    legend_ncol: int = 4,
    legend_loc: str = "upper center",
    legend_bbox_to_anchor: tuple = (0.5, 1.17),
    sharey: bool = False,
    use_style: bool = True,
    dpi: int = 800,
    show: bool = True,
    **read_csv_kwargs,
) -> List[Dict[str, pd.Series]]:
    """
    Plot metric curves from one or more folders and optionally save the figure.

    Args:
        directories: a folder path, csv path, or a list of folders/csv files.
        save_path: output path, e.g. ``"aws_combined.pdf"`` or ``"figs/aws.png"``.
        titles: subplot titles. Defaults to each directory name.
        metric: CSV metric column to plot.
        header: ``"auto"`` supports both ordinary and FedGRA config-prefixed CSVs.
        n_cols: number of subplot columns. Defaults to one row with all plots.
    """
    if isinstance(directories, (str, Path)):
        directories = [directories]
    directories = list(directories)

    if not directories:
        raise ValueError("At least one directory or csv file is required.")

    if use_style:
        _apply_default_style()

    if figsize is None:
        figsize = (5 * len(directories), 5)

    if titles is None:
        titles = [Path(directory).name for directory in directories]

    if isinstance(y_lim[0], (int, float)):
        y_lims = [y_lim] * len(directories)
    else:
        y_lims = list(y_lim)

    data_sets = [
        read_metric_folder(
            directory=directory,
            metric=metric,
            header=header,
            legend_map=legend_map,
            method_order=method_order,
            **read_csv_kwargs,
        )
        for directory in directories
    ]

    if n_cols is None:
        n_cols = len(directories)
    n_rows = int(np.ceil(len(directories) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=sharey)
    axes = np.asarray(axes).reshape(-1)
    for ax in axes[len(directories):]:
        ax.set_visible(False)
    axes = axes[:len(directories)]

    legend_labels: List[str] = []
    for data_dict in data_sets:
        for label in data_dict:
            if label not in legend_labels:
                legend_labels.append(label)

    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_by_label = {}
    for idx, label in enumerate(legend_labels):
        color_by_label[label] = (
            color_map.get(label)
            if color_map is not None and label in color_map
            else default_colors[idx % len(default_colors)]
        )

    y_label = metric.replace("_", " ").title()
    if metric == "accuracy":
        y_label = "Test Accuracy"

    for ax, title, data_dict, current_y_lim in zip(axes, titles, data_sets, y_lims):
        for label, series in data_dict.items():
            data = series.dropna().to_numpy()
            if x_range is not None:
                data = data[:x_range]
            x = np.arange(len(data))
            color = color_by_label[label]

            if plot_raw:
                ax.plot(
                    x,
                    data,
                    linestyle="dashed",
                    linewidth=raw_linewidth,
                    alpha=raw_alpha,
                    color=color,
                )

            if plot_smooth and len(data) >= window_size:
                rolling = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
                x_roll = np.arange(window_size - 1, len(data))
                ax.plot(x_roll, rolling, linewidth=smooth_linewidth, color=color, label=label)
            elif not plot_smooth:
                ax.plot(x, data, linewidth=smooth_linewidth, color=color, label=label)

        ax.set_title(title, fontsize=title_fontsize, pad=1)
        ax.set_xlabel("Communication rounds", fontsize=label_fontsize)
        if x_range is not None:
            ax.set_xlim([0, x_range])
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        ax.set_ylim(current_y_lim)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)

    axes[0].set_ylabel(y_label, fontsize=label_fontsize)
    if not sharey:
        for ax in axes[1:]:
            ax.set_ylabel(y_label, fontsize=label_fontsize)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=color_by_label[label], linewidth=smooth_linewidth, label=label)
        for label in legend_labels
    ]
    fig.legend(
        handles=handles,
        loc=legend_loc,
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        frameon=False,
        bbox_to_anchor=legend_bbox_to_anchor,
    )

    plt.tight_layout()
    if save_path:
        _save_current_figure(save_path, dpi=dpi)
    if show:
        plt.show()

    return data_sets


def plot_accuracy_from_folder(
    directory: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    **kwargs,
) -> Dict[str, pd.Series]:
    """Convenience wrapper for plotting accuracy curves from one folder."""
    data_sets = plot_metric_folders(
        directories=[directory],
        save_path=save_path,
        titles=[title or Path(directory).name],
        metric="accuracy",
        **kwargs,
    )
    return data_sets[0]


def plot_combined_smoothed_datasets(
    dir1: Union[str, Path],
    dir2: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
    **kwargs,
) -> List[Dict[str, pd.Series]]:
    """Convenience wrapper for the common two-panel smoothed accuracy figure."""
    return plot_metric_folders(
        directories=[dir1, dir2],
        save_path=save_path,
        titles=[label1, label2],
        metric="accuracy",
        plot_raw=True,
        plot_smooth=True,
        **kwargs,
    )


def plot_four_smoothed_datasets(
    directories: Sequence[Union[str, Path]],
    save_path: Optional[Union[str, Path]] = None,
    titles: Optional[Sequence[str]] = None,
    figsize: tuple = (20, 5),
    x_ticks: Optional[Sequence[float]] = (0, 25, 50, 75, 100),
    y_lim: Union[tuple, Sequence[tuple]] = (0, 0.8),
    legend_ncol: int = 4,
    legend_loc: str = "upper center",
    legend_bbox_to_anchor: tuple = (0.5, 1.17),
    return_data: bool = False,
    **kwargs,
) -> Optional[List[Dict[str, pd.Series]]]:
    """
    Convenience wrapper for a one-row, four-panel smoothed accuracy figure.

    Args:
        directories: exactly four data folders or csv paths.
        save_path: output path. If no suffix is given, PDF is used.
        titles: exactly four subplot titles.
        figsize: whole figure size, e.g. ``(20, 5)``.
        x_ticks: x-axis tick locations. Defaults to ``(0, 25, 50, 75, 100)``.
        y_lim: either one shared ``(min, max)`` tuple or four per-panel tuples.
        legend_ncol: number of columns in the shared legend.
        legend_loc: legend location passed to matplotlib, e.g. ``"lower center"``.
        legend_bbox_to_anchor: legend anchor, e.g. ``(0.5, -0.08)``.
        return_data: if True, return the data dictionaries used for plotting.
    """
    directories = list(directories)
    if len(directories) != 4:
        raise ValueError("plot_four_smoothed_datasets requires exactly four directories.")

    if titles is not None and len(titles) != 4:
        raise ValueError("titles must contain exactly four labels when provided.")

    if not isinstance(y_lim[0], (int, float)) and len(y_lim) != 4:
        raise ValueError("y_lim must be one shared tuple or exactly four per-panel tuples.")

    data_sets = plot_metric_folders(
        directories=directories,
        save_path=save_path,
        titles=titles,
        metric="accuracy",
        figsize=figsize,
        x_ticks=x_ticks,
        y_lim=y_lim,
        legend_ncol=legend_ncol,
        legend_loc=legend_loc,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        plot_raw=True,
        plot_smooth=True,
        **kwargs,
    )

    if return_data:
        return data_sets
    return None


def plot_eight_smoothed_datasets(
    directories: Sequence[Union[str, Path]],
    save_path: Optional[Union[str, Path]] = None,
    titles: Optional[Sequence[str]] = None,
    figsize: tuple = (20, 10),
    x_ticks: Optional[Sequence[float]] = (0, 25, 50, 75, 100),
    y_lim: Union[tuple, Sequence[tuple]] = (0, 0.8),
    legend_ncol: int = 8,
    legend_loc: str = "upper center",
    legend_bbox_to_anchor: tuple = (0.5, 1.08),
    return_data: bool = False,
    **kwargs,
) -> Optional[List[Dict[str, pd.Series]]]:
    """
    Convenience wrapper for a two-row, eight-panel smoothed accuracy figure.

    Args:
        directories: exactly eight data folders or csv paths.
        save_path: output path. If no suffix is given, PDF is used.
        titles: exactly eight subplot titles.
        figsize: whole figure size, e.g. ``(20, 10)``.
        x_ticks: x-axis tick locations. Defaults to ``(0, 25, 50, 75, 100)``.
        y_lim: either one shared ``(min, max)`` tuple or eight per-panel tuples.
        legend_ncol: number of columns in the shared legend.
        legend_loc: legend location passed to matplotlib, e.g. ``"lower center"``.
        legend_bbox_to_anchor: legend anchor, e.g. ``(0.5, -0.03)``.
        return_data: if True, return the data dictionaries used for plotting.
    """
    directories = list(directories)
    if len(directories) != 8:
        raise ValueError("plot_eight_smoothed_datasets requires exactly eight directories.")

    if titles is not None and len(titles) != 8:
        raise ValueError("titles must contain exactly eight labels when provided.")

    if not isinstance(y_lim[0], (int, float)) and len(y_lim) != 8:
        raise ValueError("y_lim must be one shared tuple or exactly eight per-panel tuples.")

    data_sets = plot_metric_folders(
        directories=directories,
        save_path=save_path,
        titles=titles,
        metric="accuracy",
        figsize=figsize,
        n_cols=4,
        x_ticks=x_ticks,
        y_lim=y_lim,
        legend_ncol=legend_ncol,
        legend_loc=legend_loc,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
        plot_raw=True,
        plot_smooth=True,
        **kwargs,
    )

    if return_data:
        return data_sets
    return None


def plot_fedgra_learning_curves(
    files: Union[str, Path, List[Union[str, Path]]],
    metric: str = "accuracy",
    header: int = 1,
    auto_legend: bool = True,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    legend_map: Optional[Dict[str, str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    x_range: Optional[int] = None,
    title: str = "FedGRA Learning Curves",
    x_label: str = "Communication Rounds",
    y_label: Optional[str] = None,
    y_lim: tuple = (0, 1),
    window_size: int = 10,
    figsize: tuple = (5, 5),
    is_legend: bool = True,
    save_path: Optional[str] = None,
    plot_raw: bool = True,
    raw_alpha: float = 0.35,
    raw_linewidth: float = 0.7,
    smooth_linewidth: float = 2,
    dpi: int = 800,
    fontsize_title: int = 22,
    fontsize_label: int = 24,
    fontsize_legend: int = 18,
    fontsize_tick: int = 20,
    sort_files: bool = True,
    **read_csv_kwargs,
) -> Dict[str, pd.Series]:
    """
    Parse FedGRA csv files and plot learning curves.

    Features:
    1) Auto-read method from csv config header for legend
    2) Supports manual labels / legend_map override
    3) Uses the same plotting controls as plot_learning_curves

    Returns:
        data_dict used for plotting, useful for further custom analysis.
    """
    data_dict = build_fedgra_metric_dict(
        files=files,
        metric=metric,
        header=header,
        auto_legend=auto_legend,
        labels=labels,
        legend_map=legend_map,
        sort_files=sort_files,
        **read_csv_kwargs,
    )

    if y_label is None:
        y_label = metric.replace("_", " ").title()

    plot_learning_curves(
        data_dict=data_dict,
        x_range=x_range,
        colors=colors,
        labels=list(data_dict.keys()),
        legend_map=None,
        color_map=color_map,
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_lim=y_lim,
        window_size=window_size,
        figsize=figsize,
        is_legend=is_legend,
        save_path=save_path,
        plot_raw=plot_raw,
        raw_alpha=raw_alpha,
        raw_linewidth=raw_linewidth,
        smooth_linewidth=smooth_linewidth,
        dpi=dpi,
        fontsize_title=fontsize_title,
        fontsize_label=fontsize_label,
        fontsize_legend=fontsize_legend,
        fontsize_tick=fontsize_tick,
    )

    return data_dict


if __name__ == "__main__":
    files = [
        "/mnt/data/train-20260211_173559-eb4fea9f.csv",
        "/mnt/data/train-20260212_091233-acde.csv",
    ]

    df = read_training_csv(
        files,
        columns=["round", "client_id", "loss", "train_time"],
    )
