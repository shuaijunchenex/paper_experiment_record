from typing import List, Union, Dict, Optional
import json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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
    
    # Set default colors
    default_colors = ["red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta", "yellow", "black"]
    if colors is None:
        colors = default_colors[:len(data_dict)]
    elif len(colors) < len(data_dict):
        # If not enough colors provided, cycle through them
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
        plt.savefig(f"{save_path}.pdf", dpi=dpi, bbox_inches='tight')
    
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


def plot_fedgra_learning_curves(
    files: Union[str, Path, List[Union[str, Path]]],
    metric: str = "accuracy",
    header: int = 1,
    auto_legend: bool = True,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    legend_map: Optional[Dict[str, str]] = None,
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
