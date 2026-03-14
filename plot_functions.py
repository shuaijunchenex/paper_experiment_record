from typing import List, Union, Dict, Optional
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


if __name__ == "__main__":
    files = [
        "/mnt/data/train-20260211_173559-eb4fea9f.csv",
        "/mnt/data/train-20260212_091233-acde.csv",
    ]

    df = read_training_csv(
        files,
        columns=["round", "client_id", "loss", "train_time"],
    )
