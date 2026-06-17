import importlib

import plot_functions as _plot_functions

_plot_functions = importlib.reload(_plot_functions)

plot_accuracy_from_folder = _plot_functions.plot_accuracy_from_folder
plot_combined_smoothed_datasets = _plot_functions.plot_combined_smoothed_datasets
plot_four_smoothed_datasets = _plot_functions.plot_four_smoothed_datasets
plot_metric_folders = _plot_functions.plot_metric_folders


__all__ = [
    "plot_accuracy_from_folder",
    "plot_combined_smoothed_datasets",
    "plot_four_smoothed_datasets",
    "plot_metric_folders",
]


if __name__ == "__main__":
    plot_combined_smoothed_datasets(
        "fedgra/aws/mnist",
        "fedgra/aws/fmnist",
        label1="AWS MNIST",
        label2="AWS FMNIST",
        save_path="fedgra/aws_combined_smoothed.pdf",
        y_lim=(0, 0.8),
        method_order=[
            "adafl",
            "afl",
            "fedgra",
            "fedsdr",
            "oort",
            "pow-d",
            "pyramidfl",
            "repufl",
        ],
    )
