import matplotlib.pyplot as plt

class MatplotlibStyle:
    """
    A helper class to apply consistent matplotlib styling across plots.
    """

    def __init__(self):
        self.custom_colors = ['r', 'g', 'b', '#d62728', '#9467bd', '#8c564b',
                              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def apply(self):
        """
        Apply the global matplotlib style settings.
        """
        plt.rcParams.update({
            # === 字体 ===
            "font.family": "Times New Roman",
            "font.size": 16,

            # === 坐标轴 ===
            "axes.linewidth": 1.0,
            "axes.labelsize": 24,
            "axes.titlesize": 24,
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "axes.titleweight": "normal",
            "axes.titlepad": 12,
            "axes.labelpad": 8,
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.xmargin": 0.02,
            "axes.ymargin": 0.02,
            "axes.grid": True,
            "axes.axisbelow": False,

            # === 去掉 top/right 边框和刻度 ===
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.top": False,
            "ytick.right": False,

            # === 刻度 ===
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 6,
            "xtick.minor.size": 4,
            "ytick.major.size": 6,
            "ytick.minor.size": 4,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "xtick.direction": "out",
            "ytick.direction": "out",

            # === 线条 ===
            "lines.linewidth": 1.5,
            "lines.markersize": 6,

            # === 网格 ===
            "grid.linestyle": ":",
            "grid.alpha": 0.3,
            "grid.color": "gray",
            "grid.linewidth": 0.5,

            # === 图例 (无边框) ===
            "legend.fontsize": 18,
            "legend.frameon": False,
            "legend.borderpad": 0.5,
            "legend.labelspacing": 0.5,
            "legend.handlelength": 1.5,

            # === 图像 ===
            "figure.figsize": [5, 5],
            "figure.dpi": 800,
            "savefig.dpi": 800,

            # === PDF 字体嵌入 ===
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=self.custom_colors)

    @staticmethod
    def reset():
        """
        Reset to matplotlib default style.
        """
        plt.rcParams.update(plt.rcParamsDefault)
