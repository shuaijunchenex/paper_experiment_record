import matplotlib.pyplot as plt

class MatplotlibStyle:
    """
    A helper class to apply consistent matplotlib styling across plots.
    """

    def __init__(self):
        # 自定义颜色循环
        self.custom_colors = [
            (1, 100/255, 80/255),       # 红色 (偏粉红)
            "g",                        # 默认绿色
            "dodgerblue",               # 蓝色
            (34/255, 139/255, 34/255)   # 深绿色 (ForestGreen)
        ]

    def apply(self):
        """
        Apply the global matplotlib style settings.
        """
        plt.rcParams.update({
            "font.family": "Times New Roman",       # 字体
            "axes.labelsize": 24,                   # xlabel 和 ylabel 默认字号
            "axes.titlesize": 24,                   # 标题字号
            "xtick.labelsize": 20,                  # x 轴刻度字体大小
            "ytick.labelsize": 20,                  # y 轴刻度字体大小
            "legend.fontsize": 18,                  # 图例字体大小
            "axes.edgecolor": "black",              # 坐标轴颜色
            "axes.linewidth": 1.2                   # 坐标轴线宽
        })
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=self.custom_colors)

    @staticmethod
    def reset():
        """
        Reset to matplotlib default style.
        """
        plt.rcParams.update(plt.rcParamsDefault)
