# Reconstruction of real pre-spectral RBLA aggregates

仅使用 RBLA 路径末轮、谱处理前的真实聚合结果。层选择完全由 layer_index 决定：7/17 层中的中间层，不按误差挑选。横轴是实际矩阵因子宽度比例 q/Q，卷积内部宽度不等于逻辑 rank。

## 文件

- `checkpoint_reconstruction.pdf`：唯一图片输出，3.5 × 2.0 英寸矢量母版，论文双栏使用 `width=\columnwidth`。
- `plot_figure.py` 与 `plot_style.py`：可复现脚本与冻结样式，仅读取已归档数据，不运行训练或矩阵分解。
- `sources/`：原始输入的逐字节副本与 SHA-256 溯源信息；绘图前验证。
- `plot_data.csv`：实际使用的数据；图 3 另含 `plot_summary.csv`，保留有符号分位数。
- `plot_metadata.json`：数据选择、统计口径、布局、软件版本和源文件哈希。
- `caption.txt`：完整英文图注；`figure_ieee.tex`：简洁 LaTeX 插入示例。
- `qa/verification.json` 与 `qa/visual_review.md`：PDF 检查与视觉复核记录。

## Layout and style

Single column (3.5 x 2.0 inches): CIFAR10 and MNLI in two horizontal panels with one shared figure-level legend. Times New Roman, 8 pt axis/panel labels and 7.5 pt ticks/legend. Red #C44E52 and blue #4C72B0 are retained. Lines and markers are sized for the final column width. The reconstruction and prefix-quality panels share y-axis labels; accuracy panels retain their different y-axis ranges.

QQP is omitted only from these main-text figures and their derived plotting data. Original sources, experiment archives and separate accuracy tables are preserved. Update remains logged round + 1. The LaTeX snippet uses an unstarred figure and columnwidth.

## 重现

在本文件夹运行 `python -B plot_figure.py`。需要 NumPy、pandas、Matplotlib 及 Times New Roman。脚本覆盖本文件夹内的派生图和绘图明细，不更改 sources 中的原始输入。

实验来源：`rbla+/ablation/run_all_20260915_223408_556671`。训练为单种子 42，correct assignment。原始 `.pt` 尚未同步，但本图直接读取已完成的 CSV 结果，因此重画无需 `.pt`。

Compact revision: dataset labels are below the x-axis labels; legend and axes are tightly spaced. Height is reduced from 2.45 to 2.0 inches without reducing font sizes or changing curve data. Reconstruction uses one horizontal legend with shortened method labels; their definitions are unchanged in the caption.

Panel-label revision: (a) CIFAR10 and (b) MNLI are centered below their respective x-axis labels with a 4 pt gap. Shared legends and the 3.5 x 2.0 inch page are retained. Plotting area is shifted upward to keep the layout compact.
