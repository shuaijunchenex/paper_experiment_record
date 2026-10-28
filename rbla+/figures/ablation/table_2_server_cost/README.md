# 服务器耗时与峰值增量内存

每项显示中位数 [IQR]，IQR 是八个输入之间的 Q3−Q1。每个输入的耗时先取 20 次预热后重复计时的中位数；内存使用单独 cold pass 的峰值增量 CPU RSS。该内存指标不是 GPU 显存，也不是总进程占用。

## 输出

- `server_cost_summary.tex`：双栏论文中的单列 `table` 环境，表格及表注均使用 `\columnwidth`。采用数据集分组行、五列布局、`\footnotesize` 和 2 pt 列间距，不使用整体缩放；只依赖 `\usepackage{booktabs}`。
- `server_cost_summary.pdf`：Times New Roman 矢量预览，直接渲染与 LaTeX 相同的汇总单元格。当前环境没有 LaTeX 编译器，此 PDF 由 ReportLab 生成，`.tex` 未在本机编译。
- `server_cost_summary.csv`：未四舍五入的数值汇总，数值及统计口径与原宽表完全一致。
- `server_cost_summary.md`：便于阅读和复制的 Markdown 版本。
- `build_table.py` / `table_utils.py`：可复现脚本。
- `sources/`：对应原始结果的逐字节副本及 SHA-256。
- `table_metadata.json`：统计口径、输入溯源和环境说明。
- `qa/`：数值检查和最终 PDF 检查记录。

## 重现

在本目录运行 `python -B build_table.py`。需要 Python、NumPy、pandas、ReportLab，以及 `C:/Windows/Fonts/times.ttf` 和 `timesbd.ttf`。生成过程不执行新实验，也不修改原始输入。所有已显示数字由冻结来源计算。

表 2 的 IQR 是跨 checkpoint 输入的分布宽度，不是 20 次计时重复的 IQR，也不是置信区间。逐输入计时 IQR 保存在原始 measurements.csv 中；这些 IQR 的中位数另载于汇总 CSV 的 within_input_timing_iqr_median_ms 字段。cold RSS 为 1 ms 采样峰值，可能遗漏更短的瞬时峰值。内存分配器及输入状态会影响波动，不能只凭较低中位数声称普遍省内存。

单列 PDF 预览宽 3.5 英寸，正文和表注均为 8 pt Times New Roman。数据集名称改为分组行，保留四项指标的全部中位数 [IQR]，不省略指标或降低数值精度。Markdown 保留横向数据表，便于复制。

每个数据集的八个输入来自两条训练路径及 logged rounds `{0,9,49,100}`。每个输入先预热五次，再计时 20 次；表中汇总这些输入的计时中位数。完整服务器步骤包含聚合和全部请求 rank 的广播，排除文件 I/O、序列化、设备传输、模型加载和评估。Dense SVD 包含 dense 矩阵构造，采用 economy full SVD 后截断，并非优化的 truncated-SVD 求解器。这些完整说明保留在本文件及 metadata，表下注采用紧凑表述。

原始批次：`rbla+/ablation/run_all_20260915_223408_556671`。原数据和已有三张图保持不变。
