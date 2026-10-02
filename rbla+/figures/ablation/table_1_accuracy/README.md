# 四组合准确率汇总

最终准确率使用 update 101；末十轮均值使用 updates 92–101。都为十个客户端广播模型的非加权平均准确率，单种子 42，不混入完整服务器模型。粗体标记每个数据集、每个指标的最高值。

## 输出

- `accuracy_summary.tex`：可直接放入论文的 `table*` 环境，使用 booktabs 三线表，只依赖 `\usepackage{booktabs}`。
- `accuracy_summary.pdf`：Times New Roman 矢量预览，直接渲染与 LaTeX 相同的汇总单元格。当前环境没有 LaTeX 编译器，此 PDF 由 ReportLab 生成，`.tex` 未在本机编译。
- `accuracy_summary.csv`：未四舍五入的数值汇总；表 1 另附 1212 个逐轮数据点 `accuracy_curves.csv`。
- `accuracy_summary.md`：便于阅读和复制的 Markdown 版本。
- `build_table.py` / `table_utils.py`：可复现脚本。
- `sources/`：对应原始结果的逐字节副本及 SHA-256。
- `table_metadata.json`：统计口径、输入溯源和环境说明。
- `qa/`：数值检查和最终 PDF 检查记录。

## 重现

在本目录运行 `python -B build_table.py`。需要 Python、NumPy、pandas、ReportLab，以及 `C:/Windows/Fonts/times.ttf` 和 `timesbd.ttf`。生成过程不执行新实验，也不修改原始输入。所有已显示数字由冻结来源计算。

表中最优值按未四舍五入的数据判断。末十轮平均与最终值同时报告；未按最高测试准确率选择轮次。

原始批次：`rbla+/ablation/run_all_20260915_223408_556671`。原数据和已有三张图保持不变。
