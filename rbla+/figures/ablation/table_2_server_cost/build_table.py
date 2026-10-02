"""Measured CPU server cost, with explicitly nested timing statistics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from table_utils import DATASETS, LABELS, verify_sources, render_pdf, write_metadata, value_text

ROOT = Path(__file__).resolve().parent
STEM = "server_cost_summary"
METHODS = [("rbla", "RBLA"), ("rbla_plus", "RBLA+"), ("dense_svd", "Dense SVD")]
NOTES = ("Entries are median [IQR], where IQR = Q3 - Q1, across eight checkpoint inputs per dataset "
         "(two training paths x logged rounds 0, 9, 49, 100). Each input's time is the median of 20 "
         "warm repetitions after five warmups. CPU, FP32, one PyTorch thread. Memory is cold-pass "
         "peak incremental CPU RSS above the resident-input baseline, sampled every 1 ms; it is "
         "not GPU memory. Positive IQRs that round to 0.00 are shown as <0.01. Server step includes "
         "aggregation and all requested broadcasts. File I/O, serialization, device transfers, "
         "model loading and evaluation are excluded. Dense SVD constructs the dense matrix, "
         "uses economy full SVD, then truncates; it is not an optimized truncated-SVD solver.")
COMPACT_NOTES = ("Median [IQR] across eight checkpoint inputs; each time is the median of 20 runs "
                 "after five warmups. CPU FP32, one thread. ΔRSS is cold-pass peak incremental CPU "
                 "RSS above resident inputs (1 ms sampling). Full step includes aggregation and "
                 "all requested broadcasts; excludes I/O, transfers, model loading and evaluation. "
                 "Dense SVD uses economy full SVD then truncation. Positive IQRs rounding to zero "
                 "appear as <0.01.")


def main():
    provenance = verify_sources(ROOT)
    data = pd.read_csv(ROOT / "sources/measurements.csv", float_precision="round_trip")
    manifest = json.loads((ROOT / "sources/manifest.json").read_text(encoding="utf-8"))
    assert len(data) == 144
    for key, expected in [("device", "cpu"), ("dtype", "float32"), ("threads", 1), ("warmup", 5), ("repeats", 20)]:
        assert (data[key] == expected).all(), key
    assert (data.input_kind == "training_checkpoint").all()
    assert set(data.phase) == {"broadcast", "server_step"}
    assert set(data.method) == {m for m, _ in METHODS}
    assert set(data["round"]) == {0, 9, 49, 100}
    for _, row in data.iterrows():
        raw = np.array(json.loads(row.raw_seconds))
        assert len(raw) == 20 and np.isfinite(raw).all() and (raw > 0).all()
        np.testing.assert_allclose(np.median(raw), row.median_seconds, rtol=1e-12, atol=1e-12)
    records = []
    for ds in DATASETS:
        for method, _ in METHODS:
            for phase in ["broadcast", "server_step"]:
                rows = data[(data.dataset == ds) & (data.method == method) & (data.phase == phase)]
                assert len(rows) == 8 and rows.input_sha256.nunique() == 8
                assert set(zip(rows.driver_method, rows["round"])) == {
                    (driver, r) for driver in ["rbla", "rbla_plus_rank_scale"] for r in [0, 9, 49, 100]}
                record = {"dataset": ds, "method": method, "phase": phase, "input_count": 8}
                for key, scale, name in [("median_seconds", 1000., "time_ms"),
                                          ("cold_rss_peak_incremental_bytes", 1. / 2**20, "cold_peak_increment_mib")]:
                    values = rows[key].to_numpy(dtype=float) * scale
                    assert np.isfinite(values).all() and (values >= 0).all()
                    q1, median, q3 = np.quantile(values, [.25, .5, .75], method="linear")
                    record.update({name + "_median": median, name + "_q1": q1,
                                   name + "_q3": q3, name + "_iqr": q3 - q1})
                # A separate metric retains within-input timing variability;
                # it is not the across-checkpoint IQR displayed in the table.
                record["within_input_timing_iqr_median_ms"] = np.median(rows.iqr_seconds) * 1000
                records.append(record)
    summary = pd.DataFrame(records)
    cells, latex_rows, md_rows = [], [], []
    for ds in DATASETS:
        cells.append([LABELS[ds], "", "", "", ""])
        if ds != DATASETS[0]:
            latex_rows.append(r"    \midrule")
        latex_rows.append(r"    \multicolumn{5}{@{}l}{\textbf{" + LABELS[ds] + r"}} \\")
        for index, (method, label) in enumerate(METHODS):
            values = []
            for phase in ["broadcast", "server_step"]:
                row = summary[(summary.dataset == ds) & (summary.method == method) & (summary.phase == phase)].iloc[0]
                for metric in ["time_ms", "cold_peak_increment_mib"]:
                    values.append(f"{value_text(row[metric+'_median'])} [{value_text(row[metric+'_iqr'])}]")
            cells.append([label, *values])
            latex_values = [v.replace("<0.01", r"$<0.01$") for v in values]
            latex_rows.append("    " + " & ".join([label, *latex_values]) + r" \\")
            md_rows.append("| " + " | ".join([LABELS[ds], label, *values]) + " |")
    latex = r"""% Requires \usepackage{booktabs}
\begin{table}[t]
  \centering
  \caption{Measured CPU server cost: median [IQR].}
  \label{tab:ablation-server-cost}
  \footnotesize
  \setlength{\tabcolsep}{2pt}
  \renewcommand{\arraystretch}{1.12}
  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}
    \toprule
    Method & \multicolumn{2}{c}{Broadcast} & \multicolumn{2}{c}{Full step} \\
    \cmidrule(lr){2-3}\cmidrule(l){4-5}
     & Time (ms) & $\Delta$RSS (MiB) & Time (ms) & $\Delta$RSS (MiB) \\
    \midrule
""" + "\n".join(latex_rows) + r"""
    \bottomrule
  \end{tabular*}
  \par\smallskip
  \parbox{\columnwidth}{\footnotesize
  Median [IQR] across eight checkpoint inputs; each time is the median of 20 runs after five warmups. CPU FP32, one thread. $\Delta$RSS is cold-pass peak incremental CPU RSS above resident inputs (1\,ms sampling). Full step includes aggregation and all requested broadcasts; excludes I/O, transfers, model loading and evaluation. Dense SVD uses economy full SVD then truncation. Positive IQRs rounding to zero appear as $<0.01$.}
\end{table}
"""
    (ROOT / f"{STEM}.tex").write_text(latex, encoding="utf-8")
    md = ("# Table 2. Measured server cost\n\n"
          "| Dataset | Method | Broadcast time (ms) | Broadcast peak incremental RSS (MiB) | Full-step time (ms) | Full-step peak incremental RSS (MiB) |\n"
          "|---|---|---:|---:|---:|---:|\n" + "\n".join(md_rows) + "\n\n" + NOTES + "\n")
    (ROOT / f"{STEM}.md").write_text(md, encoding="utf-8")
    summary.to_csv(ROOT / f"{STEM}.csv", index=False, float_format="%.17g")
    render_pdf(ROOT, STEM, "Table 2. Measured CPU server cost: median [IQR]",
               [["Method", "Broadcast", "", "Full step", ""],
                ["", "Time (ms)", "ΔRSS (MiB)", "Time (ms)", "ΔRSS (MiB)"]],
               cells, [44, 48, 50, 50, 54], COMPACT_NOTES,
               spans=[(1, 2), (3, 4)], group_rows=[0, 4, 8])
    write_metadata(ROOT, provenance, {"table": 2, "rows": 9, "summary_records": 18,
                                     "raw_measurements": 144, "notes": NOTES,
                                     "quantile_method": "linear", "display_decimals": 2,
                                     "environment": manifest["environment"],
                                     "measurement_boundary": manifest["boundary"],
                                     "layout": "single-column, five columns with dataset group rows",
                                     "latex_float": "table", "latex_width": "columnwidth",
                                     "preview_width_inches": 3.5, "preview_body_font_pt": 8,
                                     "compact_notes": COMPACT_NOTES,
                                     "within_input_timing_iqr": "retained separately in CSV; not displayed as across-input IQR"})
    print(ROOT / f"{STEM}.tex")


if __name__ == "__main__":
    main()
