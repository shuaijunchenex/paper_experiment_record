"""Final and last-ten-update broadcast accuracy; one fixed seed."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from table_utils import DATASETS, LABELS, verify_sources, render_pdf, write_metadata

ROOT = Path(__file__).resolve().parent
STEM = "accuracy_summary"
COMBINATIONS = [("RBLA", "RBLA"), ("RBLA", "RBLA+"), ("RBLA+", "RBLA"), ("RBLA+", "RBLA+")]
NOTES = ("Accuracy (%), unweighted mean over ten client broadcast models. Final: update 101 "
         "(logged round 100). Last 10: mean of updates 92-101 (logged rounds 91-100). "
         "One seed (42), correct rank assignment, Dirichlet alpha=0.4. Cross-broadcast evaluation "
         "does not change the training path. RBLA+ includes bidirectional RankScale. "
         "Bold marks the highest value in each dataset/metric column. Last 10 is a temporal mean, "
         "not an average over seeds. Full-width server-model accuracy is excluded.")


def main():
    provenance = verify_sources(ROOT)
    records, curve_records = [], []
    for path in sorted((ROOT / "sources").glob("*-broadcast-accuracy.csv")):
        raw = path.read_text(encoding="utf-8-sig")
        header, body = raw.split("\n\n", 1)
        config = json.loads(header.split("\n", 1)[0].removeprefix("Config,"))
        frame = pd.read_csv(io.StringIO(body), float_precision="round_trip")
        assert config["seed"] == 42
        assert frame["round"].tolist() == list(range(101))
        assert frame.update_index.tolist() == list(range(1, 102))
        assert (frame.client_count == 10).all()
        for _, row in frame.iterrows():
            actual = np.mean(list(json.loads(row.client_accuracies).values()))
            np.testing.assert_allclose(actual, row.accuracy, rtol=0, atol=1e-12)
        first, last = frame.iloc[0], frame.iloc[-1]
        records.append({"dataset": path.name.split("_")[0], "training_path": first.driver_method,
                        "broadcast": first.evaluated_method, "final_accuracy_percent": 100 * last.accuracy,
                        "last10_accuracy_percent": 100 * frame.tail(10).accuracy.mean(),
                        "final_update": 101, "last10_start_update": 92, "last10_end_update": 101,
                        "seed": 42, "client_count": 10, "source_file": path.name})
        for _, row in frame.iterrows():
            curve_records.append({"dataset": path.name.split("_")[0], "training_path": first.driver_method,
                                  "broadcast": first.evaluated_method, "update_index": row.update_index,
                                  "accuracy_percent": 100 * row.accuracy})
    summary = pd.DataFrame(records)
    assert len(summary) == 12
    cells, latex_rows, md_rows, bold_cells = [], [], [], []
    for index, (driver, broadcast) in enumerate(COMBINATIONS):
        values, latex_values, md_values = [], [], []
        for ds in DATASETS:
            row = summary[(summary.dataset == ds) & (summary.training_path == driver) & (summary.broadcast == broadcast)].iloc[0]
            for metric in ["final_accuracy_percent", "last10_accuracy_percent"]:
                text = f"{row[metric]:.2f}"
                best = row[metric] == summary.loc[summary.dataset == ds, metric].max()
                if best:
                    bold_cells.append((len(values) + 2, index))
                values.append(text)
                latex_values.append(r"\textbf{" + text + "}" if best else text)
                md_values.append(f"**{text}**" if best else text)
        cells.append([driver, broadcast, *values])
        latex_rows.append(" & ".join([driver, broadcast, *latex_values]) + r" \\")
        md_rows.append("| " + " | ".join([driver, broadcast, *md_values]) + " |")
    latex = r"""% Requires \usepackage{booktabs}
\begin{table*}[t]
  \centering
  \caption{Training-path and broadcast ablation. Accuracy (\%) at the fixed final update and averaged over the last ten updates.}
  \label{tab:ablation-accuracy}
  \small
  \setlength{\tabcolsep}{6pt}
  \begin{tabular}{@{}llrrrrrr@{}}
    \toprule
    Training path & Broadcast & \multicolumn{2}{c}{CIFAR10} & \multicolumn{2}{c}{MNLI} & \multicolumn{2}{c}{QQP} \\
    \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}
     & & Final & Last 10 & Final & Last 10 & Final & Last 10 \\
    \midrule
""" + "\n".join("    " + row for row in latex_rows) + r"""
    \bottomrule
  \end{tabular}
  \par\smallskip
  \parbox{\textwidth}{\footnotesize
  Values are the unweighted mean accuracy of ten client broadcast models. Final uses update 101 (logged round 100); Last 10 averages updates 92--101 (logged rounds 91--100). All runs use seed 42, correct rank assignment and Dirichlet $\alpha=0.4$. Cross-broadcast evaluation does not change the training path. RBLA+ includes bidirectional RankScale. Bold indicates the highest value in each dataset/metric column. Temporal averages are not averages over seeds. Full-width server-model accuracy is excluded.}
\end{table*}
"""
    (ROOT / f"{STEM}.tex").write_text(latex, encoding="utf-8")
    md = ("# Table 1. Training-path and broadcast accuracy\n\n"
          "| Training path | Broadcast | CIFAR10 final | CIFAR10 last 10 | MNLI final | MNLI last 10 | QQP final | QQP last 10 |\n"
          "|---|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(md_rows) + "\n\n" + NOTES + "\n")
    (ROOT / f"{STEM}.md").write_text(md, encoding="utf-8")
    summary.to_csv(ROOT / f"{STEM}.csv", index=False, float_format="%.17g")
    pd.DataFrame(curve_records).to_csv(ROOT / "accuracy_curves.csv", index=False, float_format="%.17g")
    render_pdf(ROOT, STEM, "Table 1. Training-path and broadcast accuracy (%)",
               [["Training path", "Broadcast", "CIFAR10", "", "MNLI", "", "QQP", ""],
                ["", "", "Final", "Last 10", "Final", "Last 10", "Final", "Last 10"]],
               cells, [84, 78, 57, 57, 57, 57, 57, 57], NOTES,
               spans=[(2, 3), (4, 5), (6, 7)], bold_cells=bold_cells)
    write_metadata(ROOT, provenance, {"table": 1, "rows": 4, "summary_records": 12,
                                     "raw_curve_points": len(curve_records), "notes": NOTES,
                                     "display_decimals": 2, "bold_selection": "highest unrounded value per dataset/metric"})
    print(ROOT / f"{STEM}.tex")


if __name__ == "__main__":
    main()
