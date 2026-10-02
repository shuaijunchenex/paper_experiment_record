"""Reproduce the RBLA+ scaling table and figure from verified archived CSVs."""
from pathlib import Path
import csv
import hashlib
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / 'rbla+/experiment/09_rank_scale_results/accuracy_comparison.json'
ORDER = ['cifar10', 'cinic10', 'svhn', 'agnews', 'imdb', 'mnli', 'qqp', 'rte', 'sst2']
LABELS = dict(zip(ORDER, ['CIFAR10', 'CINIC10', 'SVHN', 'AG News', 'IMDb', 'MNLI', 'QQP', 'RTE', 'SST-2']))
rows = json.loads(SOURCE.read_text(encoding='utf-8'))
paired = {(r['dataset'], r['rank_assignment']): r for r in rows
          if r['with_rank_scale'] and r['without_rank_scale']}
assert len(paired) == 18
provenance = []
for key, record in paired.items():
    for side in ('without_rank_scale', 'with_rank_scale'):
        run = record[side]
        path = (SOURCE.parent / run['archive_path']).resolve()
        raw = path.read_bytes()
        lines = raw.decode('utf-8-sig').splitlines()
        header = next(i for i, line in enumerate(lines) if line.startswith('round,accuracy,'))
        data = list(csv.DictReader(io.StringIO('\n'.join(lines[header:]))))
        valid = [r for r in data if r.get('evaluation_status', 'evaluated') == 'evaluated']
        assert [int(r['round']) for r in valid] == list(range(101))
        peak = max(valid, key=lambda r: float(r['accuracy']))
        assert abs(float(peak['accuracy']) - run['peak_accuracy']) < 1e-12
        assert int(peak['round']) == run['peak_round']
        provenance.append({'dataset': key[0], 'rank_assignment': key[1], 'variant': side,
                           'path': path.relative_to(ROOT).as_posix(),
                           'sha256': hashlib.sha256(raw).hexdigest(),
                           'peak_accuracy': run['peak_accuracy'], 'peak_round': run['peak_round']})
    delta = 100 * (record['with_rank_scale']['peak_accuracy'] - record['without_rank_scale']['peak_accuracy'])
    assert abs(delta - record['delta_percentage_points']) < 1e-12

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.unicode_minus': False, 'savefig.facecolor': 'white'})

def values(dataset, rank):
    row = paired[dataset, rank]
    return (100 * row['without_rank_scale']['peak_accuracy'],
            100 * row['with_rank_scale']['peak_accuracy'], row['delta_percentage_points'])

tex = [r'% Single column in a two-column paper. Requires \usepackage{booktabs}', r'\begin{table}[t]', r'  \centering',
       r'  \caption{Peak accuracy (\%) of RBLA+ with and without rank scaling under Dirichlet $\alpha=0.4$.}',
       r'  \label{tab:rbla-rank-scale-accuracy}', r'  \footnotesize', r'  \setlength{\tabcolsep}{3pt}',
       r'  \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrr@{}}', r'    \toprule',
       r'    Dataset & w/o scale & w/ scale & $\Delta$ (pp) \\',
       r'    \midrule']
table_cells = []
for rank in ('correct', 'mismatch'):
    if rank == 'mismatch':
        tex.append(r'    \midrule')
    label = 'Rank-correct' if rank == 'correct' else 'Rank-mismatch'
    tex.append(r'    \multicolumn{4}{c}{\textit{' + label + r'}} \\')
    tex.append(r'    \midrule')
    for dataset in ORDER:
        cells = [LABELS[dataset]]
        old, new, delta = values(dataset, rank)
        old_s, new_s = f'{old:.2f}', f'{new:.2f}'
        if old > new:
            old_s = r'\textbf{' + old_s + '}'
        elif new > old:
            new_s = r'\textbf{' + new_s + '}'
        cells += [old_s, new_s, f'{delta:+.2f}']
        table_cells.append(cells)
        tex.append('    ' + ' & '.join(cells) + r' \\')
tex += [r'    \bottomrule', r'  \end{tabular*}', r'  \par\smallskip',
        r'  \parbox{\columnwidth}{\scriptsize',
        r'  Values are the maximum server-model accuracy over logged rounds 0--100 of complete runs, not seed averages. Bold marks the higher accuracy within each dataset and rank assignment. $\Delta=\mathrm{ACC}_{\mathrm{w/\ scale}}-\mathrm{ACC}_{\mathrm{w/o\ scale}}$ is computed before rounding and reported in percentage points (pp). Incomplete runs and the separate broadcast ablation are excluded. MNIST, Fashion-MNIST, KMNIST, QMNIST, and MRPC lack paired scaling results and are omitted.}',
        r'\end{table}']
(OUT / 'rank_scale_accuracy.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')

# Preview rendered from the same cells, not a compiled LaTeX document.
fig, ax = plt.subplots(figsize=(3.45, 5.55))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis('off')
ax.text(0, .985, 'RBLA+ rank scaling: peak accuracy', fontsize=9, weight='bold', va='top')
ax.text(0, .948, r'Dirichlet $\alpha=0.4$; ACC (%)', fontsize=8)
xs = [.005, .45, .69, .95]
ax.hlines(.93, 0, 1, color='black', lw=.7)
headers = ['Dataset', 'w/o scale', 'w/ scale', 'Delta (pp)']
for x, header in zip(xs, headers):
    ax.text(x, .905, header, ha='left' if x == xs[0] else 'center', fontsize=8)
ax.hlines(.891, 0, 1, color='black', lw=.5)
for index, cells in enumerate(table_cells):
    group, local = divmod(index, 9)
    group_top = .864 - group * .336
    if local == 0:
        ax.text(.5, group_top, 'Rank-correct' if group == 0 else 'Rank-mismatch', ha='center', fontsize=8, style='italic')
        ax.hlines(group_top-.012, 0, 1, color='black', lw=.4)
    y = group_top - .042 - local * .030
    for x, cell in zip(xs, cells):
        bold = cell.startswith(r'\textbf{')
        text = cell[8:-1] if bold else cell
        ax.text(x, y, text, ha='left' if x == xs[0] else 'center',
                weight='bold' if bold else 'normal', fontsize=8)
ax.hlines(.230, 0, 1, color='black', lw=.7)
notes = ['Peak server ACC over rounds 0-100 of complete runs.',
         'Bold: higher ACC. Delta: with scale - without scale (pp),',
         'calculated before rounding. Values are not seed averages.',
         'Partial runs and broadcast ablations are excluded.',
         'MNIST, Fashion-MNIST, KMNIST, QMNIST and MRPC',
         'lack paired scaling results and are omitted.']
for i, note in enumerate(notes):
    ax.text(0, .205-i*.026, note, fontsize=6.8, va='top')
fig.subplots_adjust(left=.045, right=.95, top=.99, bottom=.015)
fig.savefig(OUT / 'rank_scale_accuracy.pdf')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.45, 4.15))
for rank, color, offset in zip(('correct', 'mismatch'), ['#276493', '#B56A28'], [-.19, .19]):
    delta = [values(dataset, rank)[2] for dataset in ORDER]
    ys = np.arange(len(ORDER)) + offset
    ax.barh(ys, delta, height=.33, color=color, zorder=3,
            label='Rank-correct' if rank == 'correct' else 'Rank-mismatch')
    for y, value in zip(ys, delta):
        ax.text(value + (.09 if value >= 0 else -.09), y, f'{value:+.2f}',
                va='center', ha='left' if value >= 0 else 'right', fontsize=7)
ax.axvline(0, color='#222222', lw=.7, zorder=2)
ax.axhline(2.5, color='#D0D0D0', lw=.5)
ax.set_axisbelow(True)
ax.grid(axis='x', color='#E5E5E5', lw=.5)
ax.set_xlim(-1.70, 5.60)
ax.set_xticks([-1, 0, 1, 2, 3, 4, 5])
ax.set_yticks(np.arange(len(ORDER)), [LABELS[dataset] for dataset in ORDER])
ax.tick_params(axis='both', length=0, pad=4, labelsize=8)
for spine in ('top', 'right', 'left'):
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#A0A0A0')
ax.invert_yaxis()
ax.legend(loc='lower center', bbox_to_anchor=(.40, 1.005), ncol=2, frameon=False,
          fontsize=7.5, handlelength=1.1, columnspacing=1.0)
ax.set_xlabel('Peak ACC difference (pp)\nwith scale - without scale', fontsize=8, labelpad=5)
fig.subplots_adjust(left=.215, right=.985, bottom=.15, top=.925)
fig.savefig(OUT / 'rank_scale_acc_difference.pdf')
fig.savefig(OUT / 'rank_scale_acc_difference.png', dpi=300)
plt.close(fig)

(OUT / 'rank_scale_difference_figure.tex').write_text(r'''% Requires \usepackage{graphicx}. Adjust the PDF path for your manuscript.
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{rank_scale_acc_difference.pdf}
  \caption{Peak ACC difference (with scale minus without scale) for RBLA+ under Dirichlet $\alpha=0.4$. Bars compare complete runs within each dataset and rank assignment. Positive values favor scaling; pp denotes percentage points.}
  \label{fig:rbla-rank-scale-difference}
\end{figure}
''', encoding='utf-8')
metadata = {'source': SOURCE.relative_to(ROOT).as_posix(),
            'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'source_files': provenance, 'paired_comparisons': len(paired),
            'latex_compiled_locally': False, 'table_preview_renderer': 'Matplotlib, identical formatted cells',
            'delta_unit': 'percentage points', 'precision': 2, 'layout': 'single column',
            'figure_width_inches': 3.45,
            'config_differences': [{'dataset': r['dataset'], 'rank_assignment': r['rank_assignment'],
                                    'details': r['recorded_config_differences']} for r in paired.values()
                                   if r['recorded_config_differences']]}
(OUT / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(OUT / 'README.md').write_text('''# RBLA+ RankScale accuracy comparison

- `rank_scale_accuracy.tex`: single-column booktabs table, with correct/mismatch sections stacked vertically and paired winners in bold. Uses `table` and `tabular*` at `columnwidth`. Requires `booktabs`.
- `rank_scale_accuracy.pdf`: visual preview of the same formatted values. Rendered with Matplotlib; no local LaTeX compiler was available, so it is not a compiled TeX preview.
- `rank_scale_acc_difference.pdf`: 3.45-inch-wide single-column vector figure, with two grouped bars per dataset and a shared axis.
- `rank_scale_acc_difference.png`: high-resolution image of the figure.
- `rank_scale_difference_figure.tex`: figure inclusion and caption. Requires `graphicx`.
- `metadata.json`: source paths, SHA-256 hashes, peak rounds and recorded configuration differences.

Both variants use complete server-model training logs, Dirichlet alpha=0.4, grouped by dataset and rank assignment. ACC is the maximum over logged rounds 0-100. Differences are calculated before rounding as with scale minus without scale (percentage points). No partial runs, broadcast metrics, or September 16 broadcast-ablation runs are included. Missing scaling results for MNIST, Fashion-MNIST, KMNIST, QMNIST and MRPC are explicitly noted rather than treated as zeros. These are not multi-seed averages. The historical comparison alone does not isolate all implementation effects. MNLI correct has slightly different recorded rank ratios; see metadata.

Reproduce with `python -B generate.py` using Python, NumPy and Matplotlib. Original CSVs are read and their peak values and rounds verified before generation. To use in a paper, copy the table snippet into the manuscript; place the figure PDF alongside the figure snippet or adjust its relative path.
''', encoding='utf-8')
print(f'Generated table and difference plot for {len(paired)} verified pairs in {OUT}')
