"""Generate a concise bilingual scientific report from measured data."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def write_report(output: Path):
    config=json.loads((output/'config_frozen.json').read_text(encoding='utf-8'))
    manifest=json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
    validation=json.loads((output/'validation'/'validation_summary.json').read_text(encoding='utf-8'))
    clients=pd.read_csv(output/'client_weighted_summary.csv').set_index('setting')
    by_seed=pd.read_csv(output/'client_weighted_by_seed.csv')
    ranks=pd.read_csv(output/'rank_summary.csv')
    raw=pd.read_csv(output/'raw_all.csv')
    main_id=config['main_setting']
    main_setting=next(s for s in config['settings'] if s['id']==main_id)
    main_seeds=by_seed[by_seed.setting.eq(main_id)]
    generation=(f"shared Gaussian reference factors with independent RMS-scaled perturbations "
                f"(eta={main_setting['eta']:g})" if main_setting['generator']=='shared_reference' else
                'independent Gaussian client factors')
    client_description=(f"{len(main_setting['ranks'])} equally weighted clients"
                        if len(set(main_setting['weights']))==1 else
                        f"{len(main_setting['ranks'])} clients with weights {main_setting['weights']}")
    c=clients.loc[main_id]
    main_ranks=ranks[ranks.setting.eq(main_id)]
    best=main_ranks.loc[main_ranks.gain_median.idxmax()]
    actual=main_ranks[main_ranks.is_client_rank]
    best_client=actual.loc[actual.gain_median.idxmax()]
    reduced=100*c.weighted_total_reduction_median
    english=(f"In a synthetic RBLA aggregation experiment, we generated heterogeneous-rank client LoRA factors "
             f"from {generation}. "
             f"We used {client_description}, "
             f"{main_setting['m']} x {main_setting['n']} updates, R={main_setting['R']}, client ranks "
             f"{main_setting['ranks']}, and {len(config['seeds'])} fixed independent random seeds. "
             f"For each seed, RBLA prefix redistribution and SP+ were evaluated on the same single "
             f"participation-normalized RBLA factor aggregation, without any additional post-aggregation "
             f"gauge transformation and with effective LoRA scaling set to one. Using the unsquared "
             f"Frobenius norm, the median client-weighted relative reconstruction error decreased from "
             f"{c.relative_pre_median:.4f} to {c.relative_sp_median:.4f}; the median paired normalized "
             f"decrease was {c.gain_median:.4f} (IQR {c.gain_q25:.4f}-{c.gain_q75:.4f}), corresponding "
             f"to a {reduced:.2f}% median reduction in client-weighted total reconstruction error. "
             f"Across all {len(config['settings'])} prespecified settings and {manifest['rank_comparisons']:,} rank comparisons, "
             f"the measured SP+ error differed from the optimal rank-r reference by at most "
             f"{manifest['max_abs_normalized_sp_opt_gap']:.2e} after normalization by the full update norm. "
             f"These results support improved rank-constrained reconstruction of synthetic RBLA aggregate "
             f"updates; they do not establish improvements in task accuracy, convergence, fairness, "
             f"aggregation bias, or rank-data mismatch.")
    lines=['# 合成 RBLA 聚合流程实验\n',
           '## 实验口径\n',
           f"预先固定 {len(config['settings'])} 个设置，每个 {len(config['seeds'])} 个种子，合计 "
           f"{manifest['aggregations']} 次配对聚合、{manifest['rank_comparisons']} 条 seed-rank 记录。"
           f"主设置固定为 `{main_id}`，未按结果选择或丢弃种子。种子为 "
           f"{config['seeds'][0]}–{config['seeds'][-1]}。先通过小尺寸独立验证，再运行主设置与补充设置。\n",
           '生成 → 按逐索引参与集合分别平均 B/A → 得到同一对全局因子 → 原始前缀和 SP+ → '
           '相对于 X=BᵍAᵍ 测量。X 不是客户端稠密更新的加权平均。原始索引顺序保留；聚合后没有额外 gauge、旋转、置换或排序。'
           'SP+ 内部 SVD 排序属于方法本身。有效 LoRA scaling=1，所有误差使用未平方 Frobenius 范数。\n',
           '参考 B₀/A₀ 元素分别为 N(0,1/m)、N(0,1/n)。每个参考块 F 的噪声为 '
           'N=Z‖F‖F/√|F|，Z 为独立标准高斯，因此 E‖ηN‖F²=η²‖F‖F²。'
           'η 表示合成因子差异程度，不表示真实 non-IID 程度。同一种子在不同 η 下复用参考和噪声流；'
           '不同种子独立。补充独立生成器直接生成相同逐元素尺度的独立客户端 B/A。\n',
           '## 实际客户端预算下的主要结果\n',
           '先在每个种子内按真实客户端 rᵢ、pᵢ 计算加权误差，再跨种子统计。'
           '下表为中位数；IQR 为第 25–75 百分位，不是置信区间。'
           '“总误差降低比例”是每个种子 (ΣpᵢE_pre−ΣpᵢE_SP+)/ΣpᵢE_pre 的跨种子中位数，'
           '不是所有扫描 rank 的平均，也不是中位数之差的比例。\n',
           '| 设置 | 前缀相对误差 | SP+ 相对误差 | 前缀额外误差 | 配对 gain [IQR] | 总误差降低 |',
           '|---|---:|---:|---:|---:|---:|']
    for setting in config['settings']:
        x=clients.loc[setting['id']]
        lines.append(f"| {setting['id']} | {x.relative_pre_median:.5f} | {x.relative_sp_median:.5f} | "
                     f"{x.excess_pre_median:.5f} | {x.gain_median:.5f} [{x.gain_q25:.5f}, {x.gain_q75:.5f}] | "
                     f"{100*x.weighted_total_reduction_median:.2f}% |")
    lines += ['',f"主设置 {int((main_seeds.gain>0).sum())}/{len(main_seeds)} 个种子的客户端加权 gain 为正；范围 "
              f"{main_seeds.gain.min():.5f}–{main_seeds.gain.max():.5f}。"
              f"前缀的客户端加权归一化额外误差中位数为 {c.excess_pre_median:.5f}，"
              f"SP+ 的对应绝对值最大量级见数值验证。实际 rank 条件下的逐客户端 reduction "
              '在前缀误差数值非零时才定义；r=R 的 reduction 为 NA，而 gain 在容差内为零。'
              '另存 conditional mean reduction 及其有效客户端权重，避免把 NA 当作零。\n',
              '## 哪些 rank 改善较大\n',
              f"诊断扫描中，主设置的中位数 gain 最大位于 r={int(best.r)} "
              f"(r/R={best.normalized_rank:.4f})，gain={best.gain_median:.5f}；"
              f"实际客户端 rank 中最大位于 r={int(best_client.r)}，gain={best_client.gain_median:.5f}。"
              '以下包括最小扫描 rank 及全部实际客户端 rank；完整扫描单独保存。\n',
              '| r | 客户端数 | 前缀相对误差 | SP+ 相对误差 | gain 中位数 | reduction 中位数 |',
              '|---:|---:|---:|---:|---:|---:|']
    for _,x in main_ranks[main_ranks.is_client_rank | main_ranks.r.eq(1)].iterrows():
        reduction='NA' if pd.isna(x.reduction_median) else f'{100*x.reduction_median:.2f}%'
        gain_text=f'{x.gain_median:.3e}' if abs(x.gain_median)<1e-5 else f'{x.gain_median:.5f}'
        sp_text=f'{x.relative_sp_median:.3e}' if abs(x.relative_sp_median)<1e-5 else f'{x.relative_sp_median:.5f}'
        lines.append(f'| {int(x.r)} | {int(x.client_count)} | {x.relative_pre_median:.5f} | {sp_text} | {gain_text} | {reduction} |')
    lines += ['', '低 rank、较高 rank 以及满 rank 的改善应区分绝对归一化 gain 与相对降低比例；'
              '全 rank 时两种方法都保留完整更新，实际浮点差异完整保留。'
              '“前缀本来最优”以及重复奇异值控制例同样保留零收益。\n',
              '## 稳定性与解释\n']
    for setting in config['settings']:
        x=by_seed[by_seed.setting.eq(setting['id'])]
        lines.append(f"- `{setting['id']}`: {int((x.gain>0).sum())}/{len(x)} 个种子的客户端加权 gain > 0；"
                     f"gain 范围 [{x.gain.min():.5f}, {x.gain.max():.5f}]，"
                     f"总误差降低范围 [{100*x.weighted_total_reduction.min():.2f}%, {100*x.weighted_total_reduction.max():.2f}%]。")
    lines += ['', '普通未排序高斯参考因子在 η=0 时也不保证其索引前缀为最佳低秩近似，'
              '因此 η=0 不应预设为零收益。独立客户端因子属于另一种生成机制，'
              '其收益大小不能直接外推到共享参考或训练检查点。参与客户端数随索引减少，'
              '后部因子平均的扰动方差可与前部不同；这只是当前生成机制的解释线索，'
              '本实验没有对 η 的单调关系做普遍性推断。\n',
              '敏感性 rank 映射按基准比例四舍五入（half-up），R=16 为 '
              '[4,4,6,6,8,8,10,12,14,16]，R=64 为 [16,16,24,24,32,32,40,48,56,64]；'
              '其他尺寸、权重与 30 次重复保持一致。\n',
              '## 数值验证\n',
              f"Float64；{validation['cases']} 个小尺寸控制例、{validation['rank_comparisons']} 个 rank 比较全部通过。"
              f"紧凑误差、实际稠密残差及独立 dense SVD 最优误差的最大绝对差为 "
              f"{validation['max_absolute_gap']:.3e}，按 ‖B‖F‖A‖F 归一化后最大为 "
              f"{validation['max_gap_over_factor_scale']:.3e}。"
              f"逐索引聚合与独立掩码实现最大逐元素差 {validation['aggregation_mask_reference_max_gap']:.3e}。\n",
              '覆盖满 rank、相关列与零列的秩不足、已经最优的前缀、重复奇异值、'
              '完整更新为零但第一项前缀非零、全零因子、不可可靠归一化的近零更新及极小但可归一化更新。'
              '重复奇异值只验证误差最优，不要求截断矩阵逐元素一致。'
              '刻意损坏 SP+ 输出的测试成功被测量与验收逻辑检出。\n',
              '主流程仅在 R×R 核心做一次 SVD。前缀残差以原始因子的 QR 尾部乘积计算；'
              'SP+ 误差以 [B,B⁺]、[Aᵀ,(A⁺)ᵀ] 的联合 QR 基下不超过 2R×2R 的实际残差计算，'
              '包含输出因子中可能出现的原空间外分量。所有目标 rank 共享同一次 SP+ 输出。'
              'E_opt 来自核心奇异值尾部范数，与实际 SP+ 残差分开计算。\n',
              '令 S=‖B‖F‖A‖F、u=float64 eps。仅当 ‖X‖F>64uS 且大于最小正规浮点数时归一化；'
              '否则相对指标为 NA、保留绝对误差。验收绝对容差为 128uS+10⁻¹²‖X‖F。'
              'reduction 仅在 E_pre 大于此容差时报告。原始 CSV 不裁剪；仅绘图时容差内微小负额外误差显示为零。\n',
              f"全部生产数据通过数值验收。max |E_SP+−E_opt|/‖X‖F = "
              f"{manifest['max_abs_normalized_sp_opt_gap']:.3e}；"
              f"r=R 的最大 SP+ 相对残差为 {raw.loc[raw.r.eq(raw.R),'relative_sp'].max():.3e}。"
              f"原始 SP+ excess 的最小值为 {raw.excess_sp.min():.3e}；其原始正负浮点值均保留。\n",
              '## 代码来源与结论边界\n',
              '见 [source_audit.md](../source_audit.md)。当前仓库没有 AGENTS.md、训练侧聚合、'
              'rank assignment 或 LoRA forward 实现。RBLA 按用户指定公式新实现并独立验证，'
              '与归档 YAML 的 mask-normalized padding 描述一致；无法声称已与缺失的训练聚合逐元素对照。'
              'compact SP+ 函数从现有图代码逐字冻结复用，未复用其旋转数据生成器或加 epsilon 的相对误差函数。\n',
              '本实验支持：在这些固定生成设置下，对 RBLA 原始因子聚合产生的同一个完整更新，'
              'SP+ 在相同 rank 预算下减少前缀额外重建误差，并在数值精度内达到最优低秩重建误差。'
              '它不能证明解决了聚合偏置、rank–data mismatch、任务精度、收敛或公平性问题。'
              '这些仍需真实训练检查点的同更新配对测量及完整训练评估。\n',
              '## 图与复现\n',
              '主图：[PDF](figures/main_R32_eta0.3.pdf)、[SVG](figures/main_R32_eta0.3.svg)、'
              '[600 dpi PNG](figures/main_R32_eta0.3.png)。全部设置各有英文双子图；'
              '[完整图注](figures/captions.md)包含生成方式、配置、种子数和统计口径。'
              '图线是逐 rank 跨种子中位数和 IQR，markers 标识实际客户端 rank，纵轴按数据范围自动确定。\n',
              '从项目根目录执行：\n',
              '```powershell\npython "rbla+/figures/synthetic_rbla_redistribution/run.py"\n'
              'python "rbla+/figures/synthetic_rbla_redistribution/plot_figure.py"\n```\n',
              '完整文件与依赖说明见 [README.md](../README.md)。全局 Bᵍ、Aᵍ 与奇异值按设置存于 '
              '`factors/*.npz`；客户端因子由固定种子重建；每行含同一全局因子的 SHA-256。\n',
              '## Paper-ready English summary\n',english+'\n']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    (output/'paper_summary_en.txt').write_text(english+'\n',encoding='utf-8')
    print('Wrote results/report.md and paper_summary_en.txt',flush=True)


if __name__=='__main__':
    write_report(Path(__file__).resolve().parent/'results')
