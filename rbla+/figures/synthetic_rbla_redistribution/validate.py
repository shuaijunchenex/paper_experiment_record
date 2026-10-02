"""Independent dense validation; run before the production experiment."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from core import aggregate_rbla, analyze, client_factors, measure


def validate(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(918273)
    m, n, R = 19, 17, 6
    cases = {'random_full_rank': (rng.normal(size=(m, R)), rng.normal(size=(R, n)))}
    b, a = rng.normal(size=(m, R)), rng.normal(size=(R, n))
    b[:, 3:] = b[:, :3] @ rng.normal(size=(3, 3))
    cases['rank_deficient_dependent_columns'] = b, a
    b, a = rng.normal(size=(m, R)), rng.normal(size=(R, n))
    b[:, 3:], a[3:, :] = 0, 0
    cases['rank_deficient_zero_slots'] = b, a
    u, _ = np.linalg.qr(rng.normal(size=(m, R)))
    v, _ = np.linalg.qr(rng.normal(size=(n, R)))
    for name, s in [('already_optimal_prefix', [8, 5, 3, 2, 1, .5]),
                    ('repeated_singular_values', [4, 4, 2, 2, 1, 1])]:
        cases[name] = u * np.sqrt(s), np.sqrt(s)[:, None] * v.T
    b, a = np.zeros((m, R)), np.zeros((R, n))
    b[:, 0] = b[:, 1] = 1
    a[0, :], a[1, :] = 1, -1
    cases['zero_update_nonzero_prefix'] = b.copy(), a.copy()
    a[1, :] += 1e-15
    cases['unreliable_near_zero_update'] = b.copy(), a.copy()
    cases['all_zero_factors'] = np.zeros((m, R)), np.zeros((R, n))
    cases['tiny_but_reliably_normalizable'] = cases['random_full_rank'][0]*1e-60, cases['random_full_rank'][1]*1e-60

    setting = dict(m=m, n=n, R=R, ranks=[2, 3, 6], eta=.3, generator='shared_reference')
    factors, _, _ = client_factors(setting, 1234)
    p = np.array([.2, .3, .5])
    bg, ag, _ = aggregate_rbla(factors, p)
    # Independent mask/weighted-padding implementation, not the per-slot loop.
    bs, aa, masks = np.zeros((3,m,R)), np.zeros((3,R,n)), np.zeros((3,R))
    for i, (b, a) in enumerate(factors):
        r = b.shape[1]
        bs[i, :, :r], aa[i, :r, :], masks[i, :r] = b, a, 1
    denom = p @ masks
    expected_b = np.einsum('i,imr->mr', p, bs) / denom
    expected_a = np.einsum('i,irn->rn', p, aa) / denom[:, None]
    aggregation_gap = max(np.max(np.abs(bg-expected_b)), np.max(np.abs(ag-expected_a)))
    dense_client_average = sum(w*b@a for w,(b,a) in zip(p,factors))
    target_vs_dense_average = float(np.linalg.norm(bg@ag-dense_client_average))
    cases['heterogeneous_weighted_aggregation'] = bg, ag
    setting['eta'] = 0
    factors0, b0, a0 = client_factors(setting, 1234)
    bg0, ag0, _ = aggregate_rbla(factors0, p)
    eta_zero_gap = max(np.max(np.abs(bg0-b0)), np.max(np.abs(ag0-a0)))

    records, all_checks = [], []
    for name, (b,a) in cases.items():
        rows, bp, ap, s, _ = analyze(b, a)
        x = b @ a  # Independent dense path, small matrices only.
        dense_s = np.linalg.svd(x, compute_uv=False)
        for row in rows:
            r, tol = row['r'], row['absolute_tolerance']
            pre_dense = float(np.linalg.norm(x-b[:, :r]@a[:r, :], 'fro'))
            sp_dense = float(np.linalg.norm(x-bp[:, :r]@ap[:r, :], 'fro'))
            opt_dense = float(np.linalg.norm(dense_s[r:]))
            gaps = [abs(row['error_pre']-pre_dense), abs(row['error_sp']-sp_dense),
                    abs(row['error_opt']-opt_dense), abs(sp_dense-opt_dense)]
            passed = all(g <= tol for g in gaps) and row['checks_passed']
            if name in ('already_optimal_prefix', 'repeated_singular_values'):
                passed &= abs(pre_dense-opt_dense) <= tol
            if name in ('zero_update_nonzero_prefix', 'unreliable_near_zero_update', 'all_zero_factors'):
                passed &= not row['normalization_available'] and np.isnan(row['gain'])
            if name == 'zero_update_nonzero_prefix' and r == 1:
                passed &= pre_dense > 1 and row['error_pre'] > 1 and np.linalg.norm(x) == 0
            if name == 'tiny_but_reliably_normalizable':
                passed &= row['normalization_available']
            records.append(dict(case=name, **row, dense_error_pre=pre_dense,
                                dense_error_sp=sp_dense, dense_error_opt=opt_dense,
                                max_absolute_gap=max(gaps),
                                max_gap_over_factor_scale=max(gaps)/row['factor_scale'] if row['factor_scale'] else 0,
                                validation_passed=bool(passed)))
            all_checks.append(bool(passed))

    # Deliberately corrupt actual output: a copied optimum would miss this.
    b,a = cases['random_full_rank']
    _, bp, ap, s, _ = analyze(b,a)
    bad_bp = bp.copy()
    bad_bp[:,0] += .1*rng.normal(size=m)
    bad_rows = measure(b,a,bad_bp,ap,s)
    corruption_detected = not bad_rows[-1]['checks_passed'] and bad_rows[-1]['error_sp'] > 1e-3
    direct_bad = np.linalg.norm(b@a-bad_bp@ap)
    corruption_error_measured = abs(direct_bad-bad_rows[-1]['error_sp']) <= bad_rows[-1]['absolute_tolerance']
    summary = dict(all_passed=all(all_checks) and aggregation_gap < 1e-14 and eta_zero_gap < 1e-14
                   and target_vs_dense_average > 1e-3 and corruption_detected and corruption_error_measured,
                   cases=len(cases), rank_comparisons=len(records),
                   max_absolute_gap=max(x['max_absolute_gap'] for x in records),
                   max_gap_over_factor_scale=max(x['max_gap_over_factor_scale'] for x in records),
                   aggregation_mask_reference_max_gap=float(aggregation_gap),
                   eta_zero_reference_max_gap=float(eta_zero_gap),
                   aggregate_target_vs_dense_client_average_gap=target_vs_dense_average,
                   deliberately_corrupted_output_detected=corruption_detected,
                   corrupted_output_dense_compact_agree=bool(corruption_error_measured))
    summary['all_passed'] = bool(summary['all_passed'])
    pd.DataFrame(records).to_csv(output/'dense_validation.csv', index=False, float_format='%.17g')
    (output/'validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    if not summary['all_passed']:
        raise AssertionError('Independent validation failed; see CSV')
    return summary


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        validate(Path(__file__).resolve().parent/'results'/'validation')
