"""Run validation, synthetic experiments, summary tables, paper figures, report."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits
from core import aggregate_rbla, analyze, client_factors
from validate import validate

ROOT = Path(__file__).resolve().parent
METRICS = ['error_pre', 'error_sp', 'error_opt', 'relative_pre', 'relative_sp',
           'relative_opt', 'excess_pre', 'excess_sp', 'gain', 'reduction']


def save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g')


def summarize(frame, group_keys, metrics):
    rows = []
    for keys, group in frame.groupby(group_keys, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        row['seed_count'] = group['seed'].nunique()
        for metric in metrics:
            x = group[metric].dropna()
            for stat, value in dict(count=x.size, median=x.median(), q25=x.quantile(.25),
                                    q75=x.quantile(.75), mean=x.mean(), std=x.std(),
                                    min=x.min(), max=x.max()).items():
                row[f'{metric}_{stat}'] = value
        rows.append(row)
    return pd.DataFrame(rows)


def check_config(config):
    if config['effective_lora_scaling'] != 1 or config['post_aggregation_gauge']:
        raise ValueError('Require effective scaling=1 and no post-aggregation gauge')
    if config['dtype'] != 'float64' or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Require float64 and distinct seeds')
    ids = [s['id'] for s in config['settings']]
    if len(set(ids)) != len(ids) or config['main_setting'] not in ids:
        raise ValueError('Require distinct setting IDs and a preselected main setting')
    for s in config['settings']:
        if not (0 < max(s['ranks']) == s['R'] < min(s['m'],s['n'])):
            raise ValueError('Require at least one client at R, with R < min(m,n)')
        if min(s['ranks']) < 1 or any(int(r) != r for r in s['ranks']):
            raise ValueError('Require positive integer ranks')
        if s['generator'] == 'shared_reference' and s['eta'] < 0:
            raise ValueError('Require nonnegative eta')


def run(config, output):
    check_config(config)
    output.mkdir(parents=True, exist_ok=True)
    config_text = json.dumps(config, indent=2)+'\n'
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()
    (output/'config_frozen.json').write_text(config_text,encoding='utf-8')
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), config_sha256=config_hash,
                    python=sys.version, platform=platform.platform(), command=sys.argv,
                    dependencies={k:importlib.metadata.version(k) for k in
                                  ['numpy','pandas','matplotlib','scipy','threadpoolctl','PyMuPDF','Pillow']},
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
                    blas=threadpool_info(), completed=False)
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    start = time.perf_counter()
    validate(output/'validation')
    raw, client_rows, scan_rows, diagnostic_rows, participant_rows = [], [], [], [], []
    # Preregistered main setting first, then supplements, then sensitivity checks.
    settings = sorted(config['settings'], key=lambda s: s['id'] != config['main_setting'])
    for setting in settings:
        setting_rows, factor_archive = [], {}
        R, ranks, weights = setting['R'], setting['ranks'], np.array(setting['weights'])
        for r in range(1, R+1):
            active = np.array(ranks) >= r
            participant_rows.append(dict(setting=setting['id'],slot=r,participating_clients=int(active.sum()),
                                         weight_denominator=float(weights[active].sum())))
        for seed in config['seeds']:
            factors, _, _ = client_factors(setting, seed)
            b,a,_ = aggregate_rbla(factors, weights)  # Exactly one aggregation per seed.
            rows,bp,ap,s,shapes = analyze(b,a)         # Exactly one SP+ core SVD.
            identity = hashlib.sha256(b.tobytes()+a.tobytes()).hexdigest()
            for row in rows:
                r = row['r']
                row.update(setting=setting['id'],generator=setting['generator'],eta=setting['eta'],
                           R=R,m=setting['m'],n=setting['n'],N=len(ranks),seed=seed,
                           is_client_rank=r in ranks,client_count=ranks.count(r),
                           client_weight_at_rank=float(weights[np.array(ranks)==r].sum()),
                           effective_lora_scaling=1.,global_factor_sha256=identity,sp_decompositions=1)
            setting_rows.extend(rows)
            # First weight within each seed, THEN summarize across independent seeds.
            selected = [rows[r-1] for r in ranks]
            client = dict(setting=setting['id'],seed=seed,R=R)
            for metric in METRICS:
                if metric == 'reduction':
                    continue
                values = np.array([x[metric] for x in selected])
                client[metric] = float(weights@values) if np.isfinite(values).all() else np.nan
            client['weighted_total_reduction'] = (client['error_pre']-client['error_sp'])/client['error_pre'] if client['error_pre'] > max(x['absolute_tolerance'] for x in rows) else np.nan
            valid = np.isfinite([x['reduction'] for x in selected])
            client['reduction_available_weight'] = float(weights[valid].sum())
            client['mean_client_reduction_conditional'] = float(weights[valid]@np.array([x['reduction'] for x in selected])[valid]/weights[valid].sum()) if valid.any() else np.nan
            client_rows.append(client)
            scan = dict(setting=setting['id'],seed=seed,R=R)
            for metric in METRICS:
                values = np.array([x[metric] for x in rows])
                scan[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            scan_rows.append(scan)
            diagnostic_rows.append(dict(setting=setting['id'],seed=seed,global_factor_sha256=identity,
                                        numerical_rank=int(np.sum(s > s[0]*max(b.shape[0],a.shape[1])*np.finfo(float).eps)) if s[0] else 0,
                                        adapter_width=R,core_shape=str(shapes['core']),
                                        max_abs_normalized_sp_opt_gap=max(abs(row['excess_sp']) for row in rows),
                                        full_rank_relative_sp=rows[-1]['relative_sp'],
                                        min_excess_pre=min(row['excess_pre'] for row in rows),
                                        min_excess_sp=min(row['excess_sp'] for row in rows),
                                        max_gap_to_tolerance=max(abs(row['absolute_excess_sp'])/row['absolute_tolerance'] for row in rows),
                                        all_checks_passed=all(row['checks_passed'] for row in rows)))
            if config['save_global_factors']:
                factor_archive[f'{seed}_B_global'] = b
                factor_archive[f'{seed}_A_global'] = a
                factor_archive[f'{seed}_singular_values'] = s
        frame = pd.DataFrame(setting_rows)
        save_csv(frame,output/'raw'/f"{setting['id']}.csv")
        if factor_archive:
            (output/'factors').mkdir(exist_ok=True)
            np.savez_compressed(output/'factors'/f"{setting['id']}.npz",**factor_archive)
        if not frame['checks_passed'].all():
            raise AssertionError(f"Numerical checks failed in {setting['id']}; raw results retained")
        raw.extend(setting_rows)
        print(f"PASS {setting['id']}: {len(config['seeds'])} seeds, {len(frame)} rank comparisons",flush=True)

    raw, clients, scans = pd.DataFrame(raw),pd.DataFrame(client_rows),pd.DataFrame(scan_rows)
    rank_summary = summarize(raw,['setting','R','r','normalized_rank','is_client_rank','client_count','client_weight_at_rank'],METRICS)
    client_metrics = [m for m in METRICS if m != 'reduction']+['weighted_total_reduction','mean_client_reduction_conditional','reduction_available_weight']
    client_summary = summarize(clients,['setting','R'],client_metrics)
    for frame, name in [(raw,'raw_all'),(clients,'client_weighted_by_seed'),(scans,'all_rank_scan_by_seed'),
                        (rank_summary,'rank_summary'),(client_summary,'client_weighted_summary'),
                        (summarize(scans,['setting','R'],METRICS),'all_rank_scan_summary'),
                        (pd.DataFrame(diagnostic_rows),'numerical_diagnostics'),
                        (pd.DataFrame(participant_rows),'slot_participation')]:
        save_csv(frame,output/f'{name}.csv')
    manifest.update(completed=True,completed_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter()-start,settings=len(settings),
                    aggregations=len(diagnostic_rows),rank_comparisons=len(raw),
                    validation_passed=True,all_production_checks_passed=bool(raw['checks_passed'].all()),
                    max_abs_normalized_sp_opt_gap=float(raw['excess_sp'].abs().max()),
                    raw_sha256=hashlib.sha256((output/'raw_all.csv').read_bytes()).hexdigest())
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f"Completed: {len(diagnostic_rows)} aggregations, {len(raw)} ranks, {manifest['elapsed_seconds']:.1f}s",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--no-figures',action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    with threadpool_limits(limits=config['blas_threads']):
        if args.validate_only:
            validate(args.output/'validation')
            return
        run(config,args.output)
    if not args.no_figures:
        from plot_figure import plot_all
        plot_all(args.output)
    from report import write_report
    write_report(args.output)


if __name__ == '__main__':
    main()
