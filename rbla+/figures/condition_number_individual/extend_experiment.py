"""Extend the paired FP64 gauge experiment, preserving the archived trials."""
from pathlib import Path
import hashlib
import json
import runpy

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
ADDED_KAPPAS = [20, 50, 100, 200, 500, 1000, 2000, 5000,
                10000, 20000, 50000, 100000]


def main():
    plot = runpy.run_path(str(ROOT / 'plot_figure.py'))
    provenance = json.loads((ROOT / 'sources/provenance.json').read_text())
    plot['verify_sources'](provenance)
    config = json.loads((ROOT / 'sources/config.json').read_text())
    generator = runpy.run_path(str(ROOT / 'sources/source_generator.py'))
    params = {k: config[k] for k in ['d_out', 'd_in', 'full_rank', 'client_rank',
              'trials', 'seed', 'target_seed', 'alpha', 'tolerance']}
    # At kappa=1e5, FP64 condition-number estimates have O(kappa*eps)
    # relative sensitivity. This threshold only checks the prescribed kappa;
    # reconstruction and optimality retain the original 1e-10 tolerance.
    condition_rtol = 1e-9
    with threadpool_limits(limits=1):
        frame, checks, metrics = generator['generate_condition_number_data'](
            **params, kappas=[10] + ADDED_KAPPAS, condition_rtol=condition_rtol)
    original = pd.read_csv(ROOT / 'sources/condition_raw.csv', float_precision='round_trip')
    keys = ['trial', 'method']
    old = original[original.target_condition_number.eq(10)].sort_values(keys)
    anchor = frame[frame.target_condition_number.eq(10)].sort_values(keys)
    anchor_gap = float(np.max(np.abs(old.broadcast_error.to_numpy() - anchor.broadcast_error.to_numpy())))
    assert anchor_gap < 1e-12, anchor_gap
    assert all(checks.values()), checks
    added = frame[frame.target_condition_number.isin(ADDED_KAPPAS)].copy()
    output = ROOT / 'extended_condition_raw.csv'
    added.to_csv(output, index=False, float_format='%.17g')
    config.update(kappas=config['kappas'] + ADDED_KAPPAS,
                  added_kappas=ADDED_KAPPAS, condition_rtol=condition_rtol,
                  additional_data=output.name, additional_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  new_experiments_run=True)
    (ROOT / 'extended_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    audit = dict(checks=checks, metrics=metrics, fp_dtype='float64',
                 condition_rtol=condition_rtol, reconstruction_tolerance=config['tolerance'],
                 anchor_kappa=10, anchor_maximum_broadcast_error_gap=anchor_gap,
                 added_kappas=ADDED_KAPPAS, added_rows=len(added),
                 original_source_preserved=True,
                 maximum_condition_relative_error=float(frame.condition_relative_error.max()),
                 maximum_full_update_error=float(frame.full_update_error.max()),
                 maximum_compact_reconstruction_error=float(frame.compact_full_reconstruction_error.max()),
                 maximum_sp_optimality_gap=float(frame.sp_plus_dense_error_gap.max()))
    (ROOT / 'qa/extension_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
