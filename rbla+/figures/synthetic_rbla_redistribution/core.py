"""Float64 synthetic RBLA aggregation and measured redistribution errors.

No dense m x n update is formed here. See validate.py for independent dense tests.
"""
from __future__ import annotations
import numpy as np
from reference_sp_plus import compact_sp_plus

EPS = np.finfo(np.float64).eps
NORMALIZATION_MULTIPLIER = 64.0
ABS_TOL_MULTIPLIER = 128.0
REL_TOL = 1e-12


def client_factors(setting: dict, seed: int):
    """Independent RNG streams; reuse reference/noise streams across eta values.

    B0 entries ~ N(0,1/m), A0 entries ~ N(0,1/n). For a reference block F,
    N = Z * ||F||_F / sqrt(F.size), Z iid N(0,1). Thus the expected squared
    perturbation norm is eta**2 * ||F||_F**2, independently of block dimensions.
    Noise is not rescaled to an exact realized norm.
    """
    m, n, R = setting['m'], setting['n'], setting['R']
    streams = np.random.SeedSequence([seed, R, m, n]).spawn(1 + 2 * len(setting['ranks']))
    ref_rng = np.random.default_rng(streams[0])
    b0 = ref_rng.standard_normal((m, R)) / np.sqrt(m)
    a0 = ref_rng.standard_normal((R, n)) / np.sqrt(n)
    factors = []
    for i, r in enumerate(setting['ranks']):
        z_b = np.random.default_rng(streams[1 + 2*i]).standard_normal((m, r))
        z_a = np.random.default_rng(streams[2 + 2*i]).standard_normal((r, n))
        if setting['generator'] == 'shared_reference':
            ref_b, ref_a = b0[:, :r], a0[:r, :]
            b = ref_b + setting['eta'] * z_b * (np.linalg.norm(ref_b) / np.sqrt(ref_b.size))
            a = ref_a + setting['eta'] * z_a * (np.linalg.norm(ref_a) / np.sqrt(ref_a.size))
        elif setting['generator'] == 'independent':
            b, a = z_b / np.sqrt(m), z_a / np.sqrt(n)
        else:
            raise ValueError('Unknown generator')
        factors.append((b, a))
    return factors, b0, a0


def aggregate_rbla(factors, weights):
    """Original per-index participation-normalized factor average, scaling=1."""
    p = np.asarray(weights, dtype=np.float64)
    if len(p) != len(factors) or np.any(p <= 0) or not np.isclose(p.sum(), 1):
        raise ValueError('Require positive client weights summing to one')
    m, n = factors[0][0].shape[0], factors[0][1].shape[1]
    ranks = np.array([b.shape[1] for b, _ in factors])
    R = int(ranks.max())
    b_global, a_global = np.zeros((m, R)), np.zeros((R, n))
    denominators = np.zeros(R)
    for b, a in factors:
        if b.dtype != np.float64 or a.dtype != np.float64:
            raise ValueError('Float64 factors required')
        if a.shape != (b.shape[1], n) or b.shape[0] != m:
            raise ValueError('Incompatible factors')
        if not np.isfinite(b).all() or not np.isfinite(a).all():
            raise ValueError('Non-finite factors')
    for s in range(R):
        active = np.flatnonzero(ranks > s)
        denominators[s] = p[active].sum()
        for i in active:
            alpha = p[i] / denominators[s]
            b_global[:, s] += alpha * factors[i][0][:, s]
            a_global[s, :] += alpha * factors[i][1][s, :]
    return b_global, a_global, denominators


def measure(b, a, b_plus, a_plus, singular_values):
    """Measure actual output factors, including components outside original spans.

    Prefix residual is the tail product Rb[:,r:] Ra[:,r:].T, avoiding subtraction.
    SP+ residual uses joint QR bases of [B,B+] and [A.T,A+.T]. Both target and
    actual output are represented in those bases; their difference is measured
    in a <=2R x <=2R core. This never substitutes the singular-value tail for the
    measured SP+ error, and handles a cancelling zero target/nonzero prefix.
    """
    m, R = b.shape
    n = a.shape[1]
    if not (R < min(m, n)):
        raise ValueError('This experiment requires adapter width R < min(m,n)')
    _, rb = np.linalg.qr(b, mode='reduced')
    _, ra = np.linalg.qr(a.T, mode='reduced')
    target_core = rb @ ra.T
    norm_x = float(np.linalg.norm(target_core, 'fro'))
    factor_scale = float(np.linalg.norm(b, 'fro') * np.linalg.norm(a, 'fro'))
    floor = NORMALIZATION_MULTIPLIER * EPS * factor_scale
    reliable = bool(np.isfinite(norm_x) and norm_x > max(floor, np.finfo(float).tiny))
    abs_tol = ABS_TOL_MULTIPLIER * EPS * factor_scale + REL_TOL * norm_x
    _, jb = np.linalg.qr(np.concatenate((b, b_plus), axis=1), mode='reduced')
    _, ja = np.linalg.qr(np.concatenate((a.T, a_plus.T), axis=1), mode='reduced')
    joint_target = jb[:, :R] @ ja[:, :R].T
    rows = []
    for r in range(1, R + 1):
        pre = float(np.linalg.norm(rb[:, r:] @ ra[:, r:].T, 'fro'))
        sp = float(np.linalg.norm(joint_target - jb[:, R:R+r] @ ja[:, R:R+r].T, 'fro'))
        opt = float(np.linalg.norm(singular_values[r:]))
        row = dict(r=r, normalized_rank=r/R, norm_x=norm_x, factor_scale=factor_scale,
                   normalization_floor=floor, normalization_available=reliable,
                   absolute_tolerance=abs_tol, error_pre=pre, error_sp=sp, error_opt=opt,
                   absolute_excess_pre=pre-opt, absolute_excess_sp=sp-opt,
                   absolute_gain=pre-sp)
        for key, value in dict(relative_pre=pre, relative_sp=sp, relative_opt=opt,
                               excess_pre=pre-opt, excess_sp=sp-opt, gain=pre-sp).items():
            row[key] = value / norm_x if reliable else np.nan
        row['reduction'] = (pre-sp) / pre if pre > max(abs_tol, np.finfo(float).tiny) else np.nan
        row['checks_passed'] = bool(pre >= opt-abs_tol and abs(sp-opt) <= abs_tol
                                    and sp <= pre+abs_tol and (r < R or max(pre, sp) <= abs_tol))
        rows.append(row)
    return rows


def analyze(b, a):
    # Sole public SP+ decomposition per aggregation; all ranks share it.
    bp, ap, s, shapes = compact_sp_plus(b, a)
    rows = measure(b, a, bp, ap, s)
    return rows, bp, ap, s, shapes
