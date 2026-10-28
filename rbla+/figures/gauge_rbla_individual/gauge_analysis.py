"""Verbatim analysis function frozen from the original gauge figure.
See sources/provenance.json.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
TOLERANCE=1e-10
DEFAULT_SOURCE_CSV=Path(__file__).resolve().parent/"sources"/"gauge_raw.csv"

def prepare_excess_error_data(
    source: pd.DataFrame | str | Path = DEFAULT_SOURCE_CSV,
    *, tolerance: float = TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, bool], dict[str, Any]]:
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    required = {"gauge_type", "trial", "R", "r", "method", "broadcast_error",
                "dense_optimal_error", "full_update_error", "numerical_checks_passed"}
    missing = required.difference(frame.columns)
    if missing: raise ValueError(f"Source data is missing columns: {sorted(missing)}")
    frame["excess_error"] = frame["broadcast_error"] - frame["dense_optimal_error"]
    frame["reference_method"] = "Dense truncated SVD optimum"
    frame["source_results"] = "../gauge_sensitivity/gauge_sensitivity_results.csv"
    ranks = sorted(frame["r"].unique().astype(int)); maximum_rank = int(frame["R"].max())
    rbla = frame[frame["method"].eq("RBLA prefix")]
    sp_plus = frame[frame["method"].eq("SP+ (compact QR)")]
    gauge_stats = (
        rbla.groupby(["gauge_type", "r"])["excess_error"]
        .agg(median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75),
             mean="mean", std="std", minimum="min", maximum="max").reset_index()
    )
    sp_stats = (
        sp_plus.groupby("r")["excess_error"]
        .agg(median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75),
             maximum_absolute=lambda x: np.abs(x).max()).reset_index()
    )
    source_checks = bool((frame["numerical_checks_passed"].astype(str).str.lower() == "true").all())
    diagonal_excess = rbla.loc[rbla["gauge_type"].eq("diagonal"), "excess_error"]
    full_rank_excess = frame.loc[frame["r"].eq(maximum_rank), "excess_error"]
    checks = {
        "source gauge experiment passed its numerical checks": source_checks,
        "all analysis values are finite": bool(np.isfinite(frame["excess_error"]).all()),
        "excess errors are non-negative within tolerance": bool(frame["excess_error"].min() >= -tolerance),
        "SP+ excess error is numerical zero": bool(sp_plus["excess_error"].abs().max() <= tolerance),
        "diagonal RBLA is a zero-excess negative control": bool(diagonal_excess.abs().max() <= tolerance),
        "full-rank excess error is numerical zero": bool(full_rank_excess.abs().max() <= tolerance),
    }
    frame["analysis_checks_passed"] = all(checks.values())
    metrics = {"all_checks_passed": all(checks.values()), "ranks": ranks,
               "maximum_rank": maximum_rank, "maximum_excess": float(rbla["excess_error"].max()),
               "maximum_sp_excess": float(sp_plus["excess_error"].abs().max()),
               "maximum_diagonal_excess": float(diagonal_excess.abs().max())}
    return frame, gauge_stats, sp_stats, checks, metrics
