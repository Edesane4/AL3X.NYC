"""Quant 1 — Bayesian Model Averaging with Covariance.

Replaces the plain weighted mean and fixed ±2°F uncertainty with a
bias-corrected BMA forecast plus a calibrated predictive variance.

Estimates per-source bias and variance from the last 30 days of
verified forecasts, plus the pairwise error covariance matrix. The
combined predictive variance accounts for correlated model errors (so
five tightly-coupled model sources no longer look like five independent
votes).
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("al3x.bma")


# FIX 5 — Short window so BMA adapts quickly when a source's accuracy
# improves. Longer windows carry stale bias from earlier code paths:
# BMA had learned a +6.99°F Kalman bias when Kalman's actual 8-day MAE
# was 0.55°F, because the 30-day window kept integrating legacy
# forecasts no longer reflective of current reality.
_MIN_HISTORY_DAYS = 5
_MIN_PER_SOURCE_N = 5
_DEFAULT_VARIANCE = 4.0   # (°F)^2 — corresponds to ±2°F default 1σ


def _pair_key(a: str, b: str) -> frozenset:
    return frozenset({a, b})


def compute_bma(source_values: Dict[str, float],
                 weights: Dict[str, float],
                 storage,
                 target_date: date,
                 mode: str) -> Optional[Dict[str, Any]]:
    """Return BMA result dict or None if insufficient history.

    source_values: currently-predicted values for each active source.
    weights:       current live weights (already re-normalized over
                   active sources).
    storage:       Storage instance (passed through from forecaster).
    target_date:   forecast's target calendar date (unused for math but
                   included for future temporal weighting).
    mode:          'night_before' or 'intraday' (informational).
    """
    active = [s for s, v in source_values.items() if v is not None]
    if not active:
        return None

    # Build per-source bias and variance from the last 30 days of
    # (predicted, truth) pairs.
    source_biases: Dict[str, float] = {}
    source_variances: Dict[str, float] = {}
    histories: Dict[str, List[Tuple[str, float, float]]] = {}
    for s in active:
        try:
            rows = storage.get_source_history(s, days=10)
        except Exception as e:  # noqa: BLE001
            log.info("BMA: get_source_history failed for %s: %s", s, e)
            rows = []
        histories[s] = [(r["target_date"], r["predicted_f"], r["cli_f"])
                        for r in rows]
        errs = [cli - pred for _, pred, cli in histories[s]]
        if len(errs) >= _MIN_PER_SOURCE_N:
            mean_err = sum(errs) / len(errs)
            # population variance
            var = sum((e - mean_err) ** 2 for e in errs) / len(errs)
            source_biases[s] = float(mean_err)
            source_variances[s] = float(max(var, 0.25))
        else:
            source_biases[s] = 0.0
            source_variances[s] = _DEFAULT_VARIANCE

    # Overall history count (any source with rows)
    total_rows = sum(len(h) for h in histories.values())
    if total_rows < _MIN_HISTORY_DAYS:
        log.info("BMA: only %d historical source-days, need %d — skipping",
                 total_rows, _MIN_HISTORY_DAYS)
        return None

    # Build pairwise covariance matrix: align by date so we only use days
    # where both sources had a prediction.
    def _err_by_date(hist: List[Tuple[str, float, float]]
                     ) -> Dict[str, float]:
        return {d: (cli - pred) for d, pred, cli in hist}

    err_maps = {s: _err_by_date(histories[s]) for s in active}
    covariance: Dict[frozenset, float] = {}
    for i, si in enumerate(active):
        for sj in active[i + 1:]:
            common = set(err_maps[si].keys()) & set(err_maps[sj].keys())
            if len(common) < _MIN_PER_SOURCE_N:
                # fall back to independence (cov = 0)
                covariance[_pair_key(si, sj)] = 0.0
                continue
            ei = [err_maps[si][d] for d in common]
            ej = [err_maps[sj][d] for d in common]
            mi = sum(ei) / len(ei)
            mj = sum(ej) / len(ej)
            cov = (sum((a - mi) * (b - mj) for a, b in zip(ei, ej))
                   / len(common))
            covariance[_pair_key(si, sj)] = float(cov)

    # BMA forecast: Σ w_i * (f_i + bias_i)
    T_bma = 0.0
    total_weight = 0.0
    for s in active:
        w = float(weights.get(s, 0.0))
        if w <= 0:
            continue
        total_weight += w
        T_bma += w * (source_values[s] + source_biases[s])
    if total_weight <= 0:
        return None
    T_bma /= total_weight  # re-normalize in case caller's weights didn't

    # Combined predictive variance
    # Var = Σ w_i² * σ_i² + Σ_{i!=j} w_i w_j Cov(i,j)
    #     = Σ w_i² * σ_i² + 2 * Σ_{i<j} w_i w_j Cov(i,j)
    var_sum = 0.0
    for s in active:
        w = float(weights.get(s, 0.0)) / total_weight
        var_sum += (w ** 2) * source_variances[s]
    for i, si in enumerate(active):
        wi = float(weights.get(si, 0.0)) / total_weight
        for sj in active[i + 1:]:
            wj = float(weights.get(sj, 0.0)) / total_weight
            cov = covariance.get(_pair_key(si, sj), 0.0)
            var_sum += 2.0 * wi * wj * cov

    var_sum = max(var_sum, 0.01)  # floor to avoid sqrt(0) or negative
    sigma = math.sqrt(var_sum)

    # Represent covariance dict as string keys for JSON friendliness
    cov_dict = {"|".join(sorted(list(k))): round(v, 3)
                for k, v in covariance.items()}

    return {
        "bma_forecast_f": round(float(T_bma), 2),
        "bma_variance_f": round(float(sigma), 2),
        "source_biases": {k: round(v, 2) for k, v in source_biases.items()},
        "source_variances": {k: round(v, 2)
                              for k, v in source_variances.items()},
        "covariance_matrix": cov_dict,
        "history_days": total_rows,
    }
