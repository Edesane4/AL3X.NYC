"""Core unit tests for Bayesian Model Averaging.

Covers:
  * Per-source bias is learned correctly from synthetic history
  * Pairwise covariance on correlated sources discounts combined var
  * <_MIN_PER_SOURCE_N rows per active source → compute_bma returns
    None (caller falls back to plain weighted mean)
"""

from __future__ import annotations

from datetime import date, timedelta

from al3x import bma as bma_mod
from al3x.bma import compute_bma


class _FakeStorage:
    def __init__(self, rows_per_source):
        self._rows = rows_per_source

    def get_source_history(self, name, days=30):
        return list(self._rows.get(name, []))


def _rows(n, bias_f, jitter_f=0.5):
    """Build n rows where truth = predicted + bias + alternating jitter."""
    base = date(2026, 4, 20)
    out = []
    for i in range(n):
        d = (base - timedelta(days=i)).isoformat()
        predicted = 70.0 + i * 0.2
        truth = predicted + bias_f + (jitter_f if i % 2 else -jitter_f)
        out.append({"target_date": d, "predicted_f": predicted, "cli_f": truth})
    return out


def test_bma_learns_per_source_bias_correctly():
    storage = _FakeStorage({
        "hrrr":    _rows(8, bias_f=1.0, jitter_f=0.2),
        "gfs_mos": _rows(8, bias_f=-2.0, jitter_f=0.2),
        "ecmwf":   _rows(8, bias_f=0.0, jitter_f=0.2),
    })
    source_values = {"hrrr": 72.0, "gfs_mos": 71.5, "ecmwf": 71.0}
    weights = {"hrrr": 0.4, "gfs_mos": 0.3, "ecmwf": 0.3}
    result = compute_bma(source_values, weights, storage,
                         date(2026, 4, 22), mode="intraday")
    assert result is not None
    biases = result["source_biases"]
    assert abs(biases["hrrr"] - 1.0) < 0.25
    assert abs(biases["gfs_mos"] - (-2.0)) < 0.25
    assert abs(biases["ecmwf"]) < 0.25


def test_bma_pairwise_covariance_discounts_correlated_sources():
    """Two perfectly correlated sources should NOT look like two
    independent votes — their positive covariance must increase the
    combined predictive variance above what independence would give."""
    n = 8
    base = date(2026, 4, 20)
    hrrr_rows = []
    ecmwf_rows = []
    for i in range(n):
        d = (base - timedelta(days=i)).isoformat()
        pred = 70.0 + i * 0.2
        shared_err = (0.5 if i % 2 else -0.5) + 2.0   # bias + shared jitter
        hrrr_rows.append({"target_date": d, "predicted_f": pred,
                           "cli_f": pred + shared_err})
        ecmwf_rows.append({"target_date": d, "predicted_f": pred,
                            "cli_f": pred + shared_err})
    storage = _FakeStorage({"hrrr": hrrr_rows, "ecmwf": ecmwf_rows})
    source_values = {"hrrr": 72.0, "ecmwf": 72.0}
    weights = {"hrrr": 0.5, "ecmwf": 0.5}
    result = compute_bma(source_values, weights, storage,
                         date(2026, 4, 22), mode="intraday")
    assert result is not None
    cov = result["covariance_matrix"]
    assert cov, "covariance matrix should be populated for correlated pair"
    only_cov = next(iter(cov.values()))
    assert only_cov > 0.05, (
        f"expected positive shared-error covariance, got {only_cov}"
    )


def test_bma_returns_none_with_insufficient_total_history():
    """Combined per-source row count below ``_MIN_HISTORY_DAYS`` →
    compute_bma must return None so the caller falls back to the plain
    weighted mean. Each per-source count is also below
    ``_MIN_PER_SOURCE_N``, so this covers the 'thin history' branch."""
    # Choose row counts so the sum is strictly below _MIN_HISTORY_DAYS
    # *and* each per-source count is below _MIN_PER_SOURCE_N.
    per_source = min(bma_mod._MIN_PER_SOURCE_N - 1,
                     (bma_mod._MIN_HISTORY_DAYS - 1) // 2)
    assert per_source >= 0
    storage = _FakeStorage({
        "hrrr":    _rows(per_source, bias_f=1.0),
        "gfs_mos": _rows(per_source, bias_f=-1.0),
    })
    source_values = {"hrrr": 72.0, "gfs_mos": 71.5}
    weights = {"hrrr": 0.5, "gfs_mos": 0.5}
    result = compute_bma(source_values, weights, storage,
                         date(2026, 4, 22), mode="intraday")
    assert result is None, (
        "BMA must return None when combined history is below "
        "_MIN_HISTORY_DAYS; caller falls back to plain weighted mean."
    )
