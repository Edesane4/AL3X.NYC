"""FIX 5 — BMA history window truncated from 30 to 10 days.

Evidence: BMA learned +6.99°F Kalman bias while Kalman's actual 8-day
MAE was 0.55°F. The 30-day window kept integrating stale data from
earlier code paths. Short window + lower _MIN_HISTORY_DAYS threshold
lets BMA recover faster when a source's accuracy improves.
"""

from __future__ import annotations

from datetime import date, timedelta


class _FakeStorage:
    """Minimal storage stand-in: returns pre-seeded history per source."""

    def __init__(self, rows_per_source):
        self._rows_per_source = rows_per_source

    def get_source_history(self, source_name: str, days: int = 30):
        return list(self._rows_per_source.get(source_name, []))


def _seed_rows(n: int, bias_f: float):
    """Build n (date, predicted, truth) rows with constant bias so that
    variance is non-zero and bias estimate is honest."""
    base = date(2026, 4, 15)
    rows = []
    for i in range(n):
        d = (base - timedelta(days=i)).isoformat()
        predicted = 70.0 + i * 0.1  # some variation so variance > 0
        truth = predicted + bias_f + (0.5 if i % 2 else -0.5)
        rows.append({"target_date": d, "predicted_f": predicted,
                     "cli_f": truth})
    return rows


def test_bma_produces_result_with_exactly_five_rows_per_source() -> None:
    """With 5 rows each for two sources, compute_bma must return a
    non-None result (the old 10-day minimum would have rejected this).
    """
    from al3x.bma import compute_bma

    storage = _FakeStorage({
        "hrrr": _seed_rows(5, bias_f=1.0),
        "gfs_mos": _seed_rows(5, bias_f=-0.5),
    })
    source_values = {"hrrr": 72.0, "gfs_mos": 70.5}
    weights = {"hrrr": 0.6, "gfs_mos": 0.4}

    result = compute_bma(source_values, weights, storage,
                         date(2026, 4, 22), mode="intraday")
    assert result is not None, (
        "compute_bma returned None with 5 rows/source — the "
        "_MIN_HISTORY_DAYS threshold is still too strict."
    )
    assert "bma_forecast_f" in result
    assert "bma_variance_f" in result
    # With bias correction, BMA output should be shifted toward
    # ~truth (predicted + bias). Sanity bound: within a few degrees.
    assert 70.0 < result["bma_forecast_f"] < 75.0
