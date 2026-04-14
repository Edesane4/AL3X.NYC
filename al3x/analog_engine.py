"""Quant 3 — Synoptic pattern analog matching.

Looks up the K most similar past days (by an 8-element feature vector)
with verified CLI truth, computes their ensemble-vs-truth errors, and
returns a similarity-weighted bias estimate. Applied as a signed delta
on the post-correction forecast.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("al3x.analog")


FEATURE_NAMES = [
    "temp_max_proxy", "dewpoint_f", "wind_speed_kt",
    "precip_prob_peak", "sky_cover_avg",
    "month_sin", "month_cos", "lead_hours_norm",
]

_BIAS_CAP = 2.5        # °F hard cap
_DIST_CUTOFF = 0.5     # if nearest analog is >0.5 away, reduce confidence
_MIN_HISTORY = 14      # days required before the engine engages


def build_feature_vector(*,
                          hrrr_forecast_f: Optional[float],
                          dewpoint_f: Optional[float],
                          wind_speed_kt: Optional[float],
                          precip_prob_pct: Optional[float],
                          sky_cover_pct: Optional[float],
                          month: int,
                          lead_hours: float) -> List[float]:
    """Normalize each component to [0, 1]. Missing values -> 0.5 (midpoint)."""
    def _norm(v: Optional[float], lo: float, hi: float) -> float:
        if v is None:
            return 0.5
        return max(0.0, min(1.0, (float(v) - lo) / max(hi - lo, 1e-6)))

    month_rad = 2.0 * math.pi * (month / 12.0)
    return [
        _norm(hrrr_forecast_f, 0.0, 110.0),
        _norm(dewpoint_f, 20.0, 80.0),
        _norm(wind_speed_kt, 0.0, 40.0),
        _norm(precip_prob_pct, 0.0, 100.0),
        _norm(sky_cover_pct, 0.0, 100.0),
        (math.sin(month_rad) + 1.0) / 2.0,
        (math.cos(month_rad) + 1.0) / 2.0,
        max(0.0, min(1.0, lead_hours / 24.0)),
    ]


def _euclid(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _feature_vector_for_historical(fc: Dict[str, Any]) -> Optional[List[float]]:
    """Rebuild an 8-element vector for a historical forecast row.

    Pulls what it needs out of extras.regime / sources / extras.lead_hours.
    Missing fields default to mid-range — fine for nearest-neighbor work.
    """
    extras = fc.get("extras", {}) or {}
    regime = extras.get("regime", {}) or {}
    sources = fc.get("sources", {}) or {}

    hrrr_f = (sources.get("hrrr") or {}).get("value")
    try:
        ts_month = int((fc.get("target_date") or "0000-00-00")
                        .split("-")[1] or 1)
    except Exception:
        ts_month = 1

    lead = float(extras.get("lead_hours") or 12.0)

    # Observations field isn't persisted per forecast, so we approximate
    # surface dewpoint via regime flags (inversion → close to temp)
    # and wind_speed via regime.max_wind_kt.
    wind = regime.get("max_wind_kt")
    sky = regime.get("cloud_avg")

    # Precip prob: approximate from regime flags
    if regime.get("precip_heavy"):
        precip = 80.0
    elif regime.get("any_precip_peak"):
        precip = 50.0
    else:
        precip = 10.0

    # Dewpoint: we don't have it directly; use a seasonal baseline
    #  Summer ~65, Winter ~25, Spring/Fall ~45
    if 6 <= ts_month <= 8:
        dewpoint = 65.0
    elif 12 <= ts_month or ts_month <= 2:
        dewpoint = 25.0
    else:
        dewpoint = 45.0

    return build_feature_vector(
        hrrr_forecast_f=hrrr_f,
        dewpoint_f=dewpoint,
        wind_speed_kt=wind,
        precip_prob_pct=precip,
        sky_cover_pct=sky,
        month=ts_month,
        lead_hours=lead,
    )


def find_analogs(feature_vector: List[float], storage, K: int = 7,
                  min_history_days: int = _MIN_HISTORY
                  ) -> Optional[Dict[str, Any]]:
    """Return {analog_bias_f, analog_confidence, analogs_used} or None.

    `storage` must expose `forecasts_with_truth(days=N)` returning rows
    with at least: target_date, final_f, sources, extras, cli_f.
    """
    try:
        rows = storage.forecasts_with_truth(days=365)
    except Exception as e:  # noqa: BLE001
        log.info("analog: forecasts_with_truth failed: %s", e)
        return None

    if len(rows) < min_history_days:
        log.info("analog: only %d historical days, need %d — skip",
                 len(rows), min_history_days)
        return None

    library: List[Tuple[List[float], float, str]] = []  # (vec, error, date)
    for r in rows:
        final_f = r.get("final_f")
        cli_f = r.get("cli_f")
        if final_f is None or cli_f is None:
            continue
        vec = _feature_vector_for_historical(r)
        if vec is None or len(vec) != len(feature_vector):
            continue
        err = float(cli_f) - float(final_f)
        library.append((vec, err, r.get("target_date", "")))

    if len(library) < min_history_days:
        return None

    # K nearest neighbors
    ranked = sorted(
        ((_euclid(feature_vector, v), err, d) for v, err, d in library),
        key=lambda x: x[0],
    )
    top = ranked[:max(1, K)]
    if not top:
        return None

    distances = [t[0] for t in top]
    errors = [t[1] for t in top]
    dates = [t[2] for t in top]

    # Inverse-distance weights (avoid div by zero)
    inv = [(1.0 / max(d, 1e-4)) for d in distances]
    tot = sum(inv) or 1.0
    w = [i / tot for i in inv]
    analog_bias = sum(wi * e for wi, e in zip(w, errors))

    # Confidence shrinks if the nearest analog is far
    min_d = min(distances)
    confidence = max(0.0, 1.0 - min_d / _DIST_CUTOFF)
    analog_bias *= confidence

    # Cap absolute size
    if analog_bias > _BIAS_CAP:
        analog_bias = _BIAS_CAP
    elif analog_bias < -_BIAS_CAP:
        analog_bias = -_BIAS_CAP

    return {
        "analog_bias_f": round(float(analog_bias), 2),
        "analog_confidence": round(float(confidence), 3),
        "analogs_used": [{"date": d, "distance": round(dist, 3),
                          "error_f": round(float(err), 2)}
                         for dist, err, d in top],
    }
