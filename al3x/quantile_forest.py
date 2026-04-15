"""Quant Upgrade 1 — Quantile Regression Forest.

Trains a quantile-regression forest on the growing archive of
(feature_vector, error) pairs (from forecasts × cli_truth) and emits
calibrated p10 / p50 / p90 prediction intervals for the current
forecast state.

Graceful degradation:
- If `quantile-forest` is installed → native QRF.
- Else if `scikit-learn` is available → sklearn `RandomForestRegressor`
  fallback that approximates quantiles from the distribution of
  training residuals at each prediction's leaf node.
- Else `train()` returns False and `predict()` returns None — the
  ensemble continues without any QRF contribution.

Never raises. Never blocks the forecast.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfg

log = logging.getLogger("al3x.qrf")


FEATURE_NAMES = [
    "hrrr_f_norm", "ecmwf_f_norm", "gfs_mos_f_norm", "spread_norm",
    "lead_hours_norm", "month_sin", "month_cos",
    "sea_breeze_flag", "precip_flag", "cloud_avg_norm",
]
_DELTA_CAP = 5.0


def _norm(v: Optional[float], lo: float, hi: float) -> float:
    if v is None:
        return 0.5
    try:
        vv = float(v)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, (vv - lo) / max(hi - lo, 1e-6)))


def _feature_vector(hrrr_f: Optional[float], ecmwf_f: Optional[float],
                     gfs_mos_f: Optional[float], spread: Optional[float],
                     lead_hours: float, month: int,
                     sea_breeze: bool, precip: bool,
                     cloud_avg: Optional[float]) -> List[float]:
    m_rad = 2.0 * math.pi * (month / 12.0)
    return [
        _norm(hrrr_f, 0.0, 110.0),
        _norm(ecmwf_f, 0.0, 110.0),
        _norm(gfs_mos_f, 0.0, 110.0),
        _norm(spread, 0.0, 12.0),
        max(0.0, min(1.0, float(lead_hours) / 24.0)),
        (math.sin(m_rad) + 1.0) / 2.0,
        (math.cos(m_rad) + 1.0) / 2.0,
        1.0 if sea_breeze else 0.0,
        1.0 if precip else 0.0,
        _norm(cloud_avg, 0.0, 100.0),
    ]


def _fv_from_forecast_row(fc: Dict[str, Any]) -> Optional[List[float]]:
    """Build the 10-element training FV from a stored forecast row."""
    try:
        sources = fc.get("sources", {}) or {}
        extras = fc.get("extras", {}) or {}
        regime = extras.get("regime", {}) or {}
        hrrr_f = (sources.get("hrrr") or {}).get("value")
        ecmwf_f = (sources.get("ecmwf") or {}).get("value")
        gfs_f = (sources.get("gfs_mos") or {}).get("value")
        spread = extras.get("spread_f")
        lead = extras.get("lead_hours") or 12.0
        td = fc.get("target_date") or "2000-01-01"
        month = int(td.split("-")[1])
        sea_breeze = bool(regime.get("sea_breeze_shift")
                          or regime.get("sea_breeze_full"))
        precip = bool(regime.get("any_precip_peak")
                      or regime.get("precip_heavy"))
        cloud_avg = regime.get("cloud_avg")
        return _feature_vector(
            hrrr_f, ecmwf_f, gfs_f, spread,
            float(lead), month, sea_breeze, precip, cloud_avg,
        )
    except Exception as e:  # noqa: BLE001
        log.debug("qrf fv build failed: %s", e)
        return None


class QuantileForest:
    def __init__(self) -> None:
        self._model = None
        self._trained_date: Optional[date] = None
        self._n_training_samples: int = 0
        self._is_sklearn_fallback: bool = False
        # sklearn fallback: store y_train for leaf-sampling trick
        self._y_train = None

    @property
    def trained(self) -> bool:
        return self._model is not None

    def train(self, storage, min_samples: int = 30) -> bool:
        """Rebuild the forest from 365 days of night-before forecasts.

        Returns True on a successful fit, False if insufficient data or
        no ML library is available.
        """
        try:
            rows = storage.forecasts_with_truth(days=365,
                                                 prefer_mode="night_before")
        except Exception as e:  # noqa: BLE001
            log.info("QRF train: forecasts_with_truth failed: %s", e)
            return False

        pairs: List[Tuple[List[float], float]] = []
        for r in rows:
            cli_f = r.get("cli_f")
            final_f = r.get("final_f")
            if cli_f is None or final_f is None:
                continue
            fv = _fv_from_forecast_row(r)
            if fv is None:
                continue
            pairs.append((fv, float(cli_f) - float(final_f)))

        if len(pairs) < min_samples:
            log.info("QRF train: %d pairs (need %d) — not training yet",
                     len(pairs), min_samples)
            return False

        try:
            import numpy as np
        except Exception:  # noqa: BLE001
            log.info("QRF: numpy unavailable; skipping training")
            return False

        X = np.array([p[0] for p in pairs], dtype=float)
        y = np.array([p[1] for p in pairs], dtype=float)

        # Try native quantile-forest first
        try:
            from quantile_forest import RandomForestQuantileRegressor
            self._model = RandomForestQuantileRegressor(
                n_estimators=100, min_samples_leaf=5, random_state=42,
            )
            self._model.fit(X, y)
            self._is_sklearn_fallback = False
            self._y_train = None
            self._n_training_samples = len(pairs)
            self._trained_date = datetime.now(cfg.EASTERN).date()
            log.info("QRF trained (native) on %d samples", len(pairs))
            return True
        except Exception as e:  # noqa: BLE001
            log.info("quantile-forest unavailable (%s), trying sklearn", e)

        # Fallback: sklearn RandomForestRegressor + leaf-sample trick
        try:
            from sklearn.ensemble import RandomForestRegressor
            self._model = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=5, random_state=42,
            )
            self._model.fit(X, y)
            self._is_sklearn_fallback = True
            self._y_train = y
            self._X_train = X
            self._n_training_samples = len(pairs)
            self._trained_date = datetime.now(cfg.EASTERN).date()
            log.info("QRF trained (sklearn fallback) on %d samples",
                     len(pairs))
            return True
        except Exception as e:  # noqa: BLE001
            log.info("sklearn unavailable (%s); QRF disabled", e)
            self._model = None
            return False

    def predict(self, feature_vector: List[float]) -> Optional[Dict[str, Any]]:
        """Return {p10_delta, p50_delta, p90_delta, interval_width,
        n_training} or None if the model isn't trained."""
        if self._model is None:
            return None
        try:
            import numpy as np
        except Exception:  # noqa: BLE001
            return None

        try:
            x = np.array(feature_vector, dtype=float).reshape(1, -1)
            if not self._is_sklearn_fallback:
                # quantile-forest native multi-quantile predict
                p10 = float(self._model.predict(x, quantiles=0.10)[0])
                p50 = float(self._model.predict(x, quantiles=0.50)[0])
                p90 = float(self._model.predict(x, quantiles=0.90)[0])
            else:
                # sklearn fallback: pool the training residuals that fell
                # into the same leaf across all trees, then take
                # percentiles of that pooled distribution.
                leaves = self._model.apply(x)[0]
                pooled: List[float] = []
                for tree_idx, leaf in enumerate(leaves):
                    tree = self._model.estimators_[tree_idx]
                    train_leaves = tree.apply(self._X_train)
                    mask = train_leaves == leaf
                    pooled.extend(self._y_train[mask].tolist())
                if not pooled:
                    # All leaves empty — fall back to overall percentiles
                    pooled = self._y_train.tolist()
                arr = np.array(pooled, dtype=float)
                p10 = float(np.percentile(arr, 10))
                p50 = float(np.percentile(arr, 50))
                p90 = float(np.percentile(arr, 90))
        except Exception as e:  # noqa: BLE001
            log.info("QRF predict failed: %s", e)
            return None

        # Cap each quantile at ±5°F
        p10 = max(-_DELTA_CAP, min(_DELTA_CAP, p10))
        p50 = max(-_DELTA_CAP, min(_DELTA_CAP, p50))
        p90 = max(-_DELTA_CAP, min(_DELTA_CAP, p90))

        return {
            "p10_delta": round(p10, 2),
            "p50_delta": round(p50, 2),
            "p90_delta": round(p90, 2),
            "interval_width": round(p90 - p10, 2),
            "n_training": self._n_training_samples,
            "method": ("sklearn" if self._is_sklearn_fallback else "native"),
        }

    def calibration_stats(self, storage) -> Dict[str, Any]:
        """Return calibration diagnostics over the last 30 days of scores.

        Pinball loss per quantile q:
            L(q, y, yhat) = q*(y - yhat)  if y >= yhat
                          = (1-q)*(yhat - y)  otherwise
        Empirical coverage: fraction of days where cli_f ∈ [p10, p90].
        """
        out: Dict[str, Any] = {
            "pinball_p10": None, "pinball_p50": None, "pinball_p90": None,
            "empirical_coverage": None, "samples": 0,
        }
        try:
            rows = storage.forecasts_with_truth(
                days=30, prefer_mode="night_before",
            )
        except Exception:  # noqa: BLE001
            return out

        p10s: List[float] = []
        p50s: List[float] = []
        p90s: List[float] = []
        covered = 0
        total = 0
        for r in rows:
            final_f = r.get("final_f")
            cli_f = r.get("cli_f")
            if final_f is None or cli_f is None:
                continue
            extras = r.get("extras") or {}
            qrf = extras.get("qrf") or {}
            p10 = qrf.get("p10_delta")
            p50 = qrf.get("p50_delta")
            p90 = qrf.get("p90_delta")
            if p10 is None or p50 is None or p90 is None:
                continue
            y = float(cli_f) - float(final_f)   # actual delta
            p10s.append(_pinball(0.10, y, float(p10)))
            p50s.append(_pinball(0.50, y, float(p50)))
            p90s.append(_pinball(0.90, y, float(p90)))
            total += 1
            if float(p10) <= y <= float(p90):
                covered += 1

        def _mean(xs: List[float]) -> Optional[float]:
            return round(sum(xs) / len(xs), 3) if xs else None

        coverage = (covered / total) if total else None
        if coverage is not None and coverage < 0.70:
            log.warning("QRF coverage %.0f%% (target 80%%) on %d samples",
                        coverage * 100, total)
        out["pinball_p10"] = _mean(p10s)
        out["pinball_p50"] = _mean(p50s)
        out["pinball_p90"] = _mean(p90s)
        out["empirical_coverage"] = (round(coverage, 3)
                                      if coverage is not None else None)
        out["samples"] = total
        return out


def _pinball(q: float, y: float, yhat: float) -> float:
    return q * (y - yhat) if y >= yhat else (1 - q) * (yhat - y)
