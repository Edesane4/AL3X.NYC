"""Quant 2 — ASOS climatological rate-of-rise prior for Kalman.

Computes the historical mean temperature-rise rate at KNYC as a function
of (month, sky_cover, wind_speed) from stored observations. Used by the
Kalman filter to stabilize rate estimates when few intraday observations
are available (morning cycles).

Fully degrade-gracefully: if no history is built, `get_prior_rate()`
returns None and the Kalman filter proceeds with its raw rate.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfg

log = logging.getLogger("al3x.climatology")


def _sky_bin(sky_cover_pct: Optional[float]) -> str:
    if sky_cover_pct is None:
        return "unknown"
    if sky_cover_pct < 30:
        return "clear"
    if sky_cover_pct < 70:
        return "partial"
    return "overcast"


def _wind_bin(wind_speed_kt: Optional[float]) -> str:
    if wind_speed_kt is None:
        return "unknown"
    if wind_speed_kt < 7:
        return "calm"
    if wind_speed_kt < 15:
        return "moderate"
    return "strong"


class RateClimatology:
    """Historical (month, sky, wind) → mean rate-of-rise lookup."""

    def __init__(self) -> None:
        self._table: Dict[Tuple[int, str, str], float] = {}
        self._fallback_mean: Optional[float] = None
        self._built = False
        self._samples: Dict[Tuple[int, str, str], int] = {}

    @property
    def ready(self) -> bool:
        return self._built and bool(self._table)

    def build(self, storage, min_obs_days: int = 20) -> bool:
        """Populate the rate table from stored observations.

        Returns True on success, False if insufficient data. Safe to call
        repeatedly; each call fully rebuilds from scratch.
        """
        try:
            all_obs = storage.observations_all_days(limit_days=365)
        except Exception as e:  # noqa: BLE001
            log.info("climatology build failed loading obs: %s", e)
            return False
        if not all_obs:
            return False

        # Group observations by local calendar date
        by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for o in all_obs:
            ts = o.get("observed_at")
            if not ts:
                continue
            by_day[ts[:10]].append(o)

        # For each day with >=6 observations, compute rise rate
        rates_per_bin: Dict[Tuple[int, str, str], List[float]] = defaultdict(list)
        all_rates: List[float] = []

        for day_str, obs in by_day.items():
            if len(obs) < 6:
                continue
            # Parse datetime + temperatures
            points: List[Tuple[datetime, float, Optional[float],
                               Optional[float]]] = []
            for o in obs:
                try:
                    dt = datetime.fromisoformat(o["observed_at"])
                except Exception:
                    continue
                t_f = o.get("temperature_f")
                if t_f is None:
                    continue
                points.append((dt, float(t_f),
                                o.get("sky_cover_pct"),
                                o.get("wind_speed_kt")))
            if len(points) < 6:
                continue
            points.sort(key=lambda p: p[0])

            # Find the 6 AM observation (or nearest within ±1h)
            six_am = None
            for dt, t, _, _ in points:
                if 5 <= dt.hour <= 7:
                    six_am = (dt, t)
                    break
            if six_am is None:
                continue

            # Find the running_max reached that day
            peak_dt = None
            peak_t = points[0][1]
            for dt, t, _, _ in points:
                if t > peak_t:
                    peak_t = t
                    peak_dt = dt
            if peak_dt is None or peak_dt <= six_am[0]:
                continue

            hours = (peak_dt - six_am[0]).total_seconds() / 3600.0
            if hours < 1.0:
                continue
            rate = (peak_t - six_am[1]) / hours

            # Bin by month + mean daytime sky cover + mean daytime wind
            month = peak_dt.month
            daytime_skies = [p[2] for p in points
                             if 9 <= p[0].hour <= 17 and p[2] is not None]
            daytime_winds = [p[3] for p in points
                             if 9 <= p[0].hour <= 17 and p[3] is not None]
            mean_sky = (sum(daytime_skies) / len(daytime_skies)
                        if daytime_skies else None)
            mean_wind = (sum(daytime_winds) / len(daytime_winds)
                         if daytime_winds else None)
            key = (month, _sky_bin(mean_sky), _wind_bin(mean_wind))
            rates_per_bin[key].append(rate)
            all_rates.append(rate)

        if len(all_rates) < min_obs_days:
            log.info("climatology: only %d days of rates (need %d) — skip",
                     len(all_rates), min_obs_days)
            return False

        self._table = {k: sum(v) / len(v) for k, v in rates_per_bin.items()}
        self._samples = {k: len(v) for k, v in rates_per_bin.items()}
        self._fallback_mean = sum(all_rates) / len(all_rates)
        self._built = True
        log.info("climatology built: %d bins, %d total days, "
                 "fallback rate %.2f °F/hr",
                 len(self._table), len(all_rates), self._fallback_mean)
        return True

    def get_prior_rate(self, month: int,
                        sky_cover_pct: Optional[float],
                        wind_speed_kt: Optional[float]
                        ) -> Optional[float]:
        """Return mean rise rate (°F/hr) for the given condition, or None."""
        if not self.ready:
            return None
        key = (month, _sky_bin(sky_cover_pct), _wind_bin(wind_speed_kt))
        if key in self._table:
            return float(self._table[key])
        # Fallback to month-only if specific bin missing
        month_rates = [v for (m, _, _), v in self._table.items()
                       if m == month]
        if month_rates:
            return float(sum(month_rates) / len(month_rates))
        # Last resort: global fallback mean
        if self._fallback_mean is not None:
            return float(self._fallback_mean)
        return None

    def stats(self) -> Dict[str, Any]:
        """Dashboard-friendly summary."""
        return {
            "ready": self.ready,
            "num_bins": len(self._table),
            "fallback_mean": (round(self._fallback_mean, 2)
                              if self._fallback_mean is not None else None),
            "sample_counts": {f"{m}-{s}-{w}": n
                               for (m, s, w), n in self._samples.items()},
        }
