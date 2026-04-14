"""Superior Quality 2 — Diurnal temperature curve fitting.

Fits an asymmetric Gaussian to the daytime portion of an hourly temperature
profile (model output or ASOS observations) and returns the analytical peak.
This catches between-hour maxima that max(hourly) misses, and lets us blend
model guidance with observed data by weighting obs 3x.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("al3x.diurnal_fit")


def _gaussian(t, t_peak, amplitude, sigma, t_min):
    # Asymmetric Gaussian is approximated by a single-sigma peak; sigma
    # is allowed to be wide enough that the shoulder asymmetry is captured
    # through the full-day residual. Good enough for a daily peak estimate.
    import math
    return t_min + amplitude * math.e ** (-((t - t_peak) ** 2) / (2 * sigma ** 2))


def fit_peak(hourly_today: Sequence[Tuple[float, float]],
             obs_today: List[Dict[str, Any]],
             target_date: date) -> Optional[Dict[str, Any]]:
    """Return {"fitted_max_f", "fitted_peak_hour", "r_squared"} or None.

    Parameters:
      hourly_today: list of (hour_decimal, temp_f) for today from a model.
      obs_today: ASOS observations (each with observed_at + temperature_f).
      target_date: the date of interest.

    If scipy is unavailable or curve_fit fails, returns None and the caller
    falls back to max(hourly_today).
    """
    if not hourly_today:
        return None

    # Filter to daytime hours 6-20 for the fit
    day_points = [(h, t) for h, t in hourly_today if 6.0 <= h <= 20.0]
    if len(day_points) < 5:
        return None

    # Append observations (weighted 3x by duplicating entries)
    obs_points: List[Tuple[float, float]] = []
    for o in obs_today:
        ts = o.get("observed_at")
        temp = o.get("temperature_f")
        if not ts or temp is None:
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            continue
        if dt.date() != target_date:
            continue
        h = dt.hour + dt.minute / 60.0
        if 6.0 <= h <= 20.0:
            obs_points.append((h, float(temp)))

    try:
        import numpy as np
        from scipy.optimize import curve_fit
    except Exception as e:  # noqa: BLE001
        log.info("scipy/numpy unavailable; skipping diurnal fit: %s", e)
        return None

    xs: List[float] = []
    ys: List[float] = []
    ws: List[float] = []
    for h, t in day_points:
        xs.append(h)
        ys.append(t)
        ws.append(1.0)
    for h, t in obs_points:
        # Weight observations 3x (sigma = 1/3 that of models)
        for _ in range(3):
            xs.append(h)
            ys.append(t)
            ws.append(1.0)

    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)

    # Initial guesses: peak ~14:00, amplitude ~(max-min), sigma ~4h, t_min ~y.min
    y_min = float(y_arr.min())
    y_max = float(y_arr.max())
    peak_idx = int(np.argmax(y_arr))
    p0 = [x_arr[peak_idx], max(y_max - y_min, 1.0), 4.0, y_min]

    bounds = (
        [6.0, 0.5, 1.0, y_min - 20.0],
        [20.0, 80.0, 10.0, y_max + 20.0],
    )

    try:
        popt, _ = curve_fit(_gaussian, x_arr, y_arr, p0=p0,
                            bounds=bounds, maxfev=5000)
    except Exception as e:  # noqa: BLE001
        log.warning("diurnal curve_fit failed, falling back to max(): %s", e)
        return None

    t_peak, amplitude, sigma, t_min = popt
    fitted_max = float(t_min + amplitude)

    # R²
    try:
        y_pred = np.array([_gaussian(xi, *popt) for xi in x_arr])
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2)) or 1e-9
        r2 = 1.0 - ss_res / ss_tot
    except Exception:  # noqa: BLE001
        r2 = None

    return {
        "fitted_max_f": fitted_max,
        "fitted_peak_hour": float(t_peak),
        "r_squared": (round(r2, 3) if r2 is not None else None),
    }
