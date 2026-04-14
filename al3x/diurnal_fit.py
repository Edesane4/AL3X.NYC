"""Diurnal temperature curve fitting (Superior Quality 2 + Math Gap 1).

Fits an ASYMMETRIC (split-sigma) Gaussian to the daytime portion of an
hourly temperature profile (model output or ASOS observations) and
returns the analytical peak. The morning rise is typically steeper than
the afternoon fall, so a single-sigma Gaussian biases both the peak
magnitude and peak time. Split-sigma captures this cleanly.

Proper observation weighting is enforced by passing a `sigma=` array to
`scipy.optimize.curve_fit` (BUG 4) — model points at sigma=1, obs at
sigma=1/3. R² is reported honestly on the unique model points only.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("al3x.diurnal_fit")


def _asymmetric_gaussian(t, t_peak, amplitude, sigma_left, sigma_right, t_min):
    """Split-sigma Gaussian. t may be a scalar or numpy array.

    sigma_left  — standard deviation used when t <= t_peak (morning side)
    sigma_right — standard deviation used when t >  t_peak (afternoon side)
    """
    try:
        import numpy as np
        sigma = np.where(t <= t_peak, sigma_left, sigma_right)
        return t_min + amplitude * np.exp(-((t - t_peak) ** 2) / (2 * sigma ** 2))
    except Exception:
        # Fallback scalar path (used for single-point eval in R²)
        sigma = sigma_left if t <= t_peak else sigma_right
        return t_min + amplitude * math.exp(-((t - t_peak) ** 2)
                                              / (2 * sigma ** 2))


def fit_peak(hourly_today: Sequence[Tuple[float, float]],
             obs_today: List[Dict[str, Any]],
             target_date: date) -> Optional[Dict[str, Any]]:
    """Return a dict with:
      fitted_max_f, fitted_peak_hour, r_squared, sigma_left, sigma_right
    or None if scipy is missing / too few points / fit fails to converge.
    """
    if not hourly_today:
        return None

    day_points = [(h, t) for h, t in hourly_today if 6.0 <= h <= 20.0]
    if len(day_points) < 5:
        return None

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

    # Build the fitting arrays. Model points carry sigma=1.0, observations
    # carry sigma=1/3 (i.e. 3× tighter weighting).
    xs: List[float] = []
    ys: List[float] = []
    ws: List[float] = []
    for h, t in day_points:
        xs.append(h)
        ys.append(t)
        ws.append(1.0)
    for h, t in obs_points:
        xs.append(h)
        ys.append(t)
        ws.append(3.0)   # 3× weight relative to model points

    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    # curve_fit expects sigma ∝ 1/weight; give it the inverse of our weight
    sigma_arr = np.array([1.0 / w if w > 0 else 1.0 for w in ws], dtype=float)

    y_min = float(y_arr.min())
    y_max = float(y_arr.max())
    peak_idx = int(np.argmax(y_arr))
    # Initial guesses: peak near argmax; morning sigma ~3h (fast rise),
    # afternoon sigma ~5h (slower fall); amplitude ~(max-min); t_min ~y.min
    p0 = [float(x_arr[peak_idx]), max(y_max - y_min, 1.0), 3.0, 5.0, y_min]
    bounds = (
        [6.0, 0.5, 0.5, 1.0, y_min - 20.0],
        [20.0, 80.0, 8.0, 12.0, y_max + 20.0],
    )

    try:
        popt, _ = curve_fit(
            _asymmetric_gaussian, x_arr, y_arr, p0=p0, bounds=bounds,
            sigma=sigma_arr, absolute_sigma=True, maxfev=5000,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("diurnal curve_fit failed, falling back to max(): %s", e)
        return None

    t_peak, amplitude, sigma_left, sigma_right, t_min = popt
    fitted_max = float(t_min + amplitude)

    # BUG 4 — honest R² computed ONLY over unique model points
    # (day_points), not over the weighted-duplicate array. We also
    # exclude obs_points here so the reported fit quality isn't inflated
    # by the perfectly-matched observations.
    try:
        day_x = np.array([h for h, _ in day_points], dtype=float)
        day_y = np.array([t for _, t in day_points], dtype=float)
        y_pred = _asymmetric_gaussian(
            day_x, t_peak, amplitude, sigma_left, sigma_right, t_min
        )
        ss_res = float(np.sum((day_y - y_pred) ** 2))
        ss_tot = float(np.sum((day_y - np.mean(day_y)) ** 2)) or 1e-9
        r2 = 1.0 - ss_res / ss_tot
    except Exception:  # noqa: BLE001
        r2 = None

    return {
        "fitted_max_f": fitted_max,
        "fitted_peak_hour": float(t_peak),
        "r_squared": (round(r2, 3) if r2 is not None else None),
        "sigma_left": round(float(sigma_left), 2),
        "sigma_right": round(float(sigma_right), 2),
    }
