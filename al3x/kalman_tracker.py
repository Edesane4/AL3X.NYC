"""Superior Quality 1 — 1D Kalman filter on intraday temperature.

State vector: [current_temperature, rate_of_change_per_hour].
Measurement: each ASOS observation is a noisy read of current_temperature.

Projects the daily max by extrapolating from the filtered state to the
expected peak hour (taken from HRRR hourly profile). Only engages when at
least 3 observations exist.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import config as cfg

log = logging.getLogger("al3x.kalman")

# Process noise (temperature rate of change variance in (°F/hr)^2)
Q_RATE = 0.5
# Measurement noise (ASOS 2m temperature noise in °F)
R_MEAS = 0.3


def _parse_hour(iso_ts: str) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(iso_ts)
    except Exception:
        return None
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def _parse_dt(iso_ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(iso_ts)
    except Exception:
        return None


def project_daily_max(obs_today: List[Dict[str, Any]],
                       hrrr_hourly: Optional[Sequence[Tuple[float, float]]],
                       running_max_f: Optional[float],
                       ensemble_value_f: Optional[float],
                       hrrr_peak_f: Optional[float] = None,
                       prior_rate: Optional[float] = None
                       ) -> Optional[Dict[str, Any]]:
    """Return {"projected_max_f", "kalman_uncertainty_f", "hours_of_data_used",
    "blend_weight"} or None.

    Requires at least 3 valid observations for today. Returns None when we
    can't run the filter so the ensemble proceeds without it.
    """
    valid: List[Tuple[datetime, float]] = []
    for o in obs_today:
        ts = o.get("observed_at")
        temp = o.get("temperature_f")
        if not ts or temp is None:
            continue
        dt = _parse_dt(ts)
        if dt is None:
            continue
        valid.append((dt, float(temp)))

    if len(valid) < 3:
        return None

    valid.sort(key=lambda x: x[0])
    now = datetime.now(cfg.EASTERN)

    # Initialize state with first observation, zero rate.
    x = [valid[0][1], 0.0]          # [temp, rate_°F_per_hour]
    # Covariance (diagonal): moderate initial uncertainty.
    P = [[4.0, 0.0], [0.0, 1.0]]

    prev_dt = valid[0][0]
    for dt, z in valid[1:]:
        dt_h = (dt - prev_dt).total_seconds() / 3600.0
        if dt_h <= 0:
            continue
        prev_dt = dt

        # --- Predict
        # State: x_new = [temp + rate*dt, rate]
        x = [x[0] + x[1] * dt_h, x[1]]
        # F = [[1, dt], [0, 1]]; Q diag(0, Q_RATE*dt)
        P = _f_p_ft(P, dt_h)
        P[1][1] += Q_RATE * dt_h
        P[0][0] += Q_RATE * dt_h * dt_h * 0.25  # small temp-cross term

        # --- Update
        # H = [1, 0]; y = z - H*x; S = H*P*H' + R; K = P*H'/S
        y = z - x[0]
        s = P[0][0] + R_MEAS
        k0 = P[0][0] / s
        k1 = P[1][0] / s
        x = [x[0] + k0 * y, x[1] + k1 * y]
        P00 = (1 - k0) * P[0][0]
        P01 = (1 - k0) * P[0][1]
        P10 = P[1][0] - k1 * P[0][0]
        P11 = P[1][1] - k1 * P[0][1]
        P = [[P00, P01], [P10, P11]]

    temp, rate = x
    uncertainty = float(max(P[0][0], 0.0)) ** 0.5

    # Find expected peak hour from HRRR
    peak_hour = 15.0
    if hrrr_hourly:
        try:
            peak_h, _ = max(hrrr_hourly, key=lambda x: x[1])
            peak_hour = float(peak_h)
        except Exception:  # noqa: BLE001
            pass

    current_hour = now.hour + now.minute / 60.0
    hours_to_peak = peak_hour - current_hour

    # Climatological rate prior blending (Quant 2). As hours_of_data grows,
    # trust the Kalman rate more; when data is thin, lean on climatology.
    hours_of_data = (valid[-1][0] - valid[0][0]).total_seconds() / 3600.0
    prior_used = False
    if prior_rate is not None:
        alpha = min(0.95, 0.3 + 0.065 * hours_of_data)
        rate = alpha * rate + (1 - alpha) * float(prior_rate)
        prior_used = True

    # Projection rule:
    #  - rate>0 and haven't hit peak: linearly extrapolate
    #  - rate<=0 and past 1 PM: use running_max as the projected peak
    #  - rate<=0 and before 1 PM: trust the filter temp + small upward drift
    if rate > 0 and hours_to_peak > 0:
        projected = temp + rate * hours_to_peak
    elif current_hour >= 13.0 and running_max_f is not None:
        projected = max(running_max_f, temp)
    else:
        projected = temp + max(rate, 0.0) * max(hours_to_peak, 0.0)

    # MATH GAP 2 — soft HRRR ceiling. The linear Kalman projection can
    # overshoot when the temperature curve flattens before the peak.
    # Blend toward HRRR's peak (with weight growing as we accumulate data)
    # and hard-cap at HRRR+3 °F.
    ceiling_weight = None
    if hrrr_peak_f is not None:
        ceiling_weight = min(0.7, 0.2 + 0.05 * hours_of_data)
        projected = ((1 - ceiling_weight) * projected
                     + ceiling_weight * max(projected, float(hrrr_peak_f)))
        projected = min(projected, float(hrrr_peak_f) + 3.0)

    if running_max_f is not None:
        projected = max(projected, running_max_f)

    if hours_of_data < 4.0:
        blend_weight = 0.15
    elif hours_of_data < 8.0:
        blend_weight = 0.35
    else:
        blend_weight = 0.55

    return {
        "projected_max_f": float(projected),
        "kalman_uncertainty_f": float(uncertainty),
        "hours_of_data_used": round(hours_of_data, 2),
        "blend_weight": blend_weight,
        "current_temp_f": float(temp),
        "rate_f_per_hr": float(rate),
        "peak_hour": float(peak_hour),
        "ceiling_weight": ceiling_weight,
        "prior_rate_used": prior_used,
    }


def _f_p_ft(P: List[List[float]], dt_h: float) -> List[List[float]]:
    """Compute F * P * F^T for F=[[1,dt],[0,1]]."""
    # F * P
    a = P[0][0] + dt_h * P[1][0]
    b = P[0][1] + dt_h * P[1][1]
    c = P[1][0]
    d = P[1][1]
    # (F*P) * F^T  (F^T = [[1,0],[dt,1]])
    return [
        [a + b * dt_h, b],
        [c + d * dt_h, d],
    ]
