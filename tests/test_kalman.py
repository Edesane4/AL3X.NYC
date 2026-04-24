"""Core unit tests for Kalman intraday tracker.

Covers:
  * Linear ramp → projected_max close to linear extrapolation
  * Convergence around a known truth with synthetic noise
  * <3 obs → returns None
  * Bug A verification: at 0h of data the blended rate weights the
    climatological prior ~90% against the filter's zero initial rate.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from al3x import config as cfg
from al3x.kalman_tracker import project_daily_max


def _obs(ts: datetime, temp_f: float):
    return {"observed_at": ts.isoformat(), "temperature_f": temp_f}


def test_linear_ramp_projection_matches_extrapolation():
    now = datetime.now(cfg.EASTERN).replace(hour=9, minute=0, second=0,
                                             microsecond=0)
    obs = [
        _obs(now - timedelta(hours=2), 60.0),
        _obs(now - timedelta(hours=1), 62.0),
        _obs(now, 64.0),
    ]
    # Fake HRRR hourly: peak 4h from now at 72°F.
    peak_hour = now.hour + 4.0
    hrrr = [(peak_hour - 2.0, 68.0), (peak_hour, 72.0),
            (peak_hour + 1.0, 70.0)]
    result = project_daily_max(obs, hrrr, running_max_f=64.0,
                                ensemble_value_f=None,
                                hrrr_peak_f=72.0)
    assert result is not None
    # Linear extrapolation from the last obs (64°F) at 2°F/hr over the
    # remaining hours-to-peak should land near 64 + 2*hours_to_peak.
    # With HRRR ceiling blending we expect the result to be within the
    # HRRR+3°F band. A coarse bound is enough here.
    assert 64.0 <= result["projected_max_f"] <= 75.0


def test_noisy_observations_converge_near_truth():
    random.seed(42)
    truth = 75.0
    now = datetime.now(cfg.EASTERN).replace(hour=14, minute=0, second=0,
                                             microsecond=0)
    obs = [
        _obs(now - timedelta(hours=5 - i),
             truth + random.gauss(0.0, 0.4))
        for i in range(6)
    ]
    result = project_daily_max(obs, hrrr_hourly=None,
                                running_max_f=truth,
                                ensemble_value_f=None,
                                hrrr_peak_f=None)
    assert result is not None
    # Filter's latest temp estimate should be within ±1°F of truth
    assert abs(result["current_temp_f"] - truth) <= 1.0


def test_fewer_than_three_obs_returns_none():
    now = datetime.now(cfg.EASTERN)
    obs = [_obs(now, 70.0), _obs(now - timedelta(hours=1), 69.0)]
    result = project_daily_max(obs, None, None, None)
    assert result is None


def test_bug_a_prior_rate_weights_ninety_percent_at_zero_hours():
    """At 0h of accumulated data, the Kalman rate is initialized to 0
    and should be weighted 10%; the climatological prior should be
    weighted 90%. The blended rate therefore approaches 0.9 * prior."""
    now = datetime.now(cfg.EASTERN).replace(hour=8, minute=0, second=0,
                                             microsecond=0)
    # Three obs at the exact same instant collapse hours_of_data to 0.
    obs = [_obs(now, 60.0), _obs(now, 60.0), _obs(now, 60.0)]
    hrrr = [(12.0, 70.0), (15.0, 75.0)]
    prior_rate = 2.0   # °F/hr climatological rise rate
    result = project_daily_max(obs, hrrr, running_max_f=60.0,
                                ensemble_value_f=None,
                                hrrr_peak_f=75.0,
                                prior_rate=prior_rate)
    assert result is not None
    assert result["prior_rate_used"] is True
    # alpha = min(0.90, 0.1 + 0.075*0) = 0.1. Blended rate ≈ 0.1*0 +
    # 0.9*prior = 0.9 * 2.0 = 1.8 °F/hr. The earlier bug's formula
    # would have yielded 0.3*0 + 0.7*2.0 = 1.4 °F/hr, so we also assert
    # the strict lower bound.
    assert result["rate_f_per_hr"] > 1.5, (
        f"Bug A regression: rate {result['rate_f_per_hr']:.2f}°F/hr is "
        "too low — prior weight should be ~90% at 0h of data."
    )
    # Upper bound sanity: can't exceed the prior itself.
    assert result["rate_f_per_hr"] <= prior_rate + 1e-6
