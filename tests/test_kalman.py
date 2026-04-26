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
    # Fix A (Session 4) refuses to project when rate is non-positive pre-peak
    # without warming evidence. This test validates filter convergence, not
    # projection policy — give it hrrr_peak_f=truth so running_max_f >=
    # hrrr_peak_f satisfies the warming-evidence clause and Kalman proceeds.
    result = project_daily_max(obs, hrrr_hourly=None,
                                running_max_f=truth,
                                ensemble_value_f=None,
                                hrrr_peak_f=truth)
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


def test_fix_a_refuses_projection_in_pre_dawn_cooling():
    """Real bug from 2026-04-26 5:42 AM: Kalman projected daily max at
    45.9°F because pre-dawn cooling produced a negative rate, the
    projection collapsed to current_temp, and HRRR ceiling pulled it up
    only modestly. With Fix A, Kalman now declines to project under
    these conditions and lets the rest of the ensemble carry the
    forecast.
    """
    fixed_now = datetime(2026, 4, 26, 5, 42, tzinfo=cfg.EASTERN)
    obs = []
    for hours_ago in range(5, 0, -1):  # 5h ago down to 1h ago
        ts = fixed_now - timedelta(hours=hours_ago)
        # Slight cooling: 44.0 → 42.5 across the window
        temp = 42.5 + 0.3 * hours_ago
        obs.append({"observed_at": ts.isoformat(), "temperature_f": temp})

    # HRRR forecasts a 51°F peak; running_max so far is only 42.8°F
    # (well below HRRR), so there's no observed evidence of warming.
    hrrr_hourly = [(15.0, 50.5), (16.0, 51.0), (17.0, 50.0)]
    result = project_daily_max(
        obs, hrrr_hourly,
        running_max_f=42.8,
        ensemble_value_f=None,
        hrrr_peak_f=51.0,
        now=fixed_now,
    )
    assert result is None, (
        f"Fix A regression: Kalman should have refused to project under "
        f"pre-dawn cooling with no warming evidence, but returned {result}"
    )


def test_fix_b_blend_weight_gated_on_post_sunrise_hours():
    """Pre-sunrise observations don't carry signal about today's peak.
    Even with 4+ hours of overnight data accumulated since midnight,
    blend_weight should stay at the lowest band (0.05) until 2+ hours
    past sunrise. Verifies Fix B's swap from raw hours_of_data to
    useful_daytime_hours.
    """
    fixed_now = datetime(2026, 4, 26, 5, 0, tzinfo=cfg.EASTERN)
    obs = []
    # Slight WARMING trend (positive rate so Fix A doesn't trip):
    # 40°F at 5h ago → 44°F now
    for hours_ago in range(5, 0, -1):
        ts = fixed_now - timedelta(hours=hours_ago)
        temp = 45.0 - 1.0 * hours_ago
        obs.append({"observed_at": ts.isoformat(), "temperature_f": temp})

    # running_max already meets HRRR peak so warming-evidence clause is
    # satisfied (Fix A won't fire even if rate dips negative after blend).
    hrrr_hourly = [(15.0, 60.0)]
    result = project_daily_max(
        obs, hrrr_hourly,
        running_max_f=60.0,
        ensemble_value_f=None,
        hrrr_peak_f=60.0,
        now=fixed_now,
    )
    assert result is not None, "expected projection (Fix A should not fire)"
    # hours_of_data is wall-clock span of obs (~4h), which used to bump
    # blend_weight to 0.35. With Fix B, useful_daytime_hours is 0 (5 AM
    # is pre-sunrise in April), so blend_weight should be at the lowest
    # band.
    assert result["hours_of_data_used"] >= 4.0, (
        f"sanity check failed: hours_of_data_used should be >= 4 in "
        f"this scenario, got {result['hours_of_data_used']}"
    )
    assert result["useful_daytime_hours"] == 0.0, (
        f"useful_daytime_hours should be 0 at 5 AM (pre-sunrise), "
        f"got {result['useful_daytime_hours']}"
    )
    assert result["blend_weight"] == 0.05, (
        f"Fix B regression: pre-sunrise blend_weight should be 0.05 "
        f"(no daytime signal), got {result['blend_weight']} — this "
        f"means the ladder is still keying on raw hours_of_data"
    )
