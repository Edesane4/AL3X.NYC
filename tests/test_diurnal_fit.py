"""Core unit tests for diurnal fit (Gaussian + spline competition)."""

from __future__ import annotations

import logging
import math
from datetime import date


def _asym_gauss(h, t_peak, amplitude, sigma_left, sigma_right, t_min):
    sigma = sigma_left if h <= t_peak else sigma_right
    return t_min + amplitude * math.exp(-((h - t_peak) ** 2) / (2 * sigma ** 2))


def test_asymmetric_gaussian_peak_recovered_within_half_hour():
    """Synthetic asymmetric-Gaussian samples → fit_peak should recover
    the true peak time to within 0.5h."""
    try:
        from al3x.diurnal_fit import fit_peak
    except Exception:  # scipy missing
        import pytest
        pytest.skip("scipy/numpy unavailable")

    true_peak = 15.5
    amp = 25.0
    sigma_left = 3.5
    sigma_right = 5.0
    t_min = 55.0
    hours = [h * 0.5 for h in range(12, 41)]  # 6.0 .. 20.0 step 0.5
    hourly_today = [(h, _asym_gauss(h, true_peak, amp,
                                      sigma_left, sigma_right, t_min))
                    for h in hours]
    result = fit_peak(hourly_today, obs_today=[],
                      target_date=date(2026, 7, 15))
    assert result is not None
    assert "fitted_peak_hour" in result
    assert abs(result["fitted_peak_hour"] - true_peak) <= 0.5, (
        f"peak recovery off by more than 0.5h: "
        f"fit={result['fitted_peak_hour']:.2f}, true={true_peak}"
    )


def test_noisy_profile_does_not_overpenalize_gaussian(caplog):
    """A noisy synthetic profile that the spline fits better must not
    produce a spurious warning about the Gaussian being wrong — the
    R²-driven spline-wins log message is INFO level and descriptive."""
    try:
        from al3x.diurnal_fit import fit_peak
    except Exception:
        import pytest
        pytest.skip("scipy/numpy unavailable")

    import random
    random.seed(7)
    # Build a noisy plateau that a cubic spline will track better
    # than a smooth asymmetric Gaussian.
    hourly_today = []
    for h_i in range(6, 21):
        # Flat-topped daily curve with noise
        base = 70.0 + (5.0 if 12 <= h_i <= 16 else 0.0)
        hourly_today.append((float(h_i), base + random.gauss(0, 0.8)))

    caplog.set_level(logging.INFO, logger="al3x.diurnal_fit")
    result = fit_peak(hourly_today, obs_today=[],
                      target_date=date(2026, 7, 15))
    assert result is not None
    # If spline wins, log should note it at INFO — never WARNING.
    for rec in caplog.records:
        if rec.name == "al3x.diurnal_fit":
            assert rec.levelno <= logging.INFO, (
                f"unexpected WARNING from diurnal_fit on noisy profile: "
                f"{rec.getMessage()}"
            )
