"""Session 2 — blanket correction suppression preserves attribution.

Verifies that a regime which would have triggered cloud_timing +
precip still produces correction payloads with suppressed=True,
suppressed_delta matching the would-have-applied value, delta=0.0.
"""

from __future__ import annotations

from datetime import date

from al3x.forecaster import BiasLive, _apply_corrections


def _regime(**overrides):
    base = {
        "sea_breeze_shift": False, "sea_breeze_full": False,
        "wind_nw_all_day": False, "cloud_morning_increase": False,
        "cloud_afternoon_clearing": False, "any_precip_peak": False,
        "precip_heavy": False, "inversion_hint": False,
        "inversion_strength": "weak", "calm_clear": False,
        "sustained_windy": False, "cloud_avg": None, "max_wind_kt": None,
        "sea_breeze_confirmed": False, "sea_breeze_gradient_f": None,
        "uhi_strong": False, "uhi_signal_f": None,
    }
    base.update(overrides)
    return base


def test_cloud_timing_and_precip_both_suppressed_with_delta_tracked():
    """Cloud-increase + heavy precip → both deltas would fire. Under
    Session 2 blanket suppression, final delta is 0.0 and the original
    would-have-applied delta lives in suppressed_delta."""
    bias = BiasLive(values={
        "cloud_increase_morning": -1.5,
        "precip_heavy": -2.0,
    })
    regime = _regime(cloud_morning_increase=True, precip_heavy=True,
                     any_precip_peak=True)
    total, corrections = _apply_corrections(
        date(2026, 7, 15), regime, spread=2.0, bias=bias,
    )
    # Total is 0 because all 5 corrections are suppressed
    assert total == 0.0

    ct = corrections["cloud_timing"]
    pd = corrections["precip"]

    assert ct["suppressed"] is True
    assert ct["delta"] == 0.0
    assert abs(ct["suppressed_delta"] - (-1.5)) < 1e-6
    assert "Session 2 blanket disable" in ct["reason"]

    assert pd["suppressed"] is True
    assert pd["delta"] == 0.0
    assert abs(pd["suppressed_delta"] - (-2.0)) < 1e-6
    assert "Session 2 blanket disable" in pd["reason"]


def test_suppressed_delta_survives_when_spread_branch_already_ran():
    """High spread (>=6) ALSO suppresses and writes suppressed_delta.
    Blanket suppression must not overwrite the existing record with 0."""
    bias = BiasLive(values={
        "cloud_increase_morning": -1.5,
        "precip_heavy": -2.0,
    })
    regime = _regime(cloud_morning_increase=True, precip_heavy=True,
                     any_precip_peak=True)
    total, corrections = _apply_corrections(
        date(2026, 7, 15), regime, spread=8.0, bias=bias,
    )
    assert total == 0.0
    # Spread-branch-written suppressed_delta must still reflect the
    # original values; it must not have been re-recorded as 0.
    assert abs(corrections["cloud_timing"]["suppressed_delta"] - (-1.5)) < 1e-6
    assert abs(corrections["precip"]["suppressed_delta"] - (-2.0)) < 1e-6
