"""FIX 1 — total correction magnitude cap at ±3°F + sigma widening.

Evidence: On April 20-21, corrections stacked to -6°F during a post-
frontal regime shift, making the final forecast 2.4-3.6°F worse than
night-before. No previous code path capped the total.
"""

from __future__ import annotations

from datetime import date

import pytest

from al3x import forecaster as fc
from al3x.forecaster import BiasLive, _apply_corrections


@pytest.fixture(autouse=True)
def _disable_session2_blanket_suppression(monkeypatch):
    """The FIX 1 cap runs AFTER the Session 2 blanket correction
    suppressor, so under default module state every delta is zeroed
    before the cap ever sees the sum. Clear the suppression set inside
    this test module so we can exercise the Session 1 cap logic in
    isolation — the cap itself is still live code and re-activates
    when Session 2 flips corrections back on.
    """
    monkeypatch.setattr(fc, "_SESSION_2_SUPPRESSED", set())


def _fake_regime(**overrides):
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


def _bias_with(**values):
    return BiasLive(values=values)


def test_under_cap_no_scaling():
    """Total -2.5°F → no scaling, cap not fired."""
    bias = _bias_with(sea_breeze_full=-2.5)
    regime = _fake_regime(sea_breeze_full=True)
    total, corrections = _apply_corrections(
        date(2026, 7, 15), regime, spread=2.0, bias=bias,
    )
    assert abs(total - (-2.5)) < 1e-6
    assert "_meta" not in corrections
    assert corrections["sea_breeze"]["delta"] == -2.5


def test_over_cap_negative_scaled_to_minus_three():
    """Total -6.0°F → scaled to -3.0°F proportionally; cap_fired=True."""
    bias = _bias_with(sea_breeze_full=-3.0, precip_heavy=-3.0)
    regime = _fake_regime(sea_breeze_full=True, precip_heavy=True,
                          any_precip_peak=True)
    total, corrections = _apply_corrections(
        date(2026, 7, 15), regime, spread=2.0, bias=bias,
    )
    assert abs(total - (-3.0)) < 1e-6
    assert corrections["_meta"]["cap_fired"] is True
    assert abs(corrections["_meta"]["pre_cap_total"] - (-6.0)) < 1e-6
    assert abs(corrections["_meta"]["scale"] - 0.5) < 1e-6
    # Each scaled component should be half of its original
    assert abs(corrections["sea_breeze"]["delta"] - (-1.5)) < 1e-6
    assert abs(corrections["precip"]["delta"] - (-1.5)) < 1e-6
    # spread_penalty delta is always 0.0; not modified by scaling
    assert corrections["spread_penalty"]["delta"] == 0.0


def test_over_cap_positive_scaled_to_plus_three():
    """Total +5.0°F → scaled to +3.0°F proportionally; cap_fired=True."""
    bias = _bias_with(uhi_clear_calm=3.0, sea_breeze_nw_boost=2.0)
    regime = _fake_regime(calm_clear=True, wind_nw_all_day=True)
    total, corrections = _apply_corrections(
        date(2026, 7, 15), regime, spread=2.0, bias=bias,
    )
    assert abs(total - 3.0) < 1e-6
    assert corrections["_meta"]["cap_fired"] is True
    assert abs(corrections["_meta"]["pre_cap_total"] - 5.0) < 1e-6
    expected_scale = 3.0 / 5.0
    assert abs(corrections["_meta"]["scale"] - expected_scale) < 1e-6
    assert abs(corrections["uhi"]["delta"] - 3.0 * expected_scale) < 1e-6
    assert abs(corrections["sea_breeze"]["delta"] - 2.0 * expected_scale) < 1e-6


def test_zero_total_no_issues():
    """Total 0.0°F → no scaling, no _meta key, no exception."""
    bias = _bias_with()  # all defaults; regime below keeps everything 0
    regime = _fake_regime()
    total, corrections = _apply_corrections(
        date(2026, 7, 15), regime, spread=2.0, bias=bias,
    )
    assert total == 0.0
    assert "_meta" not in corrections


def test_suppressed_delta_is_not_scaled():
    """When spread >= 6, other corrections are suppressed to 0 delta but
    ``suppressed_delta`` records what would have been applied. That field
    must survive the cap scaling unchanged for honest attribution."""
    bias = _bias_with(sea_breeze_full=-4.0, precip_heavy=-4.0)
    regime = _fake_regime(sea_breeze_full=True, precip_heavy=True,
                          any_precip_peak=True)
    # High spread triggers suppression; deltas become 0 so cap won't fire
    total, corrections = _apply_corrections(
        date(2026, 7, 15), regime, spread=8.0, bias=bias,
    )
    assert total == 0.0
    # suppressed_delta preserves the counterfactual unscaled
    assert corrections["sea_breeze"]["suppressed_delta"] == -4.0
    assert corrections["precip"]["suppressed_delta"] == -4.0
