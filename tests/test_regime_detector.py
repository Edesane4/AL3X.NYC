"""FIX 2 — regime-shift detector thresholds + rate limit.

Evidence: 41 shifts in 8 days (5.1/day, 13 on a single day). 51% were
spread shifts with empty prev_state/new_state fields — noise. Each
firing triggers an adaptive weight reset at blend_alpha=0.50, thrashing
learned weights and correlating with degraded 3-12h lead MAE.
"""

from __future__ import annotations

from types import SimpleNamespace


def _make_scheduler(*, shifts_already: int = 0):
    """Build a minimal stand-in that owns just enough state to exercise
    ``AgentScheduler.detect_regime_shift`` without a real scheduler,
    storage backend, or scheduler-level side effects.
    """
    from al3x.scheduler import AgentScheduler

    obj = object.__new__(AgentScheduler)
    obj.storage = SimpleNamespace(
        count_regime_shifts_for_date=lambda _d: shifts_already,
    )
    obj._last_regime = None
    obj._last_spread = None
    return obj


def test_spread_change_below_threshold_no_shift() -> None:
    """Spread delta 4.0°F — under the new >5.0°F threshold → no shift."""
    s = _make_scheduler()
    s._last_regime = {"sea_breeze_shift": False, "any_precip_peak": False,
                      "sustained_windy": False}
    s._last_spread = 2.0
    current_regime = {"sea_breeze_shift": False, "any_precip_peak": False,
                      "sustained_windy": False}
    flags = s.detect_regime_shift(current_regime, current_spread=6.0,
                                   target_date="2026-04-22")
    assert "spread" not in flags
    assert flags == []


def test_spread_change_above_threshold_still_skipped_when_states_empty() -> None:
    """Spread delta 6.0°F exceeds the >5.0°F threshold, but the storage-
    side prev_state/new_state for a ``spread`` shift are always empty
    (``spread`` is not a regime key), so the empty-state guard must
    suppress it. This was the dominant noise pattern in the live data.
    """
    s = _make_scheduler()
    s._last_regime = {"sea_breeze_shift": False, "any_precip_peak": False,
                      "sustained_windy": False}
    s._last_spread = 2.0
    current_regime = {"sea_breeze_shift": False, "any_precip_peak": False,
                      "sustained_windy": False}
    flags = s.detect_regime_shift(current_regime, current_spread=8.0,
                                   target_date="2026-04-22")
    assert "spread" not in flags


def test_rate_limit_skips_detection_when_three_already_recorded() -> None:
    """If 3 regime shifts are already on the books for the target date,
    skip detection entirely and return []."""
    s = _make_scheduler(shifts_already=3)
    s._last_regime = {"sea_breeze_shift": False, "any_precip_peak": False,
                      "sustained_windy": False}
    s._last_spread = 2.0
    # Provide an obvious regime flip that would normally fire.
    current_regime = {"sea_breeze_shift": True, "any_precip_peak": True,
                      "sustained_windy": True}
    flags = s.detect_regime_shift(current_regime, current_spread=20.0,
                                   target_date="2026-04-22")
    assert flags == []
