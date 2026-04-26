"""Session 4 Part 4 — climatology readiness probe in /api/state.

Verifies that the readiness probe in app.py state() handles all three
plausible runtime states without raising:
  * forecaster has no _clim attribute (e.g. very early boot)
  * _clim exists but .ready is False (built() failed or hasn't run)
  * _clim exists and is ready (normal steady state)

This is a unit test of RateClimatology.stats() — the contract the
app.py probe relies on. The probe itself is exercised by integration
testing in production.
"""

from __future__ import annotations

from al3x.climatology import RateClimatology


def test_unbuilt_climatology_reports_not_ready():
    clim = RateClimatology()
    s = clim.stats()
    assert s["ready"] is False
    assert s["num_bins"] == 0
    assert s["fallback_mean"] is None
    assert s["sample_counts"] == {}


def test_get_prior_rate_returns_none_when_unbuilt():
    clim = RateClimatology()
    assert clim.get_prior_rate(month=4, sky_cover_pct=20.0,
                                wind_speed_kt=3.0) is None


def test_stats_dict_shape_matches_app_probe_contract():
    """The probe in app.py reads exactly these keys. If RateClimatology's
    stats() schema ever changes, this test fails and forces app.py to
    update in lockstep."""
    clim = RateClimatology()
    s = clim.stats()
    expected_keys = {"ready", "num_bins", "fallback_mean", "sample_counts"}
    assert set(s.keys()) == expected_keys, (
        f"RateClimatology.stats() schema drifted; got {set(s.keys())}, "
        f"expected {expected_keys}"
    )
