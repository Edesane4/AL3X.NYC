"""Unit tests for al3x.health_summary.build_health_summary.

These tests construct synthetic /api/state-shape payloads and assert
that the derived plain-English health dict matches the documented
contract. No DB, no network, no app startup — pure-function tests.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from al3x import config as cfg
from al3x.health_summary import (
    SOURCE_ORDER,
    build_health_summary,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _now_et() -> datetime:
    return datetime.now(cfg.EASTERN)


def _all_sources_green(base_value: float = 80.0) -> Dict[str, Dict[str, Any]]:
    """Build a sources dict where every source in SOURCE_ORDER reports a value."""
    return {
        name: {"value": base_value + i * 0.1, "weight_pct": 12.5, "error": ""}
        for i, name in enumerate(SOURCE_ORDER)
    }


def _all_corrections_suppressed(
    suppressed_deltas: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build a Session-2-style corrections block: 5 corrections, all suppressed."""
    deltas = suppressed_deltas or {}
    blocks = {}
    for name in ("sea_breeze", "uhi", "cloud_timing", "precip", "inversion"):
        blocks[name] = {
            "delta": 0.0,
            "suppressed": True,
            "suppressed_delta": deltas.get(name, 0.0),
            "reason": f"{name} suppressed in Session 2",
        }
    return blocks


def _make_forecast(
    target_date: str,
    mode: str = "intraday",
    final_f: float = 80.0,
    sigma_f: float = 1.5,
    sources: Optional[Dict[str, Any]] = None,
    corrections: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
    issued_at: Optional[str] = None,
    revision: int = 1,
    fc_id: int = 1,
) -> Dict[str, Any]:
    """Forecast row in the same shape `storage._forecast_row` returns."""
    if issued_at is None:
        issued_at = _now_et().isoformat()
    return {
        "id": fc_id,
        "issued_at": issued_at,
        "target_date": target_date,
        "mode": mode,
        "revision": revision,
        "final_f": final_f,
        "sigma_f": sigma_f,
        "sources": sources or {},
        "corrections": corrections or {},
        "extras": extras or {},
    }


def _make_state(
    forecast: Optional[Dict[str, Any]] = None,
    *,
    recent_forecasts: Optional[List[Dict[str, Any]]] = None,
    scores: Optional[List[Dict[str, Any]]] = None,
    stats: Optional[Dict[str, Any]] = None,
    qrf_calibration: Optional[Dict[str, Any]] = None,
    logs: Optional[List[Any]] = None,
    regime_shifts: Optional[List[Any]] = None,
    truth_today: Optional[Dict[str, Any]] = None,
    latest_tomorrow: Optional[Dict[str, Any]] = None,
    running_max_f: Optional[float] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Wrap a forecast into a full /api/state-shape payload.

    The forecast becomes both ``latest_today`` and ``anchor`` (the
    health summary reads source/regime/corrections data from anchor).
    """
    now_iso = _now_et().isoformat()
    state: Dict[str, Any] = {
        "now": now_iso,
        "now_et": now_iso,
        "anchor": forecast,
        "latest_today": forecast,
        "latest_tomorrow": latest_tomorrow,
        "recent_forecasts": recent_forecasts if recent_forecasts is not None else (
            [forecast] if forecast else []
        ),
        "scores": scores or [],
        "stats": stats or {},
        "qrf_calibration": qrf_calibration or {},
        "logs": logs or [],
        "regime_shifts": regime_shifts or [],
        "truth_today": truth_today,
        "running_max_f": running_max_f,
    }
    state.update(overrides)
    return state


def _build_kalman_history(
    n_days: int,
    *,
    final_f: float = 80.0,
    error_f: float = 0.6,
    kalman_proj: float = 80.0,
    mode: str = "intraday",
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build matched recent_forecasts + scores so Kalman MAE reconstructs.

    For each of the past `n_days`, produces:
      - a forecast with extras.kalman.projected_max_f = kalman_proj
      - a score row with error_f so reconstructed CLI = final_f + error_f

    Per-date |kalman_proj - cli| therefore equals |kalman_proj - (final_f + error_f)|.
    """
    today = date.today()
    fcs: List[Dict[str, Any]] = []
    scs: List[Dict[str, Any]] = []
    for i in range(1, n_days + 1):
        td = (today - timedelta(days=i)).isoformat()
        issued = (_now_et() - timedelta(days=i, hours=2)).isoformat()
        fcs.append(_make_forecast(
            target_date=td,
            mode=mode,
            final_f=final_f,
            sigma_f=1.2,
            issued_at=issued,
            fc_id=1000 + i,
            extras={"kalman": {"projected_max_f": kalman_proj}},
        ))
        scs.append({
            "target_date": td,
            "mode": mode,
            "error_f": error_f,
        })
    return fcs, scs


# --------------------------------------------------------------------------- #
# Test 1 — green/yellow Session 2 state                                       #
# --------------------------------------------------------------------------- #

def test_green_state_with_session2_suppression():
    suppressed_deltas = {
        "sea_breeze":   0.0,
        "uhi":          1.2,
        "cloud_timing": -0.8,
        "precip":       0.0,
        "inversion":    0.0,
    }
    sources = _all_sources_green(base_value=68.0)
    corrections = _all_corrections_suppressed(suppressed_deltas)
    extras = {
        "bma": {"bma_forecast_f": 73.76, "bma_variance_f": 1.4},
        "gfs_ensemble": {"n_members": 1, "sigma_f": None},
        "kalman": {"hours_of_data": 12, "blend_weight": 0.4},
    }
    fc = _make_forecast(
        target_date=date.today().isoformat(),
        final_f=67.8,
        sources=sources,
        corrections=corrections,
        extras=extras,
    )

    history_fc, history_sc = _build_kalman_history(
        n_days=9, final_f=80.0, error_f=0.6, kalman_proj=80.0,
    )
    state = _make_state(
        fc,
        recent_forecasts=[fc] + history_fc,
        scores=history_sc,
    )

    h = build_health_summary(state)

    assert h["status"] == "yellow", f"expected yellow, got {h['status']}: {h['headline']}"

    gefs = next(s for s in h["sources"] if s["name"] == "gfs_ensemble")
    assert gefs["status"] == "yellow"
    assert "1 ensemble member" in gefs["status_reason"] or "1 member" in gefs["status_reason"]

    assert abs(h["what_if"]["bma"]["delta_vs_final_f"] - 5.96) < 0.01

    corr = h["what_if"]["corrections"]
    assert corr["enabled"] is False
    expected_total = sum(
        v for k, v in suppressed_deltas.items()
        if corrections[k]["suppressed"]
    )
    assert abs(corr["total_shadow_delta_f"] - expected_total) < 0.01

    narrative = h["forecast_narrative"]
    assert "67.8" in narrative
    assert ("73.76" in narrative or "73.8" in narrative or "BMA" in narrative)

    issue_ids = {i["id"] for i in h["open_issues"]}
    assert "gefs_one_member" in issue_ids


# --------------------------------------------------------------------------- #
# Test 2 — persistence warning forces red                                     #
# --------------------------------------------------------------------------- #

def test_persistence_warning_makes_status_red():
    fc = _make_forecast(
        target_date=date.today().isoformat(),
        final_f=80.0,
        sources=_all_sources_green(),
        extras={"persistence_warning": True},
    )
    state = _make_state(fc)
    h = build_health_summary(state)

    assert h["status"] == "red"
    assert "Persistence" in h["headline"]
    assert "persistence_warning" in {i["id"] for i in h["open_issues"]}


# --------------------------------------------------------------------------- #
# Test 3 — GEFS healthy when many members                                     #
# --------------------------------------------------------------------------- #

def test_gefs_healthy_when_many_members():
    fc = _make_forecast(
        target_date=date.today().isoformat(),
        final_f=80.0,
        sources=_all_sources_green(),
        extras={"gfs_ensemble": {"n_members": 21, "sigma_f": 1.4}},
    )
    state = _make_state(fc)
    h = build_health_summary(state)

    gefs = next(s for s in h["sources"] if s["name"] == "gfs_ensemble")
    assert gefs["status"] == "green"
    assert "gefs_one_member" not in {i["id"] for i in h["open_issues"]}


# --------------------------------------------------------------------------- #
# Test 4 — countdowns                                                         #
# --------------------------------------------------------------------------- #

def test_countdowns_compute_sensibly():
    fc = _make_forecast(
        target_date=date.today().isoformat(),
        final_f=80.0,
        sources=_all_sources_green(),
    )
    state = _make_state(fc)
    h = build_health_summary(state)
    cd = h["countdowns"]

    assert 0 < cd["next_intraday_in_sec"] <= 900
    assert cd["next_morning_in_sec"] > 0
    assert cd["next_cli_in_sec"] > 0
    assert isinstance(cd["cli_window_open"], bool)
    # Should be parseable
    datetime.fromisoformat(cd["next_intraday_iso"])


# --------------------------------------------------------------------------- #
# Test 5 — bet scorecard verdict bands                                        #
# --------------------------------------------------------------------------- #

def test_bet_scorecard_verdict_bands():
    # Kalman MAE ~ 0.6 by construction
    history_fc, history_sc = _build_kalman_history(
        n_days=5, final_f=80.0, error_f=0.6, kalman_proj=80.0,
    )

    fc = _make_forecast(
        target_date=date.today().isoformat(),
        final_f=80.0,
        sources=_all_sources_green(),
    )

    cases = [
        (0.8, "paying off"),  # gap = 0.2
        (2.3, "mixed"),        # gap = 1.7
        (4.5, "not yet"),      # gap = 3.9
    ]

    for final_mae, expected_prefix in cases:
        state = _make_state(
            fc,
            recent_forecasts=[fc] + history_fc,
            scores=history_sc,
            stats={
                "intraday_mae": final_mae,
                "samples": {"intraday": 10, "night_before": 8},
            },
        )
        h = build_health_summary(state)
        assert h["bet_scorecard"]["kalman_mae_f"] is not None
        assert abs(h["bet_scorecard"]["kalman_mae_f"] - 0.6) < 0.01
        verdict = h["bet_scorecard"]["verdict"]
        assert verdict.startswith(expected_prefix), (
            f"final_mae={final_mae}: expected verdict starting with "
            f"{expected_prefix!r}, got {verdict!r}"
        )


# --------------------------------------------------------------------------- #
# Test 6 — empty state                                                        #
# --------------------------------------------------------------------------- #

def test_empty_state_does_not_crash():
    h = build_health_summary({})
    assert isinstance(h, dict)
    assert h["status"] == "red"
    assert "No forecast" in h["headline"]
    for key in (
        "status", "headline", "forecast_narrative", "sources", "what_if",
        "regime_narrative", "kalman", "vitals", "cold_start", "countdowns",
        "open_issues", "alerts", "bet_scorecard",
    ):
        assert key in h, f"missing top-level key: {key}"
