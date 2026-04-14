"""Learning system: scoring, bias ledger, and auto-tune of weights + biases."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from . import config as cfg
from .storage import Storage

log = logging.getLogger("al3x.learning")


class Learning:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    # ---- Step 1 — Score every forecast ----------------------------------
    def score_day(self, target_date: str, recorded_high_f: float) -> Dict[str, Any]:
        forecasts = self.storage.forecasts_for_date(target_date)
        if not forecasts:
            return {"target_date": target_date, "scored": 0}

        night_before: Optional[Dict] = None
        last_intraday: Optional[Dict] = None
        for f in forecasts:
            if f["mode"] == "night_before" and night_before is None:
                night_before = f
            if f["mode"] == "intraday":
                last_intraday = f  # keep iterating, last wins

        scored = 0
        for f in (night_before, last_intraday):
            if not f:
                continue
            error = recorded_high_f - f["final_f"]
            issued = datetime.fromisoformat(f["issued_at"])
            target_mid = datetime.combine(
                date.fromisoformat(target_date),
                datetime.strptime("15:00", "%H:%M").time(),
                tzinfo=cfg.EASTERN,
            )
            lead_hours = max((target_mid - issued).total_seconds() / 3600, 0.0)
            regime = _regime_label(f.get("extras", {}).get("regime", {}))

            self.storage.save_score({
                "target_date": target_date,
                "forecast_id": f["id"],
                "mode": f["mode"],
                "error_f": error,
                "abs_error_f": abs(error),
                "lead_hours": lead_hours,
                "regime": regime,
            })
            scored += 1

            # Anomaly
            if abs(error) >= cfg.ANOMALY_THRESHOLD:
                rc = _root_cause(f, recorded_high_f)
                self.storage.save_anomaly({
                    "target_date": target_date,
                    "forecast_id": f["id"],
                    "error_f": error,
                    "root_cause": rc["summary"],
                    "detail": rc,
                })
                log.warning("Anomaly %+.1f°F on %s (%s): %s",
                            error, target_date, f["mode"], rc["summary"])

        return {"target_date": target_date, "scored": scored,
                "recorded_high_f": recorded_high_f}

    # ---- Step 3 — Weight auto-adjust (rolling 7 days) -------------------
    def retune_weights(self) -> Dict[str, Any]:
        scores = self.storage.recent_scores(days=7)
        if len(scores) < 5:
            return {"ok": False, "reason": "insufficient history"}

        # For each score we need each source's predicted value.
        # We load the corresponding forecast for source-level error.
        by_source: Dict[str, List[float]] = defaultdict(list)
        for s in scores:
            # find the forecast
            fc = _find_forecast_by_id(self.storage, s["forecast_id"])
            if not fc:
                continue
            truth = self.storage.get_cli_truth(s["target_date"])
            if not truth:
                continue
            cli = truth["recorded_high_f"]
            for name, payload in fc.get("sources", {}).items():
                v = payload.get("value")
                if v is None:
                    continue
                by_source[name].append(abs(cli - v))

        if not by_source:
            return {"ok": False, "reason": "no source data"}

        mae = {k: sum(v) / len(v) for k, v in by_source.items() if v}
        if "hrrr" not in mae or "gfs_mos" not in mae:
            return {"ok": False, "reason": "key sources missing"}

        changes = {}
        for mode in ("night_before", "intraday"):
            base = (dict(cfg.NIGHT_BEFORE_WEIGHTS) if mode == "night_before"
                    else dict(cfg.INTRADAY_WEIGHTS_6_12))
            # directive: if GFS-MOS MAE < HRRR MAE → bump GFS-MOS +5, HRRR -5
            if mae.get("gfs_mos", 99) < mae.get("hrrr", 99):
                base["gfs_mos"] = min(base.get("gfs_mos", 0.2) + 0.05, 0.50)
                base["hrrr"] = max(base.get("hrrr", 0.25) - 0.05, 0.05)
                changes[mode] = "gfs_mos +5, hrrr -5"
            # Save
            for k, v in base.items():
                self.storage.set_weight(f"{mode}:{k}", v)

        return {"ok": True, "mae": mae, "changes": changes}

    # ---- Step 4 — Bias auto-adjust (rolling 14 days) --------------------
    def retune_biases(self) -> Dict[str, Any]:
        scores = self.storage.recent_scores(days=14)
        if not scores:
            return {"ok": False, "reason": "no scores"}

        # Compute mean error by regime
        regime_err: Dict[str, List[float]] = defaultdict(list)
        for s in scores:
            regime_err[s.get("regime") or "other"].append(s["error_f"])

        adjustments: Dict[str, Any] = {}
        current = self.storage.get_biases()

        def _bump(key: str, delta: float, reason: str, default: float) -> None:
            cur = current.get(key, default)
            new = max(min(cur + delta, default + 2.0), default - 2.0)
            if abs(new - cur) < 1e-6:
                return
            self.storage.set_bias(key, new, reason)
            adjustments[key] = {"old": cur, "new": new, "reason": reason}

        # Sea breeze
        sb_errs = regime_err.get("sea_breeze", [])
        if len(sb_errs) >= 3:
            me = sum(sb_errs) / len(sb_errs)
            if me < -1.0:  # CLI cooler than forecast → forecast too warm, correction too small
                _bump("sea_breeze_shift", -0.5,
                      f"14d sea-breeze ME={me:+.1f}°F — strengthen penalty",
                      cfg.BIAS_DEFAULTS.sea_breeze_shift)
            elif me > 1.0:  # over-correcting (too cold)
                _bump("sea_breeze_shift", +0.5,
                      f"14d sea-breeze ME={me:+.1f}°F — weaken penalty",
                      cfg.BIAS_DEFAULTS.sea_breeze_shift)

        # UHI
        clr_errs = regime_err.get("clear_calm", [])
        if len(clr_errs) >= 3:
            me = sum(clr_errs) / len(clr_errs)
            if me > 1.0:
                _bump("uhi_clear_calm", +0.5,
                      f"14d clear-calm ME={me:+.1f}°F — strengthen UHI boost",
                      cfg.BIAS_DEFAULTS.uhi_clear_calm)
            elif me < -1.0:
                _bump("uhi_clear_calm", -0.5,
                      f"14d clear-calm ME={me:+.1f}°F — weaken UHI boost",
                      cfg.BIAS_DEFAULTS.uhi_clear_calm)

        return {"ok": True, "adjustments": adjustments,
                "regime_sample_sizes": {k: len(v) for k, v in regime_err.items()}}

    # ---- stats endpoints ------------------------------------------------
    def headline_stats(self) -> Dict[str, Any]:
        scores = self.storage.recent_scores(days=30)
        by_mode: Dict[str, List[float]] = defaultdict(list)
        for s in scores:
            by_mode[s["mode"]].append(s["abs_error_f"])
        out = {"window_days": 30,
               "night_before_mae": _mean(by_mode.get("night_before", [])),
               "intraday_mae": _mean(by_mode.get("intraday", [])),
               "samples": {k: len(v) for k, v in by_mode.items()}}
        out["target_night_before_mae"] = cfg.TARGET_NIGHT_BEFORE_MAE
        out["target_intraday_mae"] = cfg.TARGET_INTRADAY_MAE
        return out


# ---- helpers ---------------------------------------------------------------

def _mean(xs: List[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 2) if xs else None


def _regime_label(regime: Dict[str, Any]) -> str:
    if regime.get("precip_heavy"):
        return "rain_heavy"
    if regime.get("any_precip_peak"):
        return "rain_light"
    if regime.get("sea_breeze_full") or regime.get("sea_breeze_shift"):
        return "sea_breeze"
    if regime.get("calm_clear"):
        return "clear_calm"
    if regime.get("sustained_windy"):
        return "windy"
    return "mixed"


def _root_cause(fc: Dict[str, Any], cli: float) -> Dict[str, Any]:
    sources = fc.get("sources", {})
    source_errors = []
    for name, payload in sources.items():
        v = payload.get("value")
        if v is None:
            continue
        source_errors.append((name, cli - v, payload.get("weight", 0.0)))
    source_errors.sort(key=lambda x: abs(x[1]), reverse=True)
    worst = source_errors[0] if source_errors else None
    active_corrs = {k: v for k, v in fc.get("corrections", {}).items()
                    if v.get("delta", 0) != 0}
    return {
        "summary": (f"worst source {worst[0]} off {worst[1]:+.1f}°F"
                    if worst else "no source data"),
        "worst_source": worst[0] if worst else None,
        "worst_source_error_f": worst[1] if worst else None,
        "active_corrections": active_corrs,
        "regime": fc.get("extras", {}).get("regime", {}),
    }


def _find_forecast_by_id(storage: Storage, fid: int) -> Optional[Dict]:
    import sqlite3
    # lightweight direct lookup
    with storage._conn() as c:  # noqa: SLF001
        r = c.execute("SELECT * FROM forecasts WHERE id=?", (fid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        for k in ("sources_json", "corrections_json", "extras_json"):
            if d.get(k):
                try:
                    d[k.replace("_json", "")] = json.loads(d[k])
                except Exception:
                    d[k.replace("_json", "")] = {}
                d.pop(k, None)
        return d
