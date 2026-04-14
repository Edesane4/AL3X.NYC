"""Learning system: scoring, bias ledger, auto-tune, correction attribution."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from . import config as cfg
from .storage import Storage

log = logging.getLogger("al3x.learning")


# Named weight sets matching cfg.*_WEIGHTS so retune_weights can round-trip
# them cleanly. The persisted key scheme is "<set>:<source>".
_WEIGHT_SETS: Dict[str, Dict[str, float]] = {
    "night_before": cfg.NIGHT_BEFORE_WEIGHTS,
    "intraday_0_6": cfg.INTRADAY_WEIGHTS_0_6,
    "intraday_6_12": cfg.INTRADAY_WEIGHTS_6_12,
    "intraday_12_24": cfg.INTRADAY_WEIGHTS_12_24,
}
# Legacy keys preserved so the Forecaster._live_weights lookup still finds
# the tuned weights under "night_before:..." / "intraday:...".
_LEGACY_MAP = {
    "night_before": "night_before",
    "intraday_0_6": "intraday",
    "intraday_6_12": "intraday",
    "intraday_12_24": "intraday",
}


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
                last_intraday = f

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

            # Superior 3 — correction attribution ledger
            self._attribute_corrections(f, recorded_high_f, regime)

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

    def _attribute_corrections(self, fc: Dict[str, Any], cli: float,
                                regime: str) -> None:
        """For each nonzero correction that was applied to this forecast,
        record whether applying it reduced |error| or increased it.

        Logic: final = raw_ensemble + sum(correction deltas) + (ai_delta).
        For a given correction delta d, the counterfactual "without it"
        final would be final - d. We compare |cli - final| vs |cli - (final - d)|.
        """
        final = fc.get("final_f")
        if final is None:
            return
        err = cli - final
        corrections = fc.get("corrections", {}) or {}
        for name, payload in corrections.items():
            d = float(payload.get("delta") or 0.0)
            if abs(d) < 1e-6:
                continue
            err_without = cli - (final - d)
            if abs(err_without) > abs(err) + 1e-6:
                helpful = 1
            elif abs(err_without) < abs(err) - 1e-6:
                helpful = -1
            else:
                helpful = 0
            self.storage.save_attribution({
                "target_date": fc["target_date"],
                "correction_name": name,
                "delta_applied_f": d,
                "regime_label": regime,
                "error_f": err,
                "was_helpful": helpful,
                "forecast_id": fc.get("id"),
            })

    # ---- Gap 2 — inverse-MAE weight retune across all 4 weight sets -----
    def retune_weights(self) -> Dict[str, Any]:
        scores = self.storage.recent_scores(days=7)
        if len(scores) < 5:
            return {"ok": False, "reason": "insufficient history"}

        by_source_abs: Dict[str, List[float]] = defaultdict(list)
        for s in scores:
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
                by_source_abs[name].append(abs(cli - v))

        if not by_source_abs:
            return {"ok": False, "reason": "no source data"}

        mae = {k: sum(v) / len(v) for k, v in by_source_abs.items() if v}
        # Guard against MAE = 0 collapse
        EPS = 0.5

        changes: Dict[str, Dict[str, float]] = {}
        for set_name, default_weights in _WEIGHT_SETS.items():
            new_weights: Dict[str, float] = {}
            inv_sum = 0.0
            for src in default_weights:
                mae_src = mae.get(src)
                if mae_src is None:
                    continue
                inv_sum += 1.0 / max(mae_src, EPS)
            if inv_sum <= 0:
                continue

            # Inverse-MAE proportional target weights for sources in this set
            target: Dict[str, float] = {}
            for src in default_weights:
                mae_src = mae.get(src)
                if mae_src is None:
                    target[src] = default_weights[src]
                else:
                    target[src] = (1.0 / max(mae_src, EPS)) / inv_sum

            # 30/70 blend with current (stored) weights
            current_stored = self.storage.get_weights()
            legacy_prefix = _LEGACY_MAP[set_name]
            for src, default_w in default_weights.items():
                stored_key = f"{legacy_prefix}:{src}"
                cur = current_stored.get(stored_key, default_w)
                new_weights[src] = 0.7 * cur + 0.3 * target.get(src, default_w)

            # Clamp: no source > 0.50, no source < 0.05
            for src in list(new_weights):
                new_weights[src] = max(0.05, min(0.50, new_weights[src]))

            # Structural floor: HRRR >= 0.35 in intraday_0_6
            if set_name == "intraday_0_6" and "hrrr" in new_weights:
                if new_weights["hrrr"] < 0.35:
                    new_weights["hrrr"] = 0.35

            # Re-normalize to 1.0
            total = sum(new_weights.values()) or 1.0
            for src in new_weights:
                new_weights[src] = new_weights[src] / total

            # Persist (legacy key) + per-set key
            for src, w in new_weights.items():
                self.storage.set_weight(f"{legacy_prefix}:{src}", w)
                self.storage.set_weight(f"{set_name}:{src}", w)
            changes[set_name] = {k: round(v, 3) for k, v in new_weights.items()}

        return {"ok": True, "mae": {k: round(v, 2) for k, v in mae.items()},
                "changes": changes}

    # ---- Gap 3 — auto-tune ALL 5 bias corrections -----------------------
    def retune_biases(self) -> Dict[str, Any]:
        scores = self.storage.recent_scores(days=14)
        if not scores:
            return {"ok": False, "reason": "no scores"}

        regime_err: Dict[str, List[float]] = defaultdict(list)
        for s in scores:
            regime_err[s.get("regime") or "other"].append(s["error_f"])

        # For cloud + inversion we need the forecast's regime flags, not the
        # simple label. Pull the underlying forecasts.
        forecasts_by_id: Dict[int, Dict[str, Any]] = {}
        for s in scores:
            fid = s["forecast_id"]
            if fid in forecasts_by_id:
                continue
            fc = _find_forecast_by_id(self.storage, fid)
            if fc:
                forecasts_by_id[fid] = fc

        flag_err: Dict[str, List[float]] = defaultdict(list)
        for s in scores:
            fc = forecasts_by_id.get(s["forecast_id"])
            if not fc:
                continue
            flags = (fc.get("extras", {}) or {}).get("regime", {}) or {}
            for fname in ("cloud_morning_increase",
                          "cloud_afternoon_clearing",
                          "inversion_hint"):
                if flags.get(fname):
                    flag_err[fname].append(s["error_f"])

        adjustments: Dict[str, Any] = {}
        current = self.storage.get_biases()

        def _bump(key: str, delta: float, reason: str, default: float) -> None:
            cur = current.get(key, default)
            new = max(min(cur + delta, default + 2.0), default - 2.0)
            if abs(new - cur) < 1e-6:
                return
            self.storage.set_bias(key, new, reason)
            adjustments[key] = {"old": cur, "new": new, "reason": reason}

        # Superior 3 — attribution-driven "harmful correction shrink" (first)
        attr_stats = self.attribution_stats(days=30)
        harmful_keys: Dict[str, Dict[str, Any]] = {}
        for cname, st in attr_stats.items():
            if st["times_applied"] < 4:
                continue
            if st["times_applied"] == 0:
                continue
            ratio = st["times_harmful"] / st["times_applied"]
            if ratio > 0.55:
                harmful_keys[cname] = st

        def _shrink(bias_key: str, default: float, reason: str) -> None:
            cur = current.get(bias_key, default)
            # Shrink magnitude toward 0 by 1.0°F (bounded by zero)
            if cur > 0:
                new = max(0.0, cur - 1.0)
            elif cur < 0:
                new = min(0.0, cur + 1.0)
            else:
                return
            new = max(min(new, default + 2.0), default - 2.0)
            if abs(new - cur) < 1e-6:
                return
            self.storage.set_bias(bias_key, new, reason)
            adjustments[bias_key] = {"old": cur, "new": new,
                                      "reason": reason, "event": "harmful_shrink"}
            log.warning("Harmful correction shrink: %s %.1f → %.1f (%s)",
                        bias_key, cur, new, reason)

        _CORR_TO_BIAS_KEYS = {
            "sea_breeze": ("sea_breeze_shift",
                           cfg.BIAS_DEFAULTS.sea_breeze_shift),
            "uhi": ("uhi_clear_calm", cfg.BIAS_DEFAULTS.uhi_clear_calm),
            "cloud_timing": ("cloud_increase_morning",
                             cfg.BIAS_DEFAULTS.cloud_increase_morning),
            "precip": ("precip_light", cfg.BIAS_DEFAULTS.precip_light),
            "inversion": ("inversion_winter",
                          cfg.BIAS_DEFAULTS.inversion_winter),
        }
        for cname, st in harmful_keys.items():
            mapping = _CORR_TO_BIAS_KEYS.get(cname)
            if not mapping:
                continue
            bkey, default = mapping
            reason = (f"harmful {st['times_harmful']}/{st['times_applied']} "
                      f"(>{55}%) — shrinking toward zero")
            _shrink(bkey, default, reason)

        # Sea breeze
        sb_errs = regime_err.get("sea_breeze", [])
        if len(sb_errs) >= 3 and "sea_breeze_shift" not in adjustments:
            me = sum(sb_errs) / len(sb_errs)
            if me < -1.0:
                _bump("sea_breeze_shift", -0.5,
                      f"14d sea-breeze ME={me:+.1f}°F — strengthen penalty",
                      cfg.BIAS_DEFAULTS.sea_breeze_shift)
            elif me > 1.0:
                _bump("sea_breeze_shift", +0.5,
                      f"14d sea-breeze ME={me:+.1f}°F — weaken penalty",
                      cfg.BIAS_DEFAULTS.sea_breeze_shift)

        # UHI
        clr_errs = regime_err.get("clear_calm", [])
        if len(clr_errs) >= 3 and "uhi_clear_calm" not in adjustments:
            me = sum(clr_errs) / len(clr_errs)
            if me > 1.0:
                _bump("uhi_clear_calm", +0.5,
                      f"14d clear-calm ME={me:+.1f}°F — strengthen UHI boost",
                      cfg.BIAS_DEFAULTS.uhi_clear_calm)
            elif me < -1.0:
                _bump("uhi_clear_calm", -0.5,
                      f"14d clear-calm ME={me:+.1f}°F — weaken UHI boost",
                      cfg.BIAS_DEFAULTS.uhi_clear_calm)

        # Cloud increase morning
        cim_errs = flag_err.get("cloud_morning_increase", [])
        if len(cim_errs) >= 3 and "cloud_increase_morning" not in adjustments:
            me = sum(cim_errs) / len(cim_errs)
            if me > 1.0:
                _bump("cloud_increase_morning", +0.5,
                      f"14d cloud_morning ME={me:+.1f}°F — over-correcting, weaken",
                      cfg.BIAS_DEFAULTS.cloud_increase_morning)
            elif me < -1.0:
                _bump("cloud_increase_morning", -0.5,
                      f"14d cloud_morning ME={me:+.1f}°F — under-correcting, strengthen",
                      cfg.BIAS_DEFAULTS.cloud_increase_morning)

        # Cloud clearing afternoon
        cca_errs = flag_err.get("cloud_afternoon_clearing", [])
        if len(cca_errs) >= 3 and "cloud_clearing_afternoon" not in adjustments:
            me = sum(cca_errs) / len(cca_errs)
            if me > 1.0:
                _bump("cloud_clearing_afternoon", +0.5,
                      f"14d cloud_clearing ME={me:+.1f}°F — under-boosting, strengthen",
                      cfg.BIAS_DEFAULTS.cloud_clearing_afternoon)
            elif me < -1.0:
                _bump("cloud_clearing_afternoon", -0.5,
                      f"14d cloud_clearing ME={me:+.1f}°F — over-boosting, weaken",
                      cfg.BIAS_DEFAULTS.cloud_clearing_afternoon)

        # Precip light
        rl_errs = regime_err.get("rain_light", [])
        if len(rl_errs) >= 3 and "precip_light" not in adjustments:
            me = sum(rl_errs) / len(rl_errs)
            if me < -1.0:
                _bump("precip_light", -0.5,
                      f"14d rain_light ME={me:+.1f}°F — strengthen",
                      cfg.BIAS_DEFAULTS.precip_light)
            elif me > 1.0:
                _bump("precip_light", +0.5,
                      f"14d rain_light ME={me:+.1f}°F — weaken",
                      cfg.BIAS_DEFAULTS.precip_light)

        # Precip heavy
        rh_errs = regime_err.get("rain_heavy", [])
        if len(rh_errs) >= 3 and "precip_heavy" not in adjustments:
            me = sum(rh_errs) / len(rh_errs)
            if me < -1.0:
                _bump("precip_heavy", -0.5,
                      f"14d rain_heavy ME={me:+.1f}°F — strengthen",
                      cfg.BIAS_DEFAULTS.precip_heavy)
            elif me > 1.0:
                _bump("precip_heavy", +0.5,
                      f"14d rain_heavy ME={me:+.1f}°F — weaken",
                      cfg.BIAS_DEFAULTS.precip_heavy)

        # Inversion winter (Oct–Apr only)
        inv_errs = flag_err.get("inversion_hint", [])
        month = datetime.now(cfg.EASTERN).month
        if len(inv_errs) >= 3 and (month >= 10 or month <= 4) \
                and "inversion_winter" not in adjustments:
            me = sum(inv_errs) / len(inv_errs)
            if me < -1.0:
                _bump("inversion_winter", -0.5,
                      f"14d inversion ME={me:+.1f}°F — strengthen cap",
                      cfg.BIAS_DEFAULTS.inversion_winter)
            elif me > 1.0:
                _bump("inversion_winter", +0.5,
                      f"14d inversion ME={me:+.1f}°F — weaken cap",
                      cfg.BIAS_DEFAULTS.inversion_winter)

        return {"ok": True, "adjustments": adjustments,
                "regime_sample_sizes": {k: len(v)
                                         for k, v in regime_err.items()},
                "flag_sample_sizes": {k: len(v)
                                       for k, v in flag_err.items()},
                "harmful_shrinks": list(harmful_keys.keys())}

    # ---- Superior 3 — correction attribution aggregation ----------------
    def attribution_stats(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        rows = self.storage.attribution_rows(days=days)
        agg: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "times_applied": 0, "times_helpful": 0, "times_harmful": 0,
            "sum_delta": 0.0, "sum_error": 0.0,
        })
        for r in rows:
            name = r["correction_name"]
            a = agg[name]
            a["times_applied"] += 1
            a["sum_delta"] += r["delta_applied_f"]
            a["sum_error"] += r["error_f"]
            if r["was_helpful"] == 1:
                a["times_helpful"] += 1
            elif r["was_helpful"] == -1:
                a["times_harmful"] += 1

        out: Dict[str, Dict[str, Any]] = {}
        for name, a in agg.items():
            n = a["times_applied"]
            out[name] = {
                "times_applied": n,
                "times_helpful": a["times_helpful"],
                "times_harmful": a["times_harmful"],
                "mean_delta_applied": round(a["sum_delta"] / n, 2) if n else 0.0,
                "mean_error_when_active": round(a["sum_error"] / n, 2) if n else 0.0,
                "harmful_ratio": (round(a["times_harmful"] / n, 3)
                                   if n else 0.0),
            }
        return out

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
