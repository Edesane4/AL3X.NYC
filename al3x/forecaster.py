"""Ensemble forecaster + NYC local bias corrections.

Produces both Night-Before and Intraday Revision forecasts. The output is
a dict suitable for persistence and Telegram formatting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfg
from .data_sources import DataSources, SourceResult, running_max

log = logging.getLogger("al3x.forecaster")


@dataclass
class BiasLive:
    """Live (possibly auto-tuned) bias values. Falls back to defaults."""

    values: Dict[str, float]

    def get(self, key: str, default: float) -> float:
        return float(self.values.get(key, default))


@dataclass
class SourceView:
    name: str
    value: Optional[float]
    weight: float
    error: Optional[str] = None


def _weights_for_mode(mode: str, lead_hours: float,
                      override: Optional[Dict[str, float]] = None
                      ) -> Dict[str, float]:
    if override:
        return override
    if mode == "night_before":
        return dict(cfg.NIGHT_BEFORE_WEIGHTS)
    # intraday
    if lead_hours <= 6:
        return dict(cfg.INTRADAY_WEIGHTS_0_6)
    if lead_hours <= 12:
        return dict(cfg.INTRADAY_WEIGHTS_6_12)
    return dict(cfg.INTRADAY_WEIGHTS_12_24)


def _weighted_mean(source_values: Dict[str, float],
                   weights: Dict[str, float]) -> Tuple[float, Dict[str, SourceView]]:
    live_weights = {k: v for k, v in weights.items() if k in source_values}
    total = sum(live_weights.values()) or 1.0
    out: Dict[str, SourceView] = {}
    wsum = 0.0
    for k, w in live_weights.items():
        w_norm = w / total
        out[k] = SourceView(k, source_values[k], w_norm)
        wsum += source_values[k] * w_norm
    return wsum, out


def _asos_trend_projection(obs_today: List[Dict[str, Any]],
                            nws_hourly: List[Dict[str, Any]],
                            target_date: date) -> Optional[float]:
    """Estimate daily max by combining running observed max with remaining
    forecast hourly for today. Used as the 'asos_trend' pseudo-source.
    """
    running, _ = running_max(obs_today)
    remaining: List[float] = []
    if nws_hourly:
        now = datetime.now(cfg.EASTERN)
        for h in nws_hourly:
            try:
                ts = datetime.fromisoformat(h["time"])
            except Exception:
                continue
            if ts.date() != target_date:
                continue
            if ts <= now:
                continue
            remaining.append(float(h["temp_f"]))
    if running is None and not remaining:
        return None
    return max([v for v in [running, *remaining] if v is not None])


# ---- Regime detection for bias corrections ---------------------------------

def _detect_regime(hourly: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract regime flags used by the bias-correction rules."""
    sea_breeze_shift = False
    sea_breeze_full = False
    wind_nw_all_day = True
    cloud_morning_increase = False
    cloud_afternoon_clearing = False
    any_precip_peak = False
    precip_heavy = False
    inversion_hint = False
    calm_clear = False
    sustained_windy = False

    if not hourly:
        return {"sea_breeze_shift": False, "sea_breeze_full": False,
                "wind_nw_all_day": False, "cloud_morning_increase": False,
                "cloud_afternoon_clearing": False, "any_precip_peak": False,
                "precip_heavy": False, "inversion_hint": False,
                "calm_clear": False, "sustained_windy": False,
                "cloud_avg": None, "max_wind_kt": None}

    cloud_before_noon = []
    cloud_after_noon = []
    peak_cloud_morning = 0.0
    precip_probs_peak = []
    wind_speeds = []
    dir_during_afternoon = []

    def parse_wind(s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        import re
        m = re.search(r"(\d+)", s)
        return float(m.group(1)) if m else None

    for h in hourly:
        ts = datetime.fromisoformat(h["time"])
        hour = ts.hour
        wind_kt = parse_wind(h.get("wind"))
        if wind_kt is not None:
            wind_speeds.append(wind_kt)
        wind_dir = (h.get("wind_dir") or "").upper()
        sky = h.get("sky_cover")
        precip_prob = h.get("precip_prob") or 0

        if 11 <= hour <= 15:
            dir_during_afternoon.append(wind_dir)
            if wind_dir and not any(d in wind_dir for d in ("NW", "W", "N ")):
                wind_nw_all_day = False

        if sky is not None:
            if hour < 12:
                cloud_before_noon.append(sky)
                peak_cloud_morning = max(peak_cloud_morning, sky)
            else:
                cloud_after_noon.append(sky)

        if 10 <= hour <= 15:
            precip_probs_peak.append(precip_prob)
            if precip_prob >= 70 or "thunder" in (h.get("short", "") or "").lower():
                precip_heavy = True

    if any(("S" in d and "SW" not in d) or "SE" in d for d in dir_during_afternoon):
        sea_breeze_shift = True
        # Full sea breeze: sustained S/SE at >=10kt afternoon
        afternoon_winds = [w for h, w in zip(hourly, wind_speeds[-len(hourly):])
                           if 11 <= datetime.fromisoformat(h["time"]).hour <= 16]
        if afternoon_winds and max(afternoon_winds, default=0) >= 10:
            sea_breeze_full = True

    cloud_avg_morn = (sum(cloud_before_noon) / len(cloud_before_noon)
                      if cloud_before_noon else None)
    cloud_avg_aft = (sum(cloud_after_noon) / len(cloud_after_noon)
                     if cloud_after_noon else None)

    if peak_cloud_morning >= 50 and (cloud_avg_aft or 0) >= 30:
        cloud_morning_increase = True
    if (cloud_avg_morn or 0) >= 60 and (cloud_avg_aft or 100) <= 30:
        cloud_afternoon_clearing = True

    any_precip_peak = any(p >= 40 for p in precip_probs_peak)

    max_wind = max(wind_speeds) if wind_speeds else 0
    sustained_windy = max_wind >= 15
    calm_clear = (max_wind < 10 and (cloud_avg_aft or 100) <= 30
                  and (cloud_avg_morn or 100) <= 50)

    cloud_avg = None
    all_cloud = cloud_before_noon + cloud_after_noon
    if all_cloud:
        cloud_avg = sum(all_cloud) / len(all_cloud)

    return {
        "sea_breeze_shift": sea_breeze_shift,
        "sea_breeze_full": sea_breeze_full,
        "wind_nw_all_day": wind_nw_all_day and bool(dir_during_afternoon),
        "cloud_morning_increase": cloud_morning_increase,
        "cloud_afternoon_clearing": cloud_afternoon_clearing,
        "any_precip_peak": any_precip_peak,
        "precip_heavy": precip_heavy,
        "inversion_hint": inversion_hint,
        "calm_clear": calm_clear,
        "sustained_windy": sustained_windy,
        "cloud_avg": cloud_avg,
        "max_wind_kt": max_wind,
    }


def _apply_corrections(target_date: date, regime: Dict[str, Any],
                       spread: float, bias: BiasLive
                       ) -> Tuple[float, Dict[str, Dict[str, Any]]]:
    """Return (total_delta, per-correction detail)."""
    corrections: Dict[str, Dict[str, Any]] = {}
    month = target_date.month

    # 1. Sea breeze (May–Sep)
    sb_delta = 0.0
    sb_reason = "outside sea-breeze season"
    if 5 <= month <= 9:
        if regime["sea_breeze_full"]:
            sb_delta = bias.get("sea_breeze_full", cfg.BIAS_DEFAULTS.sea_breeze_full)
            sb_reason = "Full sea-breeze penetration expected (S/SE ≥10 kt)"
        elif regime["sea_breeze_shift"]:
            sb_delta = bias.get("sea_breeze_shift",
                                cfg.BIAS_DEFAULTS.sea_breeze_shift)
            sb_reason = "Afternoon wind shift to S/SE expected"
        elif regime["wind_nw_all_day"]:
            sb_delta = bias.get("sea_breeze_nw_boost",
                                cfg.BIAS_DEFAULTS.sea_breeze_nw_boost)
            sb_reason = "W/NW wind all day — no sea-breeze suppression"
        else:
            sb_reason = "No clear sea-breeze signal"
    corrections["sea_breeze"] = {"delta": sb_delta, "reason": sb_reason}

    # 2. UHI
    if regime["calm_clear"]:
        uhi = bias.get("uhi_clear_calm", cfg.BIAS_DEFAULTS.uhi_clear_calm)
        uhi_reason = "Clear, calm day — urban heat boost"
    elif regime["sustained_windy"]:
        uhi = 0.0
        uhi_reason = "Windy — atmosphere well mixed"
    else:
        uhi = 0.0
        uhi_reason = "Cloudy/moist — no UHI boost"
    corrections["uhi"] = {"delta": uhi, "reason": uhi_reason}

    # 3. Cloud timing
    if regime["cloud_morning_increase"]:
        cd = bias.get("cloud_increase_morning",
                      cfg.BIAS_DEFAULTS.cloud_increase_morning)
        cd_reason = "Cloud cover increases before 1 PM"
    elif regime["cloud_afternoon_clearing"]:
        cd = bias.get("cloud_clearing_afternoon",
                      cfg.BIAS_DEFAULTS.cloud_clearing_afternoon)
        cd_reason = "Morning overcast clears after noon"
    else:
        cd = 0.0
        cd_reason = "Cloud timing neutral"
    corrections["cloud_timing"] = {"delta": cd, "reason": cd_reason}

    # 4. Precipitation
    if regime["precip_heavy"]:
        pd_ = bias.get("precip_heavy", cfg.BIAS_DEFAULTS.precip_heavy)
        pd_reason = "Heavy / convective precip expected in peak window"
    elif regime["any_precip_peak"]:
        pd_ = bias.get("precip_light", cfg.BIAS_DEFAULTS.precip_light)
        pd_reason = "Precip likely during 10 AM – 3 PM peak heating"
    else:
        pd_ = 0.0
        pd_reason = "No precip in peak window"
    corrections["precip"] = {"delta": pd_, "reason": pd_reason}

    # 5. Inversion / fog (Oct–Apr)
    if month >= 10 or month <= 4:
        inv = bias.get("inversion_winter",
                       cfg.BIAS_DEFAULTS.inversion_winter) \
            if regime.get("inversion_hint") else 0.0
        inv_reason = ("Morning inversion / fog cap" if inv != 0
                      else "No inversion cue detected")
    else:
        inv = 0.0
        inv_reason = "Outside inversion season"
    corrections["inversion"] = {"delta": inv, "reason": inv_reason}

    # 6. Spread penalty — zero delta but drives uncertainty elsewhere
    if spread >= 6:
        corrections["spread_penalty"] = {
            "delta": 0.0,
            "reason": f"High model spread ({spread:.1f}°F) — "
                      "no additive trust; widened uncertainty",
        }
        # When spread is high, zero out other corrections (less reliable)
        for k in ("sea_breeze", "uhi", "cloud_timing", "precip", "inversion"):
            corrections[k]["delta"] = 0.0
            corrections[k]["reason"] += " (suppressed: high spread)"
    else:
        corrections["spread_penalty"] = {
            "delta": 0.0,
            "reason": f"Model spread {spread:.1f}°F — corrections trusted",
        }

    total = sum(c["delta"] for c in corrections.values())
    return total, corrections


# ---- Main entrypoint -------------------------------------------------------

class Forecaster:
    def __init__(self, storage, sources: DataSources) -> None:
        self.storage = storage
        self.sources = sources

    def _live_bias(self) -> BiasLive:
        values = self.storage.get_biases()
        return BiasLive(values=values)

    def _live_weights(self, mode: str, lead_hours: float) -> Dict[str, float]:
        base = _weights_for_mode(mode, lead_hours)
        stored = self.storage.get_weights()
        # stored keys like "night_before:hrrr" or "intraday:hrrr"
        prefix = "night_before" if mode == "night_before" else "intraday"
        for k in list(base.keys()):
            sk = f"{prefix}:{k}"
            if sk in stored:
                base[k] = stored[sk]
        # re-normalize
        total = sum(base.values()) or 1.0
        return {k: v / total for k, v in base.items()}

    async def produce(self, mode: str, target_date: date) -> Dict[str, Any]:
        now = datetime.now(cfg.EASTERN)
        lead_hours = max(
            (datetime.combine(target_date,
                              datetime.strptime("15:00", "%H:%M").time(),
                              tzinfo=cfg.EASTERN) - now).total_seconds() / 3600,
            0.0,
        )

        # Fetch everything in parallel-ish (awaited sequentially but cheap)
        import asyncio
        hourly_task = asyncio.create_task(self.sources.nws_hourly_max(target_date))
        nbm_task = asyncio.create_task(self.sources.nws_daily_max(target_date))
        hrrr_task = asyncio.create_task(self.sources.hrrr(target_date))
        ecmwf_task = asyncio.create_task(self.sources.ecmwf(target_date))
        gfs_task = asyncio.create_task(self.sources.gfs_mos(target_date))
        nam_task = asyncio.create_task(self.sources.nam_mos(target_date))
        asos_task = asyncio.create_task(self.sources.asos_observations())

        nws_r = await hourly_task
        nbm_r = await nbm_task
        hrrr_r = await hrrr_task
        ecmwf_r = await ecmwf_task
        gfs_r = await gfs_task
        nam_r = await nam_task
        obs = await asos_task

        # Persist observations we haven't seen
        today_str = target_date.isoformat() if mode == "intraday" else None
        if mode == "intraday" and obs:
            seen = {o["observed_at"]
                    for o in self.storage.observations_today(today_str)}
            for o in obs:
                if o["observed_at"][:10] == today_str \
                        and o["observed_at"] not in seen:
                    self.storage.save_observation(o)

        obs_today = self.storage.observations_today(today_str) if today_str else []

        running_max_f, _ = running_max(obs_today)

        # Assemble raw values
        raw_values: Dict[str, Optional[float]] = {
            "hrrr": hrrr_r.value,
            "nws_point": nws_r.value,
            "gfs_mos": gfs_r.value,
            "nam_mos": nam_r.value,
            "ecmwf": ecmwf_r.value,
            "nbm": nbm_r.value,
        }
        if mode == "intraday":
            raw_values["asos_trend"] = _asos_trend_projection(
                obs_today, nws_r.meta.get("hourly", []), target_date,
            )

        source_values = {k: v for k, v in raw_values.items() if v is not None}

        weights = self._live_weights(mode, lead_hours)
        raw_ensemble, views = _weighted_mean(source_values, weights)

        # Spread calc from contributing (weighted, >0) sources
        contrib_vals = [v.value for v in views.values()
                        if v.value is not None and v.weight > 0]
        spread = (max(contrib_vals) - min(contrib_vals)) if contrib_vals else 0.0

        # Corrections
        bias = self._live_bias()
        regime = _detect_regime(nws_r.meta.get("hourly", []))
        total_delta, corrections = _apply_corrections(
            target_date, regime, spread, bias
        )

        final = raw_ensemble + total_delta

        # Running-max floor for intraday
        if mode == "intraday" and running_max_f is not None:
            final = max(final, running_max_f)

        # Uncertainty
        uncertainty = 2.0
        if spread >= 6:
            uncertainty += 2.0
        elif spread <= 3:
            uncertainty = max(1.0, uncertainty - 1.0)
        if mode == "intraday" and lead_hours <= 3:
            uncertainty = max(0.5, uncertainty - 1.0)

        revision = self.storage.last_revision(target_date.isoformat(), mode) + 1 \
            if mode == "intraday" else 0

        prior = self.storage.latest_forecast(target_date.isoformat())
        delta_prior = (final - prior["final_f"]) if prior else None

        # Build sources dict for storage (include all attempted sources)
        sources_persist: Dict[str, Dict[str, Any]] = {}
        for name in ("hrrr", "nws_point", "gfs_mos", "nam_mos",
                     "ecmwf", "nbm", "asos_trend"):
            sources_persist[name] = {
                "value": raw_values.get(name),
                "weight": weights.get(name, 0.0) if name in source_values else 0.0,
                "error": {
                    "hrrr": hrrr_r.error, "nws_point": nws_r.error,
                    "gfs_mos": gfs_r.error, "nam_mos": nam_r.error,
                    "ecmwf": ecmwf_r.error, "nbm": nbm_r.error,
                }.get(name),
            }

        extras = {
            "regime": regime,
            "spread_f": spread,
            "lead_hours": lead_hours,
            "weights_used": weights,
        }

        return {
            "issued_at": now.isoformat(),
            "target_date": target_date.isoformat(),
            "mode": mode,
            "revision": revision,
            "final_f": round(final, 1),
            "raw_ensemble_f": round(raw_ensemble, 2),
            "uncertainty_f": round(uncertainty, 1),
            "running_asos_max_f": running_max_f,
            "sources": sources_persist,
            "corrections": corrections,
            "delta_prior_f": (round(delta_prior, 2)
                              if delta_prior is not None else None),
            "extras": extras,
        }
