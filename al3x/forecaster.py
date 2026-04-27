"""Ensemble forecaster + NYC local bias corrections.

Produces both Night-Before and Intraday Revision forecasts. The output is
a dict suitable for persistence and Telegram formatting.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfg
from .climatology import RateClimatology
from .data_sources import DataSources, SourceResult, running_max
from .quantile_forest import QuantileForest, _feature_vector as _qrf_fv

log = logging.getLogger("al3x.forecaster")

# z-score for the 80% central interval (10th/90th percentiles of standard normal)
_Z_80 = 1.2816


def _compute_forecast_bands(
    final_f: float,
    gefs_sigma: Optional[float],
    contributing_sources: List[Tuple[str, float, float]],
) -> Dict[str, Any]:
    """Compute P10/P50/P90 bands for the agent's blended forecast.

    Variance decomposition combines two independent uncertainty sources:
      - GEFS member spread (initial-condition epistemic uncertainty)
      - Across-model disagreement (model-physics epistemic uncertainty)

    sigma_total = sqrt(sigma_gefs**2 + sigma_model_disagreement**2)

    contributing_sources is a list of (name, value, weight) tuples for
    sources that contributed to final_f. We exclude gfs_ensemble from
    the disagreement calculation to avoid double-counting GEFS spread.

    Returns a dict suitable for attaching to extras["bands"].
    """
    gefs_var = (gefs_sigma ** 2) if (gefs_sigma is not None and gefs_sigma > 0) else 0.0

    contributors = [
        (v, w) for (name, v, w) in contributing_sources
        if v is not None and w > 0 and name != "gfs_ensemble"
    ]
    model_disagreement_var = 0.0
    if len(contributors) >= 2:
        total_w = sum(w for _, w in contributors)
        if total_w > 0:
            wmean = sum(v * w for v, w in contributors) / total_w
            model_disagreement_var = (
                sum(w * (v - wmean) ** 2 for v, w in contributors) / total_w
            )

    sigma_total = math.sqrt(gefs_var + model_disagreement_var)
    sigma_total = max(0.5, sigma_total)

    return {
        "p10": round(final_f - _Z_80 * sigma_total, 2),
        "p50": round(final_f, 2),
        "p90": round(final_f + _Z_80 * sigma_total, 2),
        "sigma_f": round(sigma_total, 2),
        "sigma_components": {
            "gefs": round(float(gefs_sigma), 2) if gefs_sigma else None,
            "model_disagreement": round(math.sqrt(model_disagreement_var), 2),
        },
        "method": "gaussian_variance_decomposition_v1",
    }

# Session 2 — hoisted from inline definitions inside produce(). Caps
# the effective Kalman weight so post-mean re-normalization can't push
# it above this ceiling when other sources drop out.
MAX_KALMAN_WEIGHT = 0.60

# Session 2 — blanket-disabled bias corrections. The correction payload
# still computes and logs (suppressed_delta, suppressed, reason); only
# the final delta applied to the forecast is zeroed. Re-enable a
# single correction by removing its key from this set.
_SESSION_2_SUPPRESSED = {"sea_breeze", "uhi", "cloud_timing",
                         "precip", "inversion"}


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


def _build_qrf_feature_vector(source_values: Dict[str, float],
                               spread: Optional[float],
                               lead_hours: float,
                               regime: Dict[str, Any],
                               target_date: date) -> List[float]:
    """10-element normalized feature vector matching QRF training layout."""
    return _qrf_fv(
        hrrr_f=source_values.get("hrrr"),
        ecmwf_f=source_values.get("ecmwf"),
        gfs_mos_f=source_values.get("gfs_mos"),
        spread=spread,
        lead_hours=lead_hours,
        month=target_date.month,
        sea_breeze=bool(regime.get("sea_breeze_shift")
                        or regime.get("sea_breeze_full")),
        precip=bool(regime.get("any_precip_peak")
                    or regime.get("precip_heavy")),
        cloud_avg=regime.get("cloud_avg"),
    )


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

def _airport_gradient_signals(
        knyc_obs: List[Dict[str, Any]],
        airport_obs_by_station: Dict[str, List[Dict[str, Any]]]
        ) -> Dict[str, Any]:
    """UPGRADE B — compute airport-gradient regime signals.

    ``uhi_signal_f``: KNYC-minus-mean(KLGA/KJFK/KEWR/KTEB) at a 5-8 AM
      observation (sunrise). Positive = KNYC is running warmer than
      the ring, indicating strong UHI. Threshold 3°F → uhi_strong.
    ``sea_breeze_gradient_f``: KEWR-minus-KJFK temperature in the
      afternoon (after 11 AM local). Positive gradient >=5°F = cool
      maritime air has reached KJFK but not the inland Newark gauge,
      meaning a sea breeze is actively active.
    Both are None if we don't have enough station coverage.
    """
    def _latest(obs_list: List[Dict[str, Any]], hour_lo: int, hour_hi: int,
                field: str) -> Optional[float]:
        best: Optional[Tuple[str, float]] = None
        for o in obs_list or []:
            ts = o.get("observed_at")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if not (hour_lo <= dt.hour <= hour_hi):
                continue
            v = o.get(field)
            if v is None:
                continue
            if best is None or ts > best[0]:
                best = (ts, float(v))
        return best[1] if best else None

    out: Dict[str, Any] = {
        "uhi_signal_f": None, "uhi_strong": False,
        "sea_breeze_gradient_f": None, "sea_breeze_confirmed": False,
    }

    # UHI: KNYC vs ring mean at sunrise (5-8 AM local)
    knyc_sunrise = _latest(knyc_obs, 5, 8, "temperature_f")
    ring = []
    for st in ("KLGA", "KJFK", "KEWR", "KTEB"):
        t = _latest(airport_obs_by_station.get(st, []), 5, 8, "temperature_f")
        if t is not None:
            ring.append(t)
    if knyc_sunrise is not None and len(ring) >= 2:
        ring_mean = sum(ring) / len(ring)
        delta = knyc_sunrise - ring_mean
        out["uhi_signal_f"] = round(delta, 2)
        if delta > 3.0:
            out["uhi_strong"] = True

    # Sea breeze: KEWR - KJFK after 11 AM (latest obs available)
    kewr_aft = _latest(airport_obs_by_station.get("KEWR", []), 11, 19,
                       "temperature_f")
    kjfk_aft = _latest(airport_obs_by_station.get("KJFK", []), 11, 19,
                       "temperature_f")
    if kewr_aft is not None and kjfk_aft is not None:
        gradient = kewr_aft - kjfk_aft
        out["sea_breeze_gradient_f"] = round(gradient, 2)
        if gradient > 5.0:
            out["sea_breeze_confirmed"] = True
    return out


def _detect_regime(hourly: List[Dict[str, Any]],
                   obs_today: Optional[List[Dict[str, Any]]] = None,
                   grid_hourly: Optional[List[Dict[str, Any]]] = None,
                   target_date: Optional[date] = None,
                   airport_obs: Optional[Dict[str, List[Dict[str, Any]]]]
                   = None) -> Dict[str, Any]:
    """Extract regime flags used by the bias-correction rules.

    BUG 1: inversion_hint is now populated from obs_today (morning dewpoint
    spread <= 5°F before 8 AM).
    GAP 1: if grid_hourly (quantitative NWS grid data) is provided, its
    numeric fields override string-parsed values from the hourly forecast.
    """
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

    if not hourly and not grid_hourly:
        # Inversion can still be detected from ASOS obs alone.
        iv = False
        iv_strength = "weak"
        if obs_today:
            for o in obs_today:
                ts = o.get("observed_at")
                if not ts:
                    continue
                try:
                    dt = datetime.fromisoformat(ts)
                except Exception:
                    continue
                t_f = o.get("temperature_f")
                d_f = o.get("dewpoint_f")
                if t_f is None or d_f is None:
                    continue
                if 5 <= dt.hour <= 8:
                    sky = o.get("sky_cover_pct")
                    wind = o.get("wind_speed_kt")
                    if ((t_f - d_f) <= 3.0
                            and (sky is None or sky <= 30)
                            and (wind is None or wind <= 5)):
                        iv = True
                        iv_strength = "strong"
                        break
                if dt.hour < 8 and abs(t_f - d_f) <= 5.0:
                    iv = True
        airport_sig = (_airport_gradient_signals(obs_today or [],
                                                 airport_obs or {})
                       if (airport_obs) else
                       {"uhi_signal_f": None, "uhi_strong": False,
                        "sea_breeze_gradient_f": None,
                        "sea_breeze_confirmed": False})
        base = {"sea_breeze_shift": False, "sea_breeze_full": False,
                "wind_nw_all_day": False, "cloud_morning_increase": False,
                "cloud_afternoon_clearing": False, "any_precip_peak": False,
                "precip_heavy": False, "inversion_hint": iv,
                "inversion_strength": iv_strength,
                "calm_clear": False, "sustained_windy": False,
                "cloud_avg": None, "max_wind_kt": None}
        base.update(airport_sig)
        return base

    cloud_before_noon = []
    cloud_after_noon = []
    peak_cloud_morning = 0.0
    precip_probs_peak = []
    wind_speeds = []
    dir_during_afternoon = []
    grid_dir_afternoon: List[float] = []  # hoisted so final assembly can see it

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
        # BUG 2 — build afternoon wind speeds from the SAME hourly entry at
        # the SAME time we check direction; the old code correlated against a
        # flat wind_speeds list that skipped None entries, so it was wildly
        # misaligned on any day with missing wind data.
        afternoon_winds: List[float] = []
        for h in hourly:
            try:
                ts = datetime.fromisoformat(h["time"])
            except Exception:
                continue
            if not (11 <= ts.hour <= 16):
                continue
            wd = (h.get("wind_dir") or "").upper()
            if not (("S" in wd and "SW" not in wd) or "SE" in wd):
                continue
            ws = parse_wind(h.get("wind"))
            if ws is not None:
                afternoon_winds.append(ws)
        if afternoon_winds and max(afternoon_winds) >= 10:
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

    # BUG 1 — detect inversion/fog from this morning's ASOS observations.
    # UPGRADE C — classify STRONG inversion (clear + calm + ≤3°F depression,
    # textbook radiation inversion signature) vs WEAK (just the moisture
    # cue). The strong classification triggers a 1.5× cap in
    # _apply_corrections. Scan observations between midnight and 8 AM;
    # only take the 5-8 AM window seriously for the strong classification
    # since that's the sunrise inversion signal.
    inversion_strength = "weak"
    if obs_today:
        strong_found = False
        for o in obs_today:
            ts = o.get("observed_at")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            t_f = o.get("temperature_f")
            d_f = o.get("dewpoint_f")
            if t_f is None or d_f is None:
                continue

            if 5 <= dt.hour <= 8:
                depression = t_f - d_f
                sky = o.get("sky_cover_pct")
                wind = o.get("wind_speed_kt")
                if (depression <= 3.0
                        and (sky is None or sky <= 30)
                        and (wind is None or wind <= 5)):
                    inversion_hint = True
                    inversion_strength = "strong"
                    strong_found = True
                    break

            # Weak cue: any morning observation with moisture-capped T-Td
            if dt.hour < 8 and abs(t_f - d_f) <= 5.0 and not strong_found:
                inversion_hint = True

    # GAP 1 — if the quantitative NWS grid data is available, override the
    # string-parsed sky/precip/wind values with real numbers.
    if grid_hourly:
        grid_cloud_morning: List[float] = []
        grid_cloud_afternoon: List[float] = []
        grid_precip_peak: List[float] = []
        grid_wind_kt: List[float] = []
        for g in grid_hourly:
            t = g.get("time")
            if not t:
                continue
            try:
                gdt = datetime.fromisoformat(t)
            except Exception:
                continue
            if target_date and gdt.date() != target_date:
                continue
            hour = gdt.hour
            sky = g.get("sky_cover_pct")
            precip_prob = g.get("precip_prob_pct")
            wspd = g.get("wind_speed_kt")
            wdir = g.get("wind_dir_deg")
            if sky is not None:
                if hour < 12:
                    grid_cloud_morning.append(sky)
                else:
                    grid_cloud_afternoon.append(sky)
            if precip_prob is not None and 10 <= hour <= 15:
                grid_precip_peak.append(precip_prob)
            if wspd is not None:
                grid_wind_kt.append(wspd)
            if wdir is not None and 11 <= hour <= 15:
                grid_dir_afternoon.append(wdir)

        if grid_cloud_morning:
            peak_cloud_morning = max(peak_cloud_morning, max(grid_cloud_morning))
            cloud_avg_morn = sum(grid_cloud_morning) / len(grid_cloud_morning)
        if grid_cloud_afternoon:
            cloud_avg_aft = (sum(grid_cloud_afternoon)
                             / len(grid_cloud_afternoon))
        if grid_cloud_morning or grid_cloud_afternoon:
            if peak_cloud_morning >= 50 and (cloud_avg_aft or 0) >= 30:
                cloud_morning_increase = True
            if (cloud_avg_morn or 0) >= 60 and (cloud_avg_aft or 100) <= 30:
                cloud_afternoon_clearing = True
        if grid_precip_peak:
            any_precip_peak = any(p >= 40 for p in grid_precip_peak)
            if any(p >= 70 for p in grid_precip_peak):
                precip_heavy = True
        if grid_wind_kt:
            max_wind = max(max_wind, max(grid_wind_kt))
            sustained_windy = max_wind >= 15
        # Sea breeze from quantitative direction (S=180±45, SE=135±22.5).
        # BUG 3 — `all()` over an empty iterable returns True, so we must
        # guard the NW-all-day check on grid_dir_afternoon being non-empty.
        if grid_dir_afternoon:
            if any(120 <= d <= 210 for d in grid_dir_afternoon):
                sea_breeze_shift = True
            if (grid_dir_afternoon
                    and all(d >= 270 or d <= 45 for d in grid_dir_afternoon)):
                wind_nw_all_day = True

    # Re-evaluate calm_clear with possibly-updated cloud/wind signals
    calm_clear = (max_wind < 10 and (cloud_avg_aft or 100) <= 30
                  and (cloud_avg_morn or 100) <= 50)

    result = {
        "sea_breeze_shift": sea_breeze_shift,
        "sea_breeze_full": sea_breeze_full,
        # BUG 3 — require actual afternoon directional evidence. Don't fall
        # back to "grid_hourly exists" because grid_hourly can be non-empty
        # even when it contains zero wind_direction entries.
        "wind_nw_all_day": wind_nw_all_day and (bool(dir_during_afternoon)
                                                  or bool(grid_dir_afternoon)),
        "cloud_morning_increase": cloud_morning_increase,
        "cloud_afternoon_clearing": cloud_afternoon_clearing,
        "any_precip_peak": any_precip_peak,
        "precip_heavy": precip_heavy,
        "inversion_hint": inversion_hint,
        # UPGRADE C — "strong" = textbook radiation inversion
        # (6 AM depression ≤3°F AND clear AND calm); "weak" = just
        # the moisture cue from the legacy BUG 1 heuristic.
        "inversion_strength": inversion_strength,
        "calm_clear": calm_clear,
        "sustained_windy": sustained_windy,
        "cloud_avg": cloud_avg,
        "max_wind_kt": max_wind,
    }

    # UPGRADE B — airport gradient signals
    if airport_obs:
        gradient = _airport_gradient_signals(obs_today or [], airport_obs)
        result.update(gradient)
    else:
        result["uhi_signal_f"] = None
        result["uhi_strong"] = False
        result["sea_breeze_gradient_f"] = None
        result["sea_breeze_confirmed"] = False

    return result


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
    # UPGRADE B — when full sea-breeze is confirmed by the KEWR-KJFK
    # gradient (>=5°F), scale the penalty by 1.25× (cap, gradient
    # confirms the regime detection).
    if regime.get("sea_breeze_full") and regime.get("sea_breeze_confirmed"):
        sb_delta = sb_delta * 1.25
        grad = regime.get("sea_breeze_gradient_f")
        sb_reason += (f" [KEWR-KJFK {grad:+.1f}°F confirms sea breeze]"
                      if grad is not None else " [airport gradient confirms]")
    corrections["sea_breeze"] = {"delta": sb_delta, "reason": sb_reason}

    # 2. UHI
    if regime["calm_clear"]:
        uhi = bias.get("uhi_clear_calm", cfg.BIAS_DEFAULTS.uhi_clear_calm)
        uhi_reason = "Clear, calm day — urban heat boost"
        # UPGRADE B — if KNYC ran >3°F warmer than the airport ring at
        # sunrise, the UHI signal is visibly active → 1.5× boost.
        if regime.get("uhi_strong"):
            uhi = uhi * 1.5
            sig = regime.get("uhi_signal_f")
            uhi_reason += (f" [KNYC {sig:+.1f}°F vs airport ring at sunrise]"
                           if sig is not None else " [strong UHI signal]")
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
    # UPGRADE C — "strong" inversion (6 AM dewpoint depression ≤3°F AND
    # clear AND calm) triggers a 1.5× cap. Weak inversion (moisture cue
    # only) uses the base bias.
    if month >= 10 or month <= 4:
        if regime.get("inversion_hint"):
            base_inv = bias.get("inversion_winter",
                                cfg.BIAS_DEFAULTS.inversion_winter)
            inv_strength = regime.get("inversion_strength", "weak")
            inv = base_inv * (1.5 if inv_strength == "strong" else 1.0)
            inv_reason = (f"{inv_strength.title()} morning inversion "
                          "/ fog cap")
        else:
            inv = 0.0
            inv_reason = "No inversion cue detected"
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
        # FIX 2 — when spread is high, suppress other corrections but
        # preserve the would-have-been-applied delta on each payload so
        # the attribution ledger can score whether suppression was the
        # right call. Before this, the learning loop silently dropped
        # every high-spread day (no attribution rows written at all),
        # biasing retunes toward calm/predictable days.
        for k in ("sea_breeze", "uhi", "cloud_timing", "precip", "inversion"):
            corrections[k]["suppressed_delta"] = corrections[k]["delta"]
            corrections[k]["suppressed"] = True
            corrections[k]["delta"] = 0.0
            corrections[k]["reason"] += " (suppressed: high spread)"
    else:
        corrections["spread_penalty"] = {
            "delta": 0.0,
            "reason": f"Model spread {spread:.1f}°F — corrections trusted",
        }

    # Session 2 — uniformly suppress all correction deltas until 30+
    # days of per-correction evidence proves value. Record the would-
    # have-fired delta to ``suppressed_delta`` for post-hoc attribution
    # scoring. Re-enable an individual correction by removing its key
    # from the module-level ``_SESSION_2_SUPPRESSED`` set. The prior
    # high-spread suppression block above remains in place; under this
    # blanket suppression it becomes a no-op but we don't remove the code.
    for name in _SESSION_2_SUPPRESSED:
        if name not in corrections:
            continue
        c = corrections[name]
        # Preserve an already-recorded suppressed_delta (e.g. from the
        # high-spread branch above) rather than overwriting with 0.
        if "suppressed_delta" not in c:
            c["suppressed_delta"] = c["delta"]
        c["suppressed"] = True
        c["delta"] = 0.0
        c["reason"] += (
            " (suppressed: Session 2 blanket disable, tracking only)")

    total = sum(c["delta"] for c in corrections.values())

    # FIX 1 — cap total correction magnitude at ±3°F. On regime-shift days
    # (e.g. April 20-21 post-frontal cool-down) corrections empirically
    # stacked to ±6°F, which is a 1-in-30-year adjustment to apply
    # routinely. Scale each component proportionally when the cap fires,
    # but leave ``suppressed_delta`` alone — that field records the
    # counterfactual for attribution.
    CAP = 3.0
    if abs(total) > CAP:
        pre_cap_total = total
        scale = CAP / abs(total)
        for name, c in corrections.items():
            if name == "spread_penalty":
                continue
            c["delta"] = c["delta"] * scale
        total = sum(c["delta"] for c in corrections.values())
        log.warning(
            "Correction cap fired: pre-cap=%+.2f°F scale=%.3f post-cap=%+.2f°F",
            pre_cap_total, scale, total,
        )
        corrections["_meta"] = {
            "cap_fired": True,
            "pre_cap_total": pre_cap_total,
            "scale": scale,
        }
    return total, corrections


# ---- Main entrypoint -------------------------------------------------------

class Forecaster:
    def __init__(self, storage, sources: DataSources,
                 ai_calibrator=None, notifier=None) -> None:
        self.storage = storage
        self.sources = sources
        self.ai_calibrator = ai_calibrator
        # UPGRADE D — optional TelegramNotifier for persistence-warning
        # alerts. Kept optional so existing callers are unaffected.
        self.notifier = notifier
        # BUG 2 — keep the climatology across produce() calls and rebuild
        # at most once per calendar day.
        self._clim: RateClimatology = RateClimatology()
        self._clim_built_date: Optional[date] = None
        # Session 2 G14 — per-day dedupe so the Obsidian cap-fired
        # event note is appended at most once per target_date.
        self._cap_event_fired_for: set[str] = set()
        # PERF 1 — cache BMA results by (target_date, mode) for 60 minutes
        self._bma_cache: Dict[str, Any] = {}
        self._bma_cache_ttl: int = 3600
        # Quant Upgrade 1 — Quantile Regression Forest, rebuilt at most
        # once per calendar day.
        self._qrf: QuantileForest = QuantileForest()
        self._qrf_trained_date: Optional[date] = None

    def _live_bias(self) -> BiasLive:
        values = self.storage.get_biases()
        return BiasLive(values=values)

    def _live_weights(self, mode: str, lead_hours: float) -> Dict[str, float]:
        base = _weights_for_mode(mode, lead_hours)
        stored = self.storage.get_weights()
        # Unique prefix per window matches learning._LEGACY_MAP
        if mode == "night_before":
            prefix = "nb"
        elif lead_hours <= 6:
            prefix = "id06"
        elif lead_hours <= 12:
            prefix = "id612"
        else:
            prefix = "id1224"
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

        # PERF 2 — create ALL non-model tasks in parallel before awaiting.
        # Only ASOS must complete before the diurnal-fit-aware HRRR/ECMWF
        # calls, because their curve-fit weights observations 3×.
        import asyncio
        asos_task    = asyncio.create_task(self.sources.asos_observations())
        hourly_task  = asyncio.create_task(self.sources.nws_hourly_max(target_date))
        nbm_task     = asyncio.create_task(self.sources.nws_daily_max(target_date))
        grid_task    = asyncio.create_task(self.sources.nws_grid_data(target_date))
        gfs_task     = asyncio.create_task(self.sources.gfs_mos(target_date))
        # UPGRADE A — GEFS ensemble spread (native forecast sigma)
        gefs_task    = asyncio.create_task(
            self.sources.gfs_ensemble_spread(target_date))
        # UPGRADE B — KLGA/KJFK/KEWR/KTEB airport ring (regime confirmation)
        airport_task = asyncio.create_task(
            self.sources.airport_observations())

        # Await ASOS first so we can pass obs to the diurnal fitter
        obs = await asos_task

        today_str = target_date.isoformat() if mode == "intraday" else None
        if mode == "intraday" and obs:
            seen = {o["observed_at"]
                    for o in self.storage.observations_today(today_str)}
            for o in obs:
                if o["observed_at"][:10] == today_str \
                        and o["observed_at"] not in seen:
                    self.storage.save_observation(o)
        obs_today = (self.storage.observations_today(today_str)
                     if today_str else [])

        # Kick off model tasks that depend on obs_today
        hrrr_task = asyncio.create_task(
            self.sources.hrrr(target_date, lead_hours=lead_hours,
                              also_obs=obs_today))
        ecmwf_task = asyncio.create_task(
            self.sources.ecmwf(target_date, also_obs=obs_today))

        # Await all remaining tasks concurrently
        (nws_r, nbm_r, grid_r, hrrr_r, ecmwf_r, gfs_r, gefs_r,
         airport_obs_by_station) = (
            await asyncio.gather(
                hourly_task, nbm_task, grid_task, hrrr_task, ecmwf_task,
                gfs_task, gefs_task, airport_task,
            )
        )

        # UPGRADE B — persist new airport observations to DB for history
        today_str_all = target_date.isoformat()
        if airport_obs_by_station:
            for station, station_obs in airport_obs_by_station.items():
                for o in station_obs:
                    if o["observed_at"][:10] != today_str_all:
                        continue
                    try:
                        self.storage.save_airport_observation(station, o)
                    except Exception as e:  # noqa: BLE001
                        log.info("airport obs save failed (%s): %s",
                                 station, e)

        running_max_f, running_max_ts = running_max(obs_today)

        # Assemble raw values
        raw_values: Dict[str, Optional[float]] = {
            "hrrr": hrrr_r.value,
            "nws_point": nws_r.value,
            "gfs_mos": gfs_r.value,
            "ecmwf": ecmwf_r.value,
            "nbm": nbm_r.value,
            # UPGRADE A — GEFS ensemble mean participates in the ensemble
            "gfs_ensemble": gefs_r.value,
        }
        if mode == "intraday":
            raw_values["asos_trend"] = _asos_trend_projection(
                obs_today, nws_r.meta.get("hourly", []), target_date,
            )

        # Superior Quality 1 — Kalman tracker on intraday temperature state.
        # Quant 2 — climatological rate-of-rise prior for stability early
        #           in the day (<4h of obs).
        # Math Gap 2 — soft ceiling against HRRR's forecast peak to
        #              prevent runaway linear projection.
        kalman_result = None
        if mode == "intraday" and len(obs_today) >= 3:
            try:
                from .kalman_tracker import project_daily_max
                # BUG 2 — rebuild climatology at most once per calendar day
                today_d = datetime.now(cfg.EASTERN).date()
                if self._clim_built_date != today_d:
                    try:
                        self._clim.build(self.storage)
                        self._clim_built_date = today_d
                    except Exception as e:  # noqa: BLE001
                        log.info("climatology build error: %s", e)
                _sky_vals = [o.get("sky_cover_pct") for o in obs_today
                             if o.get("sky_cover_pct") is not None]
                _wind_vals = [o.get("wind_speed_kt") for o in obs_today
                              if o.get("wind_speed_kt") is not None]
                prior_rate = self._clim.get_prior_rate(
                    target_date.month,
                    (sum(_sky_vals) / len(_sky_vals)) if _sky_vals else None,
                    (sum(_wind_vals) / len(_wind_vals)) if _wind_vals else None,
                )
                hrrr_peak_f = (hrrr_r.meta.get("raw_max_f")
                               if hrrr_r.meta else None)
                if hrrr_peak_f is None:
                    hrrr_peak_f = hrrr_r.value
                kalman_result = project_daily_max(
                    obs_today,
                    hrrr_r.meta.get("hourly_today"),
                    running_max_f,
                    None,
                    hrrr_peak_f=hrrr_peak_f,
                    prior_rate=prior_rate,
                )
                if kalman_result:
                    raw_values["kalman"] = kalman_result["projected_max_f"]
            except Exception as e:  # noqa: BLE001
                log.warning("Kalman tracker failed: %s", e)

        source_values = {k: v for k, v in raw_values.items() if v is not None}

        weights = self._live_weights(mode, lead_hours)
        # Inject Kalman weight dynamically (blend_weight from the tracker)
        if kalman_result and "kalman" in source_values:
            kw = float(kalman_result.get("blend_weight", 0.2))
            # Scale remaining weights down so they still sum with kalman to 1
            remaining_sum = sum(v for k, v in weights.items() if k != "kalman")
            scale = (1.0 - kw) / remaining_sum if remaining_sum > 0 else 0.0
            weights = {k: v * scale for k, v in weights.items() if k != "kalman"}
            weights["kalman"] = kw

            # BUG 6 — cap Kalman's effective weight. _weighted_mean will
            # re-normalize over only present sources, which can push
            # Kalman above its intended blend_weight ceiling when other
            # sources drop out. Enforce an absolute ceiling here and
            # redistribute any overflow proportionally.
            # Session 2 — ``MAX_KALMAN_WEIGHT`` now lives at module scope.
            if weights["kalman"] > MAX_KALMAN_WEIGHT:
                overflow = weights["kalman"] - MAX_KALMAN_WEIGHT
                weights["kalman"] = MAX_KALMAN_WEIGHT
                others = {k: v for k, v in weights.items() if k != "kalman"}
                others_total = sum(others.values()) or 1.0
                for k in others:
                    weights[k] += overflow * (others[k] / others_total)

        raw_ensemble, views = _weighted_mean(source_values, weights)

        # FIX 3 — re-cap Kalman after _weighted_mean re-normalizes. When
        # other model sources drop out, their missing weight is
        # redistributed and Kalman's realized weight can exceed
        # MAX_KALMAN_WEIGHT even though the pre-mean cap held. Enforce
        # the ceiling again on the realized weights and recompute.
        # Session 2 — uses module-level ``MAX_KALMAN_WEIGHT``.
        if kalman_result and "kalman" in views:
            realized_kw = views["kalman"].weight
            if realized_kw > MAX_KALMAN_WEIGHT + 1e-9:
                overflow = realized_kw - MAX_KALMAN_WEIGHT
                others = {k: v for k, v in views.items() if k != "kalman"}
                others_total = sum(v.weight for v in others.values()) or 1.0
                views["kalman"].weight = MAX_KALMAN_WEIGHT
                for name, sv in others.items():
                    sv.weight += overflow * (sv.weight / others_total)
                raw_ensemble = sum(v.value * v.weight for v in views.values()
                                   if v.value is not None)
                log.debug(
                    "Kalman post-_weighted_mean cap fired: realized %.3f → "
                    "%.3f; sources present=%s weights=%s",
                    realized_kw, MAX_KALMAN_WEIGHT,
                    sorted(views.keys()),
                    {k: round(v.weight, 3) for k, v in views.items()},
                )

        contrib_vals = [v.value for v in views.values()
                        if v.value is not None and v.weight > 0]
        spread = ((max(contrib_vals) - min(contrib_vals))
                  if contrib_vals else 0.0)

        # Quant 1 — Bayesian Model Averaging. If we have enough history,
        # replace the plain weighted mean with the bias-corrected BMA
        # forecast and use its predictive variance to size uncertainty.
        # PERF 1 — cache results by (target_date, mode) for 60 minutes so
        # intraday cycles don't re-run 5+ DB aggregations every 15 min.
        import time as _time
        bma_cache_key = f"{target_date.isoformat()}:{mode}"
        bma_cached = self._bma_cache.get(bma_cache_key)
        bma_result = None
        if bma_cached and (_time.time() - bma_cached[0]) < self._bma_cache_ttl:
            bma_result = bma_cached[1]
        else:
            try:
                from .bma import compute_bma
                bma_result = compute_bma(
                    source_values, weights, self.storage, target_date, mode,
                )
                self._bma_cache[bma_cache_key] = (_time.time(), bma_result)
            except Exception as e:  # noqa: BLE001
                log.info("BMA unavailable: %s", e)
        # Evict cache entries older than 2 hours to bound memory
        now_ts = _time.time()
        self._bma_cache = {k: v for k, v in self._bma_cache.items()
                           if now_ts - v[0] < 7200}
        if bma_result is not None:
            # Session 2 — BMA disabled in output due to phantom bias
            # (+5.96°F Kalman bias not supported by 0.60°F Kalman MAE
            # evidence). BMA continues computing and persisting to
            # extras.bma for observability. Re-enable by uncommenting
            # the line below after 30+ days of validated bias
            # convergence.
            # raw_ensemble = float(bma_result["bma_forecast_f"])
            _bma_disabled_in_output = True  # noqa: F841

        # Corrections — regime now gets obs_today (Bug 1) + grid data (Gap 1)
        # + UPGRADE B airport ring obs (gradient-based regime confirmation)
        bias = self._live_bias()
        regime = _detect_regime(
            nws_r.meta.get("hourly", []),
            obs_today=obs_today,
            grid_hourly=grid_r.meta.get("hourly_grid") if grid_r else None,
            target_date=target_date,
            airport_obs=airport_obs_by_station,
        )
        total_delta, corrections = _apply_corrections(
            target_date, regime, spread, bias
        )

        final = raw_ensemble + total_delta

        # BUG 7 — extract the latest live dewpoint + wind from ASOS and
        # persist them in extras. This gives future analog-library
        # rebuilds access to the *actual* surface humidity/wind at the
        # time of forecast, rather than a hardcoded seasonal fallback.
        live_dewpoint_f: Optional[float] = None
        live_wind_speed_kt: Optional[float] = None
        for o in reversed(obs_today or []):
            if live_dewpoint_f is None and o.get("dewpoint_f") is not None:
                live_dewpoint_f = float(o["dewpoint_f"])
            if live_wind_speed_kt is None and o.get("wind_speed_kt") is not None:
                live_wind_speed_kt = float(o["wind_speed_kt"])
            if live_dewpoint_f is not None and live_wind_speed_kt is not None:
                break

        # Quant 3 — analog pattern matching applied after bias corrections
        # but BEFORE AI calibration (per directive pipeline order).
        analog_result = None
        try:
            from .analog_engine import build_feature_vector, find_analogs
            precip_prob = None
            grid_rows = (grid_r.meta.get("hourly_grid")
                         if grid_r and grid_r.meta else None)
            if grid_rows:
                peaks = [row.get("precip_prob_pct") for row in grid_rows
                         if row.get("precip_prob_pct") is not None]
                if peaks:
                    precip_prob = max(peaks)
            fv = build_feature_vector(
                hrrr_forecast_f=hrrr_r.value if hrrr_r else None,
                dewpoint_f=live_dewpoint_f,
                wind_speed_kt=live_wind_speed_kt,
                precip_prob_pct=precip_prob,
                sky_cover_pct=regime.get("cloud_avg"),
                month=target_date.month,
                lead_hours=lead_hours,
            )
            analog_result = find_analogs(fv, self.storage)
        except Exception as e:  # noqa: BLE001
            log.info("analog engine unavailable: %s", e)
        if analog_result is not None:
            final = final + analog_result.get("analog_bias_f", 0.0)

        # Gap 5 — Claude AI calibration layer (optional). Applied after
        # bias corrections and analog bias, before the running-max floor.
        ai_result = {"delta_f": 0.0, "confidence": 0.0,
                     "reasoning": "AI layer not configured", "source": "disabled"}
        if self.ai_calibrator is not None:
            try:
                recent_scores = self.storage.recent_scores(days=14)
                temp_fc = {
                    "mode": mode,
                    "target_date": target_date.isoformat(),
                    "final_f": round(final, 1),
                    "raw_ensemble_f": round(raw_ensemble, 2),
                    "uncertainty_f": 2.0,
                    "running_asos_max_f": running_max_f,
                    "sources": {
                        k: {"value": raw_values.get(k),
                            "weight": weights.get(k, 0.0) if k in source_values else 0.0,
                            "error": None}
                        for k in raw_values
                    },
                    "corrections": corrections,
                    "extras": {"regime": regime, "spread_f": spread,
                               "lead_hours": lead_hours},
                }
                ai_result = await self.ai_calibrator.calibrate(
                    temp_fc, obs_today, recent_scores
                )
                # FIX 4 — weight the delta by the AI's own confidence.
                # Previously the raw delta was added at full strength
                # regardless of confidence, which amplified noise on
                # low-certainty calls. Soft floor at 0.3: below that the
                # AI is too uncertain to move the forecast.
                ai_delta_raw = float(ai_result.get("delta_f", 0.0) or 0.0)
                ai_confidence = float(ai_result.get("confidence", 0.0) or 0.0)
                if ai_confidence < 0.3:
                    ai_delta_applied = 0.0
                else:
                    ai_delta_applied = ai_delta_raw * ai_confidence
                ai_result["delta_f_raw"] = round(ai_delta_raw, 2)
                ai_result["delta_f_applied"] = round(ai_delta_applied, 2)
                ai_result["confidence"] = round(ai_confidence, 3)
                final = final + ai_delta_applied
            except Exception as e:  # noqa: BLE001
                log.warning("AI calibration failed, continuing without: %s", e)

        # Quant Upgrade 1 — Quantile Regression Forest calibrates a p10/
        # p50/p90 prediction interval around the current forecast state.
        # Applied after bias corrections, analog matching, and AI
        # calibration; before the running-max floor.
        qrf_result = None
        try:
            today_d = datetime.now(cfg.EASTERN).date()
            if self._qrf_trained_date != today_d:
                trained = self._qrf.train(self.storage)
                if trained:
                    self._qrf_trained_date = today_d
            qrf_fv = _build_qrf_feature_vector(
                source_values, spread, lead_hours, regime, target_date,
            )
            qrf_result = self._qrf.predict(qrf_fv)
            if qrf_result is not None:
                p50 = float(qrf_result.get("p50_delta") or 0.0)
                if abs(p50) > 1e-6:
                    final = final + p50
                    log.debug("QRF p50 correction applied: %+.2f°F", p50)
        except Exception as e:  # noqa: BLE001
            log.info("QRF train/predict failed: %s", e)

        # Running-max floor for intraday
        if mode == "intraday" and running_max_f is not None:
            final = max(final, running_max_f)

        # Uncertainty — priority order:
        #   QRF interval width > GEFS native spread > BMA sigma > heuristic
        # UPGRADE A — GEFS ensemble stddev is the natively-calibrated
        # forecast sigma and doesn't need any local training to be
        # meaningful. It carries us through the cold-start window where
        # QRF/BMA haven't yet accumulated enough history.
        gefs_sigma = None
        if gefs_r and gefs_r.meta:
            s = gefs_r.meta.get("sigma_f")
            if s is not None:
                gefs_sigma = float(s)

        if qrf_result is not None:
            # Half of the 80% prediction interval ≈ 1-sigma for a
            # roughly-symmetric residual distribution.
            uncertainty = max(0.5, qrf_result["interval_width"] / 2.0)
        # Session 6 — sigma_total via variance decomposition is now the
        # primary uncertainty estimate when QRF hasn't trained yet.
        # Falls through to heuristic only when both QRF and GEFS are
        # missing.
        # Session 2 — BMA variance no longer contributes to uncertainty
        # because BMA output is disabled. Re-enable alongside BMA output.
        # elif bma_result is not None:
        #     uncertainty = float(bma_result["bma_variance_f"])
        else:
            uncertainty = 2.0
            if spread >= 6:
                uncertainty += 2.0
            elif spread <= 3:
                uncertainty = max(1.0, uncertainty - 1.0)
            if mode == "intraday" and lead_hours <= 3:
                uncertainty = max(0.5, uncertainty - 1.0)

        # FIX 1 — when the correction cap fires, widen stated sigma by 1.5×
        # so the interval honestly reflects that we hit a hard ceiling.
        cap_meta = corrections.get("_meta") or {}
        cap_fired = bool(cap_meta.get("cap_fired"))
        if cap_fired:
            uncertainty = max(uncertainty, uncertainty * 1.5)
            # Session 2 G14 — emit a single Obsidian event note per day.
            date_key = target_date.isoformat()
            if date_key not in self._cap_event_fired_for:
                self._cap_event_fired_for.add(date_key)
                try:
                    from .obsidian_writer import append_event
                    event_body = (
                        f"**Correction cap fired** at {now.isoformat()}\n"
                        f"- Forecast: {final:.1f}°F\n"
                        f"- Pre-cap total: "
                        f"{cap_meta.get('pre_cap_total', 0):+.2f}°F\n"
                        f"- Scale applied: "
                        f"{cap_meta.get('scale', 1.0):.3f}"
                    )
                    append_event("cap-fired", event_body, target_date)
                except Exception as e:  # noqa: BLE001
                    log.info("obsidian cap-fired event skipped: %s", e)

        # UPGRADE D — persistence sanity check. If forecast diverges
        # from yesterday's CLI truth by >10°F with no significant regime
        # change, log a warning, stash extras.persistence_warning = True,
        # and (if a notifier is wired) enqueue a Telegram alert. This is
        # informational ONLY — never auto-correct, or real regime shifts
        # get silently suppressed.
        persistence_warning = False
        try:
            from datetime import timedelta as _td
            yday = (target_date - _td(days=1)).isoformat()
            yday_cli = self.storage.get_cli_truth(yday)
            yday_high = (float(yday_cli["recorded_high_f"])
                         if yday_cli and yday_cli.get("recorded_high_f") is not None
                         else None)
            yday_fc = self.storage.latest_forecast(yday)
            yday_regime = ((yday_fc.get("extras", {}) or {}).get("regime", {})
                           if yday_fc else {}) or {}
            regime_changed = (
                regime.get("any_precip_peak") != yday_regime.get("any_precip_peak")
                or regime.get("sea_breeze_full") != yday_regime.get("sea_breeze_full")
                or regime.get("calm_clear") != yday_regime.get("calm_clear")
            )
            if (yday_high is not None
                    and abs(final - yday_high) > 10.0
                    and not regime_changed):
                persistence_warning = True
                log.warning(
                    "PERSISTENCE_MISMATCH: forecast %.1f°F vs yesterday "
                    "%.1f°F (%+.1f) with no regime change; check sources",
                    final, yday_high, final - yday_high,
                )
                if self.notifier is not None and getattr(
                        self.notifier, "configured", False):
                    self.notifier.enqueue(
                        "⚠️ <b>Persistence sanity check failed</b>\n"
                        f"Forecast: {final:.1f}°F | Yesterday: "
                        f"{yday_high:.1f}°F | "
                        "No regime change — verify sources before trusting."
                    )
        except Exception as e:  # noqa: BLE001
            log.info("persistence sanity check skipped: %s", e)

        # Session 6 Part 1 — probabilistic bands via variance
        # decomposition. Combines GEFS member spread (initial-condition
        # uncertainty) and across-model disagreement (model-physics
        # uncertainty) into sigma_total. Excludes gfs_ensemble from
        # disagreement to avoid double-counting GEFS spread. Bands are
        # additive to extras; uncertainty_f is re-derived from
        # sigma_total for back-compat.
        contributing_sources = [
            (name, raw_values.get(name), weights.get(name, 0.0))
            for name in source_values
            if raw_values.get(name) is not None
        ]
        bands = _compute_forecast_bands(
            final_f=round(final, 2),
            gefs_sigma=gefs_sigma,
            contributing_sources=contributing_sources,
        )
        if cap_fired:
            inflated = round(bands["sigma_f"] * 1.5, 2)
            bands["sigma_f"] = inflated
            bands["p10"] = round(final - _Z_80 * inflated, 2)
            bands["p90"] = round(final + _Z_80 * inflated, 2)
        uncertainty = bands["sigma_f"]

        revision = self.storage.last_revision(target_date.isoformat(), mode) + 1 \
            if mode == "intraday" else 0

        prior = self.storage.latest_forecast(target_date.isoformat())
        delta_prior = (final - prior["final_f"]) if prior else None

        # Build sources dict for storage (include all attempted sources)
        sources_persist: Dict[str, Dict[str, Any]] = {}
        for name in ("hrrr", "nws_point", "gfs_mos",
                     "ecmwf", "nbm", "asos_trend", "kalman",
                     "gfs_ensemble"):
            sources_persist[name] = {
                "value": raw_values.get(name),
                "weight": weights.get(name, 0.0) if name in source_values else 0.0,
                "error": {
                    "hrrr": hrrr_r.error, "nws_point": nws_r.error,
                    "gfs_mos": gfs_r.error,
                    "ecmwf": ecmwf_r.error, "nbm": nbm_r.error,
                    "gfs_ensemble": gefs_r.error,
                }.get(name),
            }

        # Diurnal fit + HRRR model metadata for dashboard visibility
        if hrrr_r.meta.get("fit_quality") is not None:
            sources_persist["hrrr"]["fit_quality"] = hrrr_r.meta["fit_quality"]
            sources_persist["hrrr"]["fitted_peak_hour"] = hrrr_r.meta.get(
                "fitted_peak_hour")
            sources_persist["hrrr"]["model_used"] = hrrr_r.meta.get("model_used")
        if ecmwf_r.meta.get("fit_quality") is not None:
            sources_persist["ecmwf"]["fit_quality"] = ecmwf_r.meta["fit_quality"]
            sources_persist["ecmwf"]["fitted_peak_hour"] = ecmwf_r.meta.get(
                "fitted_peak_hour")

        extras = {
            "regime": regime,
            "spread_f": spread,
            "lead_hours": lead_hours,
            "weights_used": weights,
            "kalman": kalman_result,
            "ai_calibration": ai_result,
            "bma": bma_result,
            "analog": analog_result,
            "qrf": qrf_result,
            # BUG 7 — persist live surface fields so the analog library
            # rebuild can use actual dewpoint/wind instead of the
            # hardcoded seasonal fallback.
            "surface_dewpoint_f": live_dewpoint_f,
            "surface_wind_speed_kt": live_wind_speed_kt,
            "running_max_observed_at": running_max_ts,
            # UPGRADE A — GEFS ensemble statistics (natively-calibrated
            # sigma). Also carries n_members and p10/p50/p90.
            "gfs_ensemble": ({
                "value": gefs_r.value,
                "sigma_f": gefs_r.meta.get("sigma_f"),
                "p10": gefs_r.meta.get("p10"),
                "p50": gefs_r.meta.get("p50"),
                "p90": gefs_r.meta.get("p90"),
                "n_members": gefs_r.meta.get("n_members"),
            } if gefs_r and gefs_r.value is not None else None),
            # FIX 1 — correction cap fired on this cycle (±3°F ceiling).
            "cap_fired": cap_fired,
            # UPGRADE D — informational flag if |forecast - yesterday| >10°F
            # with no regime change. Never auto-corrects the forecast.
            "persistence_warning": persistence_warning,
            # Session 6 Part 1 — probabilistic bands (p10/p50/p90)
            # via variance decomposition. sigma_total combines GEFS
            # member spread (initial-condition uncertainty) and
            # across-model disagreement (model-physics uncertainty).
            "bands": bands,
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
