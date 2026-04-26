"""Plain-English observability layer over /api/state.

Pure derivation: no DB, no HTTP, no awaits, no side effects. The single
public entry point is :func:`build_health_summary`, which never raises;
on malformed input it returns a red-status dict whose ``headline``
explains what failed.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfg


SOURCE_META: Dict[str, Dict[str, Any]] = {
    "hrrr":         {"label": "HRRR (3 km)",        "ttl_min": 75},
    "nws_point":    {"label": "NWS Point Forecast", "ttl_min": 75},
    "gfs_mos":      {"label": "GFS-MOS",            "ttl_min": 180},
    "ecmwf":        {"label": "ECMWF (Open-Meteo)", "ttl_min": 180},
    "nbm":          {"label": "NBM",                "ttl_min": 180},
    "asos_trend":   {"label": "KNYC ASOS trend",    "ttl_min": 45},
    "kalman":       {"label": "Kalman projection",  "ttl_min": 30},
    "gfs_ensemble": {"label": "GEFS ensemble",      "ttl_min": 360},
}

SOURCE_ORDER = [
    "kalman", "hrrr", "ecmwf", "nws_point",
    "gfs_mos", "nbm", "asos_trend", "gfs_ensemble",
]

CORRECTION_LABEL = {
    "sea_breeze":     "Sea breeze",
    "uhi":            "Urban heat island",
    "cloud_timing":   "Cloud timing",
    "precip":         "Precipitation",
    "inversion":      "Inversion / fog",
    "spread_penalty": "Model spread",
}


# --------------------------------------------------------------------------- #
# Small parse helpers                                                         #
# --------------------------------------------------------------------------- #

def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_dt(s: Any) -> Optional[datetime]:
    if not isinstance(s, str):
        return None
    try:
        # Allow trailing Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _parse_date(s: Any) -> Optional[date]:
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    if not isinstance(s, str):
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _now_et(state: Dict[str, Any]) -> datetime:
    raw = state.get("now_et") if isinstance(state, dict) else None
    dt = _parse_dt(raw) if raw else None
    if dt is None:
        dt = datetime.now(cfg.EASTERN)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=cfg.EASTERN)
    return dt


# --------------------------------------------------------------------------- #
# Source health                                                               #
# --------------------------------------------------------------------------- #

def _classify_source(
    name: str,
    current: Dict[str, Any],
    last_seen: Optional[Tuple[datetime, Optional[float]]],
    now_et: datetime,
) -> Tuple[str, str]:
    meta = SOURCE_META.get(name, {})
    ttl_min = int(meta.get("ttl_min") or 180)
    cur_val = _safe_float(current.get("value"))
    err = current.get("error") or ""

    if cur_val is not None:
        return "green", "reporting"

    if last_seen is not None:
        last_dt, last_val = last_seen
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=cfg.EASTERN)
        age_min = max(0.0, (now_et - last_dt).total_seconds() / 60.0)
        if age_min <= ttl_min:
            val_txt = f"{last_val:.1f}°F" if last_val is not None else "n/a"
            extra = f" — {err}" if err else ""
            return "yellow", f"last value {int(age_min)} min ago ({val_txt}){extra}"
        extra = f" — {err}" if err else ""
        return "red", f"stale: last value {int(age_min)} min ago, TTL {ttl_min}{extra}"

    extra = f" — {err}" if err else ""
    return "red", f"no value ever{extra}"


def _source_health(
    anchor: Dict[str, Any],
    recent_fc: List[Dict[str, Any]],
    now_et: datetime,
) -> List[Dict[str, Any]]:
    sources = (anchor or {}).get("sources") or {}
    extras = (anchor or {}).get("extras") or {}

    # last_seen map: source -> (issued_dt, value) of most recent non-None
    last_seen: Dict[str, Tuple[datetime, Optional[float]]] = {}
    try:
        ordered = sorted(
            recent_fc or [],
            key=lambda fc: _parse_dt(fc.get("issued_at")) or datetime.min.replace(tzinfo=cfg.EASTERN),
            reverse=True,
        )
    except Exception:
        ordered = list(recent_fc or [])

    for fc in ordered:
        src_block = (fc or {}).get("sources") or {}
        issued = _parse_dt(fc.get("issued_at"))
        if issued is None:
            continue
        for name in SOURCE_ORDER:
            if name in last_seen:
                continue
            entry = src_block.get(name) or {}
            v = _safe_float(entry.get("value"))
            if v is not None:
                last_seen[name] = (issued, v)

    out: List[Dict[str, Any]] = []
    for name in SOURCE_ORDER:
        meta = SOURCE_META.get(name, {})
        cur = sources.get(name) or {}
        cur_val = _safe_float(cur.get("value"))
        weight = _safe_float(cur.get("weight"))
        err = cur.get("error") or ""

        status, reason = _classify_source(name, cur, last_seen.get(name), now_et)

        item: Dict[str, Any] = {
            "name": name,
            "label": meta.get("label", name),
            "value_f": round(cur_val, 2) if cur_val is not None else None,
            "weight_pct": round(weight * 100) if weight is not None else None,
            "status": status,
            "status_reason": reason,
            "error": err,
        }

        if name == "gfs_ensemble":
            ens = extras.get("gfs_ensemble") or {}
            n_members = ens.get("n_members")
            sigma_f = _safe_float(ens.get("sigma_f"))
            item["n_members"] = n_members
            item["sigma_f"] = round(sigma_f, 2) if sigma_f is not None else None
            try:
                if n_members is not None and int(n_members) <= 1:
                    item["status"] = "yellow"
                    item["status_reason"] = (
                        f"only {int(n_members)} ensemble member parsed — "
                        "sigma unreliable (Session 1 fix didn't land)"
                    )
            except (TypeError, ValueError):
                pass

        out.append(item)

    return out


# --------------------------------------------------------------------------- #
# What-if (BMA shadow + correction shadow)                                    #
# --------------------------------------------------------------------------- #

def _what_if(anchor: Dict[str, Any]) -> Dict[str, Any]:
    anchor = anchor or {}
    extras = anchor.get("extras") or {}
    final_f = _safe_float(anchor.get("final_f")) or 0.0

    bma_block = extras.get("bma") or {}
    bma_fc = _safe_float(bma_block.get("bma_forecast_f"))
    if bma_fc is not None:
        bma_out: Optional[Dict[str, Any]] = {
            "enabled": False,
            "shadow_forecast_f": round(bma_fc, 2),
            "delta_vs_final_f": round(bma_fc - final_f, 2),
            "variance_f": bma_block.get("bma_variance_f"),
            "note": "BMA still computing; output disabled in Session 2",
        }
    else:
        bma_out = None

    corrections_block = anchor.get("corrections") or {}
    per: List[Dict[str, Any]] = []
    any_suppressed = False
    total_shadow_delta = 0.0

    for name, block in corrections_block.items():
        if name == "_meta" or not isinstance(block, dict):
            continue
        suppressed = bool(block.get("suppressed"))
        applied = _safe_float(block.get("delta")) or 0.0
        if suppressed:
            shadow = _safe_float(block.get("suppressed_delta")) or 0.0
            total_shadow_delta += shadow
            any_suppressed = True
        else:
            shadow = applied
        per.append({
            "name": name,
            "label": CORRECTION_LABEL.get(name, name),
            "applied_delta_f": round(applied, 2),
            "shadow_delta_f": round(shadow, 2),
            "suppressed": suppressed,
            "reason": block.get("reason") or "",
        })

    corrections_out = {
        "per_correction": per,
        "any_suppressed": any_suppressed,
        "total_shadow_delta_f": round(total_shadow_delta, 2),
        "shadow_forecast_f": round(final_f + total_shadow_delta, 2),
        "enabled": not any_suppressed,
        "note": (
            "All corrections suppressed Session 2 — tracking only"
            if any_suppressed
            else "Corrections active"
        ),
    }

    return {"bma": bma_out, "corrections": corrections_out}


# --------------------------------------------------------------------------- #
# Regime narrative                                                            #
# --------------------------------------------------------------------------- #

def _regime_narrative(anchor: Dict[str, Any]) -> str:
    if not anchor:
        return "No regime data yet — waiting for first cycle."
    extras = anchor.get("extras") or {}
    regime = extras.get("regime")
    if not isinstance(regime, dict):
        return "Regime detector returned no flags this cycle."

    parts: List[str] = []

    if regime.get("precip_heavy"):
        parts.append("heavy precipitation expected in peak window")
    if regime.get("any_precip_peak"):
        parts.append("precip likely during peak heating")
    if regime.get("sea_breeze_full"):
        parts.append("full sea-breeze penetration expected")
    if regime.get("sea_breeze_shift"):
        parts.append("afternoon wind shift to S/SE expected")
    if regime.get("calm_clear"):
        parts.append("clear + calm (UHI conditions)")
    if regime.get("sustained_windy"):
        parts.append("sustained windy — atmosphere well-mixed")
    if regime.get("wind_nw_all_day"):
        parts.append("W/NW wind all day (no sea-breeze)")
    if regime.get("inversion_hint"):
        strength = regime.get("inversion_strength") or "weak"
        parts.append(f"{strength} morning inversion hint")
    if regime.get("cloud_morning_increase"):
        parts.append("cloud cover increasing before 1 PM")
    if regime.get("cloud_afternoon_clearing"):
        parts.append("morning overcast clears after noon")
    cloud_avg = regime.get("cloud_avg")
    if cloud_avg is not None:
        try:
            parts.append(f"avg sky {int(round(float(cloud_avg)))}%")
        except (TypeError, ValueError):
            pass

    if not parts:
        return "Neutral regime — no sea breeze, no precip, no UHI signal."
    return "Regime: " + "; ".join(parts) + "."


# --------------------------------------------------------------------------- #
# Kalman info                                                                 #
# --------------------------------------------------------------------------- #

def _kalman_info(
    anchor: Dict[str, Any],
    recent_fc: List[Dict[str, Any]],
    scores: List[Dict[str, Any]],
) -> Dict[str, Any]:
    sources = (anchor or {}).get("sources") or {}
    extras = (anchor or {}).get("extras") or {}
    k_cur = sources.get("kalman") or {}
    k_extras = extras.get("kalman") or {}

    cur_val = _safe_float(k_cur.get("value"))
    weight = _safe_float(k_cur.get("weight"))

    # Reconstruct per-date Kalman projection from intraday forecasts
    per_date_kalman: Dict[date, float] = {}
    try:
        ordered = sorted(
            recent_fc or [],
            key=lambda fc: _parse_dt(fc.get("issued_at")) or datetime.min.replace(tzinfo=cfg.EASTERN),
            reverse=True,
        )
    except Exception:
        ordered = list(recent_fc or [])

    for fc in ordered:
        if (fc or {}).get("mode") != "intraday":
            continue
        td = _parse_date(fc.get("target_date"))
        if td is None or td in per_date_kalman:
            continue
        kproj = _safe_float(((fc.get("extras") or {}).get("kalman") or {}).get("projected_max_f"))
        if kproj is not None:
            per_date_kalman[td] = kproj

    # Reconstruct per-date CLI high from scores + recent_fc
    per_date_cli: Dict[date, float] = {}
    fc_index: Dict[Tuple[date, str], Dict[str, Any]] = {}
    for fc in recent_fc or []:
        td = _parse_date(fc.get("target_date"))
        mode = fc.get("mode")
        if td is not None and isinstance(mode, str):
            fc_index.setdefault((td, mode), fc)

    for s in scores or []:
        td = _parse_date(s.get("target_date"))
        if td is None or td in per_date_cli:
            continue
        mode = s.get("mode")
        err = _safe_float(s.get("error_f"))
        if not isinstance(mode, str) or err is None:
            continue
        fc = fc_index.get((td, mode))
        if not fc:
            continue
        final_f = _safe_float(fc.get("final_f"))
        if final_f is None:
            continue
        per_date_cli[td] = final_f + err

    errs = [
        abs(per_date_kalman[td] - per_date_cli[td])
        for td in per_date_kalman
        if td in per_date_cli
    ]
    mae_f = round(mean(errs), 2) if errs else None
    samples = len(errs)

    if mae_f is None:
        verdict = "unscored"
        narrative = "Kalman MAE: not yet enough verified days."
        healthy = False
    else:
        if mae_f < 1.0:
            verdict = "elite"
        elif mae_f < 2.0:
            verdict = "healthy"
        elif mae_f < 3.0:
            verdict = "mediocre"
        else:
            verdict = "struggling"
        healthy = mae_f < 1.5
        narrative = f"Kalman {verdict} — {mae_f:.2f}°F MAE across {samples} verified days."

    return {
        "value_f": round(cur_val, 2) if cur_val is not None else None,
        "weight_pct": round(weight * 100) if weight is not None else None,
        "mae_f": mae_f,
        "samples": samples,
        "healthy": healthy,
        "narrative": narrative,
        "hours_of_data": k_extras.get("hours_of_data"),
        "blend_weight": k_extras.get("blend_weight"),
    }


# --------------------------------------------------------------------------- #
# Vitals                                                                      #
# --------------------------------------------------------------------------- #

def _vitals(
    stats: Dict[str, Any],
    kalman_info: Dict[str, Any],
    qrf_cal: Dict[str, Any],
    scores: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stats = stats or {}
    qrf_cal = qrf_cal or {}
    samples_block = stats.get("samples") or {}
    nb_n = int(samples_block.get("night_before") or 0)
    in_n = int(samples_block.get("intraday") or 0)

    intraday_mae = _safe_float(stats.get("intraday_mae"))
    night_before_mae = _safe_float(stats.get("night_before_mae"))
    target_intraday = _safe_float(stats.get("target_intraday_mae"))
    target_night_before = _safe_float(stats.get("target_night_before_mae"))

    coverage = _safe_float(qrf_cal.get("empirical_coverage"))
    if coverage is not None:
        insigma_pct: Optional[float] = round(coverage * 100.0, 1)
        insigma_target_pct = 80.0
        try:
            insigma_samples = int(qrf_cal.get("samples") or 0)
        except (TypeError, ValueError):
            insigma_samples = 0
    else:
        insigma_pct = None
        insigma_target_pct = 68.0
        insigma_samples = 0

    return {
        "intraday_mae_f": round(intraday_mae, 2) if intraday_mae is not None else None,
        "intraday_mae_target_f": round(target_intraday, 2) if target_intraday is not None else None,
        "night_before_mae_f": round(night_before_mae, 2) if night_before_mae is not None else None,
        "night_before_mae_target_f": (
            round(target_night_before, 2) if target_night_before is not None else None
        ),
        "kalman_mae_f": kalman_info.get("mae_f"),
        "kalman_samples": kalman_info.get("samples", 0),
        "insigma_pct": insigma_pct,
        "insigma_target_pct": insigma_target_pct,
        "insigma_samples": insigma_samples,
        "samples": {
            "night_before": nb_n,
            "intraday": in_n,
            "total": nb_n + in_n,
        },
    }


# --------------------------------------------------------------------------- #
# Cold-start progress                                                         #
# --------------------------------------------------------------------------- #

def _cold_start(
    stats: Dict[str, Any],
    qrf_cal: Dict[str, Any],
    kalman_info: Dict[str, Any],
) -> Dict[str, Any]:
    stats = stats or {}
    qrf_cal = qrf_cal or {}
    samples_block = stats.get("samples") or {}
    nb_n = int(samples_block.get("night_before") or 0)
    in_n = int(samples_block.get("intraday") or 0)
    days_scored = max(nb_n, in_n)

    try:
        qrf_samples = int(qrf_cal.get("samples") or 0)
    except (TypeError, ValueError):
        qrf_samples = 0

    k_samples = int(kalman_info.get("samples") or 0)

    def _block(needed: int, have: int) -> Dict[str, Any]:
        progress = min(100, round(have / needed * 100)) if needed > 0 else 100
        return {
            "needed": needed,
            "have": have,
            "progress_pct": progress,
            "unlocks_in_days": max(0, needed - have),
        }

    return {
        "days_scored": days_scored,
        "analog": _block(14, days_scored),
        "qrf": _block(30, qrf_samples),
        "kalman_verified": _block(14, k_samples),
    }


# --------------------------------------------------------------------------- #
# Countdowns                                                                  #
# --------------------------------------------------------------------------- #

def _next_quarter(now_et: datetime) -> datetime:
    minute = now_et.minute
    next_q = ((minute // 15) + 1) * 15
    if next_q >= 60:
        base = now_et.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return base
    return now_et.replace(minute=next_q, second=0, microsecond=0)


def _next_at(now_et: datetime, hour: int, minute: int = 0) -> datetime:
    today_at = now_et.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now_et < today_at:
        return today_at
    return today_at + timedelta(days=1)


def _next_night_before(now_et: datetime) -> datetime:
    slots = [(h, m) for h in range(18, 24) for m in (0, 30)]
    today = now_et.date()
    for h, m in slots:
        cand = datetime(today.year, today.month, today.day, h, m, tzinfo=now_et.tzinfo)
        if cand > now_et:
            return cand
    tomorrow = today + timedelta(days=1)
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day, 18, 0, tzinfo=now_et.tzinfo)


def _countdowns(now_et: datetime) -> Dict[str, Any]:
    next_intraday = _next_quarter(now_et)
    next_cli = _next_at(now_et, 17, 0)
    cli_open = (
        time(17, 0) <= now_et.time() <= time(21, 59, 59)
    )
    next_morning = _next_at(now_et, 7, 0)
    next_nb = _next_night_before(now_et)

    def _pair(target: datetime) -> Tuple[str, int]:
        return target.isoformat(), max(0, int((target - now_et).total_seconds()))

    intraday_iso, intraday_sec = _pair(next_intraday)
    cli_iso, cli_sec = _pair(next_cli)
    morning_iso, morning_sec = _pair(next_morning)
    nb_iso, nb_sec = _pair(next_nb)

    return {
        "next_intraday_iso": intraday_iso,
        "next_intraday_in_sec": intraday_sec,
        "next_cli_iso": cli_iso,
        "next_cli_in_sec": cli_sec,
        "cli_window_open": cli_open,
        "next_morning_iso": morning_iso,
        "next_morning_in_sec": morning_sec,
        "next_night_before_iso": nb_iso,
        "next_night_before_in_sec": nb_sec,
    }


# --------------------------------------------------------------------------- #
# Open issues                                                                 #
# --------------------------------------------------------------------------- #

def _open_issues(
    anchor: Dict[str, Any],
    recent_fc: List[Dict[str, Any]],
    logs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    extras = (anchor or {}).get("extras") or {}

    # GEFS one-member
    ens = extras.get("gfs_ensemble") or {}
    n_members = ens.get("n_members")
    try:
        if n_members is not None and int(n_members) <= 1:
            issues.append({
                "severity": "medium",
                "id": "gefs_one_member",
                "title": "GEFS ensemble parsing stuck at 1 member",
                "detail": (
                    "Session 1 fix did not land — sigma unreliable, ensemble "
                    "reduces to a point forecast. Investigate GRIB member decode."
                ),
            })
    except (TypeError, ValueError):
        pass

    # ECMWF rate limit (last 10 forecasts)
    try:
        ordered = sorted(
            recent_fc or [],
            key=lambda fc: _parse_dt(fc.get("issued_at")) or datetime.min.replace(tzinfo=cfg.EASTERN),
            reverse=True,
        )[:10]
    except Exception:
        ordered = list(recent_fc or [])[:10]

    rate_hits = 0
    rate_total = 0
    for fc in ordered:
        ecm = ((fc or {}).get("sources") or {}).get("ecmwf") or {}
        err = ecm.get("error")
        if isinstance(err, str) and err:
            rate_total += 1
            if "rate" in err.lower():
                rate_hits += 1
    if rate_total > 0 and rate_hits / rate_total >= 0.5:
        issues.append({
            "severity": "medium",
            "id": "ecmwf_rate_limit",
            "title": "ECMWF rate-limited on Open-Meteo",
            "detail": (
                f"{rate_hits}/{rate_total} recent cycles hit rate-limit. "
                "Likely shared-IP contention."
            ),
        })

    # Kalshi 401 noise
    kalshi_401 = 0
    for entry in logs or []:
        msg = ""
        if isinstance(entry, dict):
            msg = str(entry.get("message") or entry.get("msg") or "")
        elif isinstance(entry, str):
            msg = entry
        m = msg.lower()
        if "401" in m and "kalshi" in m:
            kalshi_401 += 1
    if kalshi_401 >= 3:
        issues.append({
            "severity": "low",
            "id": "kalshi_401_noise",
            "title": f"Kalshi 401 log noise ({kalshi_401} recent)",
            "detail": (
                "Session 2 filter attempt failed due to variable-name mismatch "
                "— deferred to Session 4."
            ),
        })

    # Persistence warning
    if extras.get("persistence_warning") is True:
        issues.append({
            "severity": "high",
            "id": "persistence_warning",
            "title": "Persistence sanity check failed",
            "detail": (
                "Today's forecast differs from yesterday's CLI by >10°F with "
                "no regime change — sources need review."
            ),
        })

    return issues


# --------------------------------------------------------------------------- #
# Alerts                                                                      #
# --------------------------------------------------------------------------- #

def _alerts(
    anchor: Dict[str, Any],
    recent_fc: List[Dict[str, Any]],
    regime_shifts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []

    try:
        ordered = sorted(
            recent_fc or [],
            key=lambda fc: _parse_dt(fc.get("issued_at")) or datetime.min.replace(tzinfo=cfg.EASTERN),
            reverse=True,
        )[:20]
    except Exception:
        ordered = list(recent_fc or [])[:20]

    cap_count = 0
    for fc in ordered:
        if ((fc or {}).get("extras") or {}).get("cap_fired") is True:
            cap_count += 1
    if cap_count >= 1:
        alerts.append({
            "kind": "cap_fired",
            "message": (
                f"Correction cap fired on {cap_count} of last {len(ordered)} "
                "forecast cycles — corrections would have stacked past ±3°F threshold."
            ),
        })

    if regime_shifts:
        alerts.append({
            "kind": "regime_shift",
            "message": f"{len(regime_shifts)} regime shift(s) flagged in last 7 days.",
        })

    return alerts


# --------------------------------------------------------------------------- #
# Bet scorecard                                                               #
# --------------------------------------------------------------------------- #

def _bet_scorecard(
    stats: Dict[str, Any],
    vitals: Dict[str, Any],
    what_if: Dict[str, Any],
) -> Dict[str, Any]:
    final_mae = vitals.get("intraday_mae_f")
    kalman_mae = vitals.get("kalman_mae_f")

    if final_mae is not None and kalman_mae is not None:
        gap = round(final_mae - kalman_mae, 2)
        if gap < 0.5:
            verdict = "paying off — final converging on Kalman"
        elif gap < 1.5:
            verdict = "improving — gap narrowing"
        elif gap < 3.0:
            verdict = "mixed — ensemble lags Kalman, expected early"
        else:
            verdict = "not yet — ensemble still much worse than Kalman alone"
    else:
        gap = None
        verdict = "not enough data"

    return {
        "final_mae_f": final_mae,
        "kalman_mae_f": kalman_mae,
        "gap_f": gap,
        "verdict": verdict,
        "note": (
            "Thesis: final MAE should converge toward Kalman MAE as disabled "
            "components are gradually re-enabled based on evidence."
        ),
    }


# --------------------------------------------------------------------------- #
# Forecast narrative                                                          #
# --------------------------------------------------------------------------- #

def _forecast_narrative(
    latest_today: Optional[Dict[str, Any]],
    latest_tomorrow: Optional[Dict[str, Any]],
    truth_today: Optional[Dict[str, Any]],
    running_max: Optional[float],
    kalman_info: Dict[str, Any],
    what_if: Dict[str, Any],
) -> str:
    sentences: List[str] = []

    if truth_today:
        cli = _safe_float(truth_today.get("cli_high_f")) or _safe_float(truth_today.get("value"))
        if cli is not None:
            sentences.append(f"CLI is in: Central Park hit {int(round(cli))}°F today.")
        else:
            sentences.append("CLI is in for today.")
    elif latest_today:
        f_val = _safe_float(latest_today.get("final_f"))
        sigma = _safe_float(latest_today.get("sigma_f"))
        mode = latest_today.get("mode") or "intraday"
        rev = latest_today.get("revision") or latest_today.get("rev") or 0
        if f_val is not None:
            sigma_txt = f"{sigma:.1f}" if sigma is not None else "n/a"
            sentences.append(
                f"Today's forecast: {f_val:.1f}°F ±{sigma_txt}°F ({mode}, rev {rev})."
            )
        if running_max is not None:
            sentences.append(
                f"Running ASOS max so far: {running_max:.1f}°F (observed peak so far today)."
            )
        k_val = kalman_info.get("value_f")
        if k_val is not None:
            sentences.append(f"Kalman projection: {k_val:.1f}°F.")

    corrections = (what_if or {}).get("corrections") or {}
    total_shadow = _safe_float(corrections.get("total_shadow_delta_f")) or 0.0
    shadow_fc = _safe_float(corrections.get("shadow_forecast_f"))
    if shadow_fc is not None and abs(total_shadow) >= 0.1:
        sign = "+" if total_shadow >= 0 else "-"
        sentences.append(
            f"If suppressed corrections were active: {shadow_fc:.1f}°F "
            f"({sign}{abs(total_shadow):.1f}°F vs current)."
        )

    bma = (what_if or {}).get("bma")
    if bma:
        delta = _safe_float(bma.get("delta_vs_final_f")) or 0.0
        shadow = _safe_float(bma.get("shadow_forecast_f"))
        if shadow is not None and abs(delta) >= 0.1:
            sign = "+" if delta >= 0 else "-"
            sentences.append(
                f"If BMA output were enabled: {shadow:.1f}°F "
                f"({sign}{abs(delta):.1f}°F vs current)."
            )

    if latest_tomorrow:
        f_val = _safe_float(latest_tomorrow.get("final_f"))
        sigma = _safe_float(latest_tomorrow.get("sigma_f"))
        if f_val is not None:
            sigma_txt = f"{sigma:.1f}" if sigma is not None else "n/a"
            sentences.append(f"Tomorrow's night-before: {f_val:.1f}°F ±{sigma_txt}°F.")

    if not sentences:
        return "No intraday forecast yet."
    return " ".join(sentences)


# --------------------------------------------------------------------------- #
# Overall status                                                              #
# --------------------------------------------------------------------------- #

def _overall_status(
    anchor: Dict[str, Any],
    sources: List[Dict[str, Any]],
    vitals: Dict[str, Any],
    open_issues: List[Dict[str, Any]],
    alerts: List[Dict[str, Any]],
    truth_today: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    # RED checks
    if not anchor:
        return "red", "No forecast produced yet — waiting on first cycle."

    high = [i for i in open_issues if i.get("severity") == "high"]
    if high:
        return "red", f"{high[0].get('title', 'High-severity issue')}."

    red_sources = [s for s in sources if s.get("status") == "red"]
    if len(red_sources) >= 2:
        names = ", ".join(s.get("name", "?") for s in red_sources)
        return "red", f"Multiple sources down: {names}."

    # YELLOW checks
    intraday_mae = vitals.get("intraday_mae_f")
    intraday_target = vitals.get("intraday_mae_target_f")
    if (
        intraday_mae is not None
        and intraday_target is not None
        and intraday_target > 0
        and intraday_mae > intraday_target * 1.5
    ):
        return "yellow", (
            f"Intraday MAE {intraday_mae:.1f}°F is above target of "
            f"{intraday_target:.1f}°F — Session 2 bet still playing out."
        )

    medium = [i for i in open_issues if i.get("severity") == "medium"]
    if medium:
        return "yellow", f"{medium[0].get('title', 'Medium-severity issue')}."

    yellow_sources = [s for s in sources if s.get("status") == "yellow"]
    if yellow_sources:
        names = ", ".join(s.get("name", "?") for s in yellow_sources[:2])
        return "yellow", f"Sources degraded: {names}."

    if len(red_sources) == 1:
        return "yellow", f"Source down: {red_sources[0].get('name', '?')} — others covering."

    # GREEN
    if truth_today:
        cli = _safe_float(truth_today.get("cli_high_f")) or _safe_float(truth_today.get("value"))
        if cli is not None:
            return "green", f"CLI verified at {int(round(cli))}°F — day is closed."
        return "green", "CLI verified — day is closed."
    return "green", "All sources reporting; forecast healthy."


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #

def build_health_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a structured health summary from the /api/state payload.

    Returns a dict with all top-level keys always present (never
    missing, so the dashboard never needs defensive checks). Returns
    a red-status dict with a headline explaining what failed rather
    than raising on malformed state input.
    """
    try:
        state = state if isinstance(state, dict) else {}
        now_et = _now_et(state)

        anchor = state.get("anchor") if isinstance(state.get("anchor"), dict) else {}
        recent_fc = state.get("recent_forecasts") or state.get("recent_fc") or []
        if not isinstance(recent_fc, list):
            recent_fc = []
        scores = state.get("scores") or []
        if not isinstance(scores, list):
            scores = []
        stats = state.get("stats") if isinstance(state.get("stats"), dict) else {}
        qrf_cal = state.get("qrf_calibration") if isinstance(state.get("qrf_calibration"), dict) else {}
        logs = state.get("logs") or []
        if not isinstance(logs, list):
            logs = []
        regime_shifts = state.get("regime_shifts") or []
        if not isinstance(regime_shifts, list):
            regime_shifts = []
        latest_today = state.get("latest_today") if isinstance(state.get("latest_today"), dict) else None
        latest_tomorrow = (
            state.get("latest_tomorrow") if isinstance(state.get("latest_tomorrow"), dict) else None
        )
        truth_today = state.get("truth_today") if isinstance(state.get("truth_today"), dict) else None
        running_max = _safe_float(state.get("running_max_f"))

        sources = _source_health(anchor, recent_fc, now_et)
        what_if = _what_if(anchor)
        regime_narrative = _regime_narrative(anchor)
        kalman_info = _kalman_info(anchor, recent_fc, scores)
        vitals = _vitals(stats, kalman_info, qrf_cal, scores)
        cold_start = _cold_start(stats, qrf_cal, kalman_info)
        countdowns = _countdowns(now_et)
        open_issues = _open_issues(anchor, recent_fc, logs)
        alerts = _alerts(anchor, recent_fc, regime_shifts)
        bet_scorecard = _bet_scorecard(stats, vitals, what_if)
        forecast_narrative = _forecast_narrative(
            latest_today, latest_tomorrow, truth_today, running_max, kalman_info, what_if
        )
        status, headline = _overall_status(
            anchor, sources, vitals, open_issues, alerts, truth_today
        )

        return {
            "status": status,
            "headline": headline,
            "forecast_narrative": forecast_narrative,
            "sources": sources,
            "what_if": what_if,
            "regime_narrative": regime_narrative,
            "kalman": kalman_info,
            "vitals": vitals,
            "cold_start": cold_start,
            "countdowns": countdowns,
            "open_issues": open_issues,
            "alerts": alerts,
            "bet_scorecard": bet_scorecard,
        }
    except Exception as exc:
        # Never raise — fall through to a red-status payload that still has
        # every top-level key the dashboard expects.
        try:
            now_et = datetime.now(cfg.EASTERN)
            countdowns = _countdowns(now_et)
        except Exception:
            countdowns = {
                "next_intraday_iso": "",
                "next_intraday_in_sec": 0,
                "next_cli_iso": "",
                "next_cli_in_sec": 0,
                "cli_window_open": False,
                "next_morning_iso": "",
                "next_morning_in_sec": 0,
                "next_night_before_iso": "",
                "next_night_before_in_sec": 0,
            }
        return {
            "status": "red",
            "headline": f"Health summary failed to build: {type(exc).__name__}: {exc}",
            "forecast_narrative": "No data available.",
            "sources": [],
            "what_if": {"bma": None, "corrections": {
                "per_correction": [], "any_suppressed": False, "total_shadow_delta_f": 0.0,
                "shadow_forecast_f": 0.0, "enabled": True, "note": "Corrections active",
            }},
            "regime_narrative": "No regime data yet — waiting for first cycle.",
            "kalman": {
                "value_f": None, "weight_pct": None, "mae_f": None, "samples": 0,
                "healthy": False, "narrative": "Kalman MAE: not yet enough verified days.",
                "hours_of_data": None, "blend_weight": None,
            },
            "vitals": {
                "intraday_mae_f": None, "intraday_mae_target_f": None,
                "night_before_mae_f": None, "night_before_mae_target_f": None,
                "kalman_mae_f": None, "kalman_samples": 0,
                "insigma_pct": None, "insigma_target_pct": 68.0, "insigma_samples": 0,
                "samples": {"night_before": 0, "intraday": 0, "total": 0},
            },
            "cold_start": {
                "days_scored": 0,
                "analog": {"needed": 14, "have": 0, "progress_pct": 0, "unlocks_in_days": 14},
                "qrf": {"needed": 30, "have": 0, "progress_pct": 0, "unlocks_in_days": 30},
                "kalman_verified": {"needed": 14, "have": 0, "progress_pct": 0, "unlocks_in_days": 14},
            },
            "countdowns": countdowns,
            "open_issues": [],
            "alerts": [],
            "bet_scorecard": {
                "final_mae_f": None, "kalman_mae_f": None, "gap_f": None,
                "verdict": "not enough data",
                "note": (
                    "Thesis: final MAE should converge toward Kalman MAE as disabled "
                    "components are gradually re-enabled based on evidence."
                ),
            },
        }
