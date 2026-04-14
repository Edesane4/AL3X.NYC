"""Async fetchers for every forecasting data source described in the directive.

Each fetcher returns a normalized dict. All fetchers catch transport errors
and return a structure with `error` populated so the ensemble can continue
without that source.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from . import config as cfg

log = logging.getLogger("al3x.data")


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _mps_to_kt(v: float) -> float:
    return v * 1.9438445


def _now_eastern() -> datetime:
    return datetime.now(cfg.EASTERN)


@dataclass
class SourceResult:
    source: str
    value: Optional[float] = None          # predicted max °F for target date
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "value": self.value,
                "error": self.error, "meta": self.meta}


class DataSources:
    """Thin async wrapper that fans out HTTP calls with shared client."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=cfg.HTTP_TIMEOUT,
            headers={"User-Agent": cfg.USER_AGENT,
                     "Accept": "application/geo+json, application/json, text/html"},
            follow_redirects=True,
        )
        self._points_cache: Optional[Dict[str, str]] = None
        # Async lock: only one /points/ lookup at a time (three NWS tasks
        # race each other on the first cycle otherwise).
        import asyncio as _asyncio
        self._points_lock = _asyncio.Lock()

    async def close(self) -> None:
        await self._client.aclose()

    async def _nws_endpoints(self) -> Dict[str, str]:
        """Resolve the dynamic /points/{lat},{lon} endpoint (once)."""
        if self._points_cache:
            return self._points_cache
        async with self._points_lock:
            if self._points_cache:
                return self._points_cache
            url = f"https://api.weather.gov/points/{cfg.LAT},{cfg.LON}"
            r = await self._client.get(url)
            r.raise_for_status()
            props = r.json().get("properties", {})
            self._points_cache = {
                "forecast": props.get("forecast"),
                "forecastHourly": props.get("forecastHourly"),
                "forecastGridData": props.get("forecastGridData"),
            }
            return self._points_cache

    # ---- NWS point forecast (hourly) ------------------------------------
    async def nws_hourly_max(self, target_date: date) -> SourceResult:
        try:
            ep = await self._nws_endpoints()
            if not ep.get("forecastHourly"):
                return SourceResult("nws_point",
                                    error="no forecastHourly url from /points")
            r = await self._client.get(ep["forecastHourly"])
            r.raise_for_status()
            js = r.json()
            periods = js.get("properties", {}).get("periods", [])
            vals: List[float] = []
            hourly: List[Dict[str, Any]] = []
            for p in periods:
                start = datetime.fromisoformat(p["startTime"])
                start = start.astimezone(cfg.EASTERN)
                if start.date() != target_date:
                    continue
                t = p.get("temperature")
                if t is None:
                    continue
                if p.get("temperatureUnit", "F") == "C":
                    t = _c_to_f(t)
                vals.append(float(t))
                hourly.append({
                    "time": start.isoformat(),
                    "temp_f": float(t),
                    "wind": p.get("windSpeed"),
                    "wind_dir": p.get("windDirection"),
                    "sky_cover": p.get("skyCover"),
                    "short": p.get("shortForecast"),
                    "precip_prob": (p.get("probabilityOfPrecipitation") or {}).get("value"),
                })
            if not vals:
                return SourceResult("nws_point", error="no periods for target date")
            return SourceResult("nws_point", value=max(vals),
                                meta={"hourly": hourly})
        except Exception as e:  # noqa: BLE001
            log.warning("nws hourly fetch failed: %s", e)
            return SourceResult("nws_point", error=str(e))

    # ---- NWS / NBM daily ------------------------------------------------
    async def nws_daily_max(self, target_date: date) -> SourceResult:
        try:
            ep = await self._nws_endpoints()
            if not ep.get("forecast"):
                return SourceResult("nbm", error="no forecast url from /points")
            r = await self._client.get(ep["forecast"])
            r.raise_for_status()
            js = r.json()
            periods = js.get("properties", {}).get("periods", [])
            for p in periods:
                if not p.get("isDaytime"):
                    continue
                start = datetime.fromisoformat(p["startTime"]).astimezone(cfg.EASTERN)
                if start.date() == target_date:
                    t = p.get("temperature")
                    if p.get("temperatureUnit") == "C":
                        t = _c_to_f(t)
                    return SourceResult("nbm", value=float(t),
                                        meta={"short": p.get("shortForecast")})
            return SourceResult("nbm", error="no daytime period for target")
        except Exception as e:  # noqa: BLE001
            log.warning("nws daily fetch failed: %s", e)
            return SourceResult("nbm", error=str(e))

    # ---- GAP 1: quantitative NWS grid data ------------------------------
    async def nws_grid_data(self, target_date: date) -> SourceResult:
        """Fetch the raw numerical gridded forecast (sky %, precip %, dewpoint
        °F, wind speed kt, wind dir deg) for target_date.

        Replaces string-matching of NWS shortForecast with real numbers for
        regime detection.
        """
        try:
            ep = await self._nws_endpoints()
            if not ep.get("forecastGridData"):
                return SourceResult("nws_grid",
                                    error="no forecastGridData url")
            r = await self._client.get(ep["forecastGridData"])
            r.raise_for_status()
            props = r.json().get("properties", {}) or {}

            hourly: Dict[str, Dict[str, Any]] = {}
            _insert_grid_series(hourly, props.get("skyCover"),
                                "sky_cover_pct", target_date, to_f=False)
            _insert_grid_series(hourly,
                                props.get("probabilityOfPrecipitation"),
                                "precip_prob_pct", target_date, to_f=False)
            # NWS gridData reports temperature / dewpoint in °C via wmoUnit.
            _insert_grid_series(hourly, props.get("temperature"),
                                "temperature_f", target_date, to_f=True)
            _insert_grid_series(hourly, props.get("dewpoint"),
                                "dewpoint_f", target_date, to_f=True)
            # Wind speed in km/h per NWS convention -> convert to kt.
            _insert_grid_series(hourly, props.get("windSpeed"),
                                "wind_speed_kt", target_date,
                                convert_kmh_to_kt=True)
            _insert_grid_series(hourly, props.get("windDirection"),
                                "wind_dir_deg", target_date, to_f=False)

            if not hourly:
                return SourceResult("nws_grid", error="no hourly grid series")

            series = [v for _, v in sorted(hourly.items())]
            return SourceResult("nws_grid", value=None,
                                meta={"hourly_grid": series})
        except Exception as e:  # noqa: BLE001
            log.warning("nws grid fetch failed: %s", e)
            return SourceResult("nws_grid", error=str(e))

    # ---- KNYC observations (via NWS /stations/KNYC/observations) --------
    async def asos_observations(self) -> List[Dict[str, Any]]:
        """Live KNYC METAR observations for the past ~48 hours.

        Uses the NWS station observations endpoint, which is the canonical
        feed for the same ASOS station and doesn't require parsing IEM's
        less-stable schema.
        """
        url = (f"https://api.weather.gov/stations/{cfg.STATION_ID}"
               "/observations?limit=72")
        try:
            r = await self._client.get(url)
            r.raise_for_status()
            js = r.json()
            obs: List[Dict[str, Any]] = []
            for f in js.get("features", []):
                p = f.get("properties", {}) or {}
                ts = p.get("timestamp")
                if not ts:
                    continue
                try:
                    ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    continue
                temp_c = (p.get("temperature") or {}).get("value")
                dew_c = (p.get("dewpoint") or {}).get("value")
                wind_dir = (p.get("windDirection") or {}).get("value")
                wind_mps = (p.get("windSpeed") or {}).get("value")
                layers = p.get("cloudLayers") or []
                sky_pct = None
                if layers:
                    best = 0.0
                    for layer in layers:
                        v = _sky_to_pct(layer.get("amount"))
                        if v is not None and v > best:
                            best = v
                    sky_pct = best
                elif p.get("textDescription"):
                    td = p["textDescription"].lower()
                    if "clear" in td or "sunny" in td:
                        sky_pct = 0.0
                    elif "overcast" in td:
                        sky_pct = 100.0
                obs.append({
                    "observed_at": ts_dt.astimezone(cfg.EASTERN).isoformat(),
                    "temperature_f": _c_to_f(temp_c) if temp_c is not None else None,
                    "wind_dir_deg": wind_dir,
                    "wind_speed_kt": (_mps_to_kt(wind_mps)
                                      if wind_mps is not None else None),
                    "dewpoint_f": _c_to_f(dew_c) if dew_c is not None else None,
                    "sky_cover_pct": sky_pct,
                    "raw": {"textDescription": p.get("textDescription")},
                })
            obs.sort(key=lambda o: o["observed_at"])
            return obs
        except Exception as e:  # noqa: BLE001
            log.warning("asos fetch failed: %s", e)
            return []

    # ---- GFS-MOS / NAM-MOS (via IEM) ------------------------------------
    #
    # Iowa Environmental Mesonet hosts a per-station MOS text endpoint that
    # returns the latest available model run in MAV/MET format — ~2 KB per
    # request, no cycle-walking, no national-bulletin slicing. We cache the
    # response in-process for 1 hour so the intraday cycle doesn't re-fetch
    # on every 15-minute tick.
    async def _fetch_mos_iem(self, url: str,
                              source_name: str) -> Optional[str]:
        cache = getattr(self, "_mos_cache", None)
        if cache is None:
            cache: Dict[str, Dict[str, Any]] = {}
            self._mos_cache = cache

        # Cache key is source + current UTC hour → one fetch per hour
        hour_key = datetime.now(timezone.utc).strftime("%Y%m%d%H")
        key = f"{source_name}:{hour_key}"
        if key in cache:
            return cache[key].get("text")

        try:
            r = await self._client.get(url, timeout=10.0)
            if r.status_code == 200 and len(r.text) > 200:
                # Evict older cache entries for this source
                stale = [k for k in cache
                         if k.startswith(f"{source_name}:") and k != key]
                for k in stale:
                    cache.pop(k, None)
                cache[key] = {"text": r.text, "bytes": len(r.text), "url": url}
                log.info("%s fetched from IEM (%d bytes)",
                         source_name, len(r.text))
                return r.text
            log.info("%s IEM returned HTTP %s (len=%d)",
                     source_name, r.status_code, len(r.text))
            return None
        except Exception as e:  # noqa: BLE001
            msg = str(e) or type(e).__name__
            log.info("%s IEM fetch error (%s): %s",
                     source_name, type(e).__name__, msg)
            return None

    async def mos(self, url: str, source_name: str,
                  target_date: date) -> SourceResult:
        try:
            txt = await self._fetch_mos_iem(url, source_name)
            if not txt:
                return SourceResult(source_name, error="no bulletin fetched")
            val = _parse_mos_max(txt, target_date)
            if val is None:
                log.info("%s: bulletin parsed but no X/N for target date",
                         source_name)
                return SourceResult(source_name, error="no max field parsed")
            return SourceResult(source_name, value=val,
                                meta={"raw_len": len(txt)})
        except Exception as e:  # noqa: BLE001
            msg = str(e) or type(e).__name__
            log.info("%s fetch skipped (%s): %s",
                     source_name, type(e).__name__, msg)
            return SourceResult(source_name, error=f"{type(e).__name__}: {msg}")

    async def gfs_mos(self, target_date: date) -> SourceResult:
        return await self.mos(cfg.GFS_MOS_URL, "gfs_mos", target_date)

    async def nam_mos(self, target_date: date) -> SourceResult:
        return await self.mos(cfg.NAM_MOS_URL, "nam_mos", target_date)

    # ---- Open-Meteo: HRRR + ECMWF --------------------------------------
    async def open_meteo(self, target_date: date, model: str,
                         name: str,
                         also_obs: Optional[List[Dict[str, Any]]] = None
                         ) -> SourceResult:
        """Fetch hourly temperatures from Open-Meteo and derive the daily max.

        Superior Quality 2 — when scipy is available, the raw hourly profile
        is fed through diurnal_fit.fit_peak() to get the analytical peak
        (which catches between-hour maxima), blending any available ASOS
        obs for `also_obs`.
        """
        try:
            params = {
                "latitude": cfg.LAT, "longitude": cfg.LON,
                "hourly": "temperature_2m",
                "models": model,
                "temperature_unit": "fahrenheit",
                "timezone": "America/New_York",
                "forecast_days": 3,
                "past_days": 1,
            }
            r = await self._client.get(cfg.OPEN_METEO, params=params)
            r.raise_for_status()
            js = r.json()
            times = js.get("hourly", {}).get("time", [])
            temps = js.get("hourly", {}).get("temperature_2m", [])
            if not times:
                return SourceResult(name, error="empty hourly")

            hourly_today: List[Tuple[float, float]] = []
            for ts, t in zip(times, temps):
                if t is None:
                    continue
                dt = _parse_dt(ts)
                if dt.date() != target_date:
                    continue
                hourly_today.append((dt.hour + dt.minute / 60.0, float(t)))

            if not hourly_today:
                return SourceResult(name, error="no target-date hours")

            raw_max = max(t for _, t in hourly_today)

            fitted_max = raw_max
            fit_quality = None
            fitted_peak_hour = None
            try:
                from .diurnal_fit import fit_peak
                fit = fit_peak(hourly_today, also_obs or [], target_date)
                if fit is not None:
                    fitted_max = fit["fitted_max_f"]
                    fit_quality = fit.get("r_squared")
                    fitted_peak_hour = fit.get("fitted_peak_hour")
            except Exception as e:  # noqa: BLE001
                log.info("diurnal fit unavailable for %s: %s", name, e)

            return SourceResult(
                name,
                value=fitted_max,
                meta={
                    "hours_count": len(hourly_today),
                    "raw_max_f": raw_max,
                    "fit_quality": fit_quality,
                    "fitted_peak_hour": fitted_peak_hour,
                    "hourly_today": hourly_today,
                },
            )
        except Exception as e:  # noqa: BLE001
            log.warning("%s fetch failed: %s", name, e)
            return SourceResult(name, error=str(e))

    async def hrrr(self, target_date: date, lead_hours: float = 12.0,
                   also_obs: Optional[List[Dict[str, Any]]] = None
                   ) -> SourceResult:
        """BUG 5 — use the actual HRRR model for short lead times instead of
        best_match (which Open-Meteo blends across models).

        - lead_hours < 18: prefer ncep_hrrr (3km HRRR)
        - lead_hours >= 18: use gfs_seamless (GFS is HRRR's day+1 parent)
        - if the preferred model fails or returns no data, fall back to
          best_match and log a specific warning.
        """
        # Open-Meteo's HRRR is served under the gfs_hrrr model id (not
        # ncep_hrrr — that returns 400 Bad Request).
        primary = "gfs_hrrr" if lead_hours < 18 else "gfs_seamless"
        res = await self.open_meteo(target_date, primary, "hrrr",
                                    also_obs=also_obs)
        if res.value is not None:
            res.meta["model_used"] = primary
            return res
        log.warning("HRRR primary model %s unavailable (%s); falling back "
                    "to best_match", primary, res.error)
        fb = await self.open_meteo(target_date, "best_match", "hrrr",
                                   also_obs=also_obs)
        fb.meta["model_used"] = "best_match (fallback)"
        fb.meta["primary_error"] = res.error
        return fb

    async def ecmwf(self, target_date: date,
                    also_obs: Optional[List[Dict[str, Any]]] = None
                    ) -> SourceResult:
        return await self.open_meteo(target_date, "ecmwf_ifs04", "ecmwf",
                                     also_obs=also_obs)

    # ---- CLI verification feed -----------------------------------------
    async def cli_latest(self) -> Optional[Dict[str, Any]]:
        try:
            r = await self._client.get(cfg.NWS_CLI)
            r.raise_for_status()
            return _parse_cli(r.text)
        except Exception as e:  # noqa: BLE001
            log.warning("CLI fetch failed: %s", e)
            return None


# ---------- Helpers ----------------------------------------------------------

_SKY_MAP = {"CLR": 0, "SKC": 0, "FEW": 15, "SCT": 40, "BKN": 75,
            "OVC": 100, "VV": 100}


def _sky_to_pct(code: Any) -> Optional[float]:
    if code is None:
        return None
    s = str(code).upper().strip()
    for k, v in _SKY_MAP.items():
        if s.startswith(k):
            return float(v)
    return None


def _parse_dt(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M")


def _iter_grid_series(series: Optional[Dict[str, Any]]):
    """NWS grid series = {"uom": ..., "values": [{"validTime": "...", "value": x}]}.

    validTime is ISO8601 interval "2026-04-14T12:00:00+00:00/PT1H"; expand to
    per-hour entries for slicing.
    """
    if not series or not series.get("values"):
        return
    for v in series["values"]:
        vt = v.get("validTime") or ""
        if "/" not in vt:
            continue
        start_s, dur = vt.split("/", 1)
        try:
            start = datetime.fromisoformat(start_s)
        except Exception:
            continue
        # parse ISO-8601 duration PTxH (we only need hours; fall back to 1h)
        m = re.search(r"PT(\d+)H", dur)
        hours = int(m.group(1)) if m else 1
        val = v.get("value")
        if val is None:
            continue
        for h in range(hours):
            yield (start + timedelta(hours=h)).astimezone(cfg.EASTERN), val


def _insert_grid_series(out: Dict[str, Dict[str, Any]],
                         series: Optional[Dict[str, Any]],
                         key: str, target_date: date,
                         to_f: bool = False,
                         convert_kmh_to_kt: bool = False) -> None:
    for ts, val in _iter_grid_series(series):
        if ts.date() != target_date:
            continue
        iso_key = ts.isoformat()
        slot = out.setdefault(iso_key, {"time": iso_key})
        v = val
        if to_f:
            v = _c_to_f(val)
        elif convert_kmh_to_kt:
            v = val * 0.539957
        slot[key] = v


_MONTH_ABBR = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _extract_knyc_block(text: str) -> Optional[str]:
    """Pull the KNYC section from a national MAV/MET text bulletin.

    MAV/MET format: each station's block starts with a line whose first token
    is the ICAO identifier (e.g. `KNYC   GFS MOS GUIDANCE   4/14/2026  1200 UTC`)
    and runs until the next station header.
    """
    start = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{cfg.STATION_ID} ") and "MOS GUIDANCE" in stripped:
            start = i
            break
    if start is None:
        return None
    # Find the next station header (4-letter ICAO + "MOS GUIDANCE")
    end = len(lines)
    for j in range(start + 1, len(lines)):
        s = lines[j].strip()
        if "MOS GUIDANCE" in s and re.match(r"^[A-Z0-9]{4}\s", s):
            end = j
            break
    return "\n".join(lines[start:end])


def _parse_mos_max(text: str, target_date: date) -> Optional[float]:
    """Parse a MAV/MET national bulletin (or KNYC-only text) to find the
    daily max temperature valid for ``target_date``.

    Robust approach: anchor by character position, not fixed column width.
      1. Extract the KNYC block.
      2. On the DT line, record each `/MON DD` token's start column.
      3. On the X/N line, record each numeric token's start column.
      4. For each X/N token, its date = the most recent DT token at
         or before this column's position.
      5. Return max of X/N values assigned to target_date.day (prefers
         maxima over minima when both fall on the same day).
    """
    try:
        block = _extract_knyc_block(text) or text
        lines = block.splitlines()
        dt_line = None
        xn_line = None
        for ln in lines:
            s = ln.lstrip()
            if dt_line is None and s.startswith("DT "):
                dt_line = ln
            elif xn_line is None and s.startswith("X/N "):
                xn_line = ln
            if dt_line and xn_line:
                break
        if not xn_line:
            return None

        # Locate each /MON DD token on DT line
        dt_tokens: List[Tuple[int, int]] = []  # (col, day)
        if dt_line:
            for m in re.finditer(r"/([A-Z]{3})\s+(\d{1,2})", dt_line):
                dt_tokens.append((m.start(), int(m.group(2))))

        # Locate each numeric on X/N line (skip the "X/N" label itself)
        xn_body_start = xn_line.lstrip().find("X/N") + len("X/N")
        xn_numbers: List[Tuple[int, int]] = []  # (col, value)
        for m in re.finditer(r"-?\d+", xn_line):
            if m.start() < len(xn_line) - len(xn_line.lstrip()) + xn_body_start:
                continue  # inside the "X/N" label — skip
            try:
                xn_numbers.append((m.start(), int(m.group(0))))
            except ValueError:
                continue

        if not xn_numbers:
            return None

        # Assign each X/N number to the day of the most recent DT token
        # whose column is <= this number's column.
        def _day_for(col: int) -> Optional[int]:
            day = None
            for tcol, d in dt_tokens:
                if tcol <= col:
                    day = d
                else:
                    break
            return day

        candidates: List[int] = []
        for col, v in xn_numbers:
            if _day_for(col) == target_date.day:
                candidates.append(v)
        if candidates:
            return float(max(candidates))

        # Fallback: the first X/N value (earliest period in the bulletin)
        return float(xn_numbers[0][1])
    except Exception as e:  # noqa: BLE001
        log.debug("mos parse error: %s", e)
        return None


def _parse_cli(text: str) -> Optional[Dict[str, Any]]:
    plain = re.sub(r"<[^>]+>", "\n", text)
    plain = re.sub(r"&nbsp;", " ", plain)
    plain = re.sub(r"[ \t]+", " ", plain)

    date_match = re.search(
        r"CLIMATE REPORT.*?VALID[^\n]*?"
        r"(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|"
        r"SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(\d{1,2})\s+(\d{4})",
        plain, re.IGNORECASE | re.DOTALL,
    )
    target_dt: Optional[date] = None
    if date_match:
        try:
            target_dt = datetime.strptime(
                f"{date_match.group(1)} {date_match.group(2)} {date_match.group(3)}",
                "%B %d %Y",
            ).date()
        except Exception:
            pass

    max_match = re.search(r"MAXIMUM\s+(-?\d+)", plain, re.IGNORECASE)
    if not max_match:
        return None
    try:
        recorded = float(max_match.group(1))
    except ValueError:
        return None

    return {
        "target_date": target_dt.isoformat() if target_dt else None,
        "recorded_high_f": recorded,
        "posted_at": datetime.now(timezone.utc).isoformat(),
        "raw_text": plain[:4000],
    }


def running_max(obs_today: List[Dict[str, Any]]) -> Tuple[Optional[float],
                                                           Optional[str]]:
    best: Optional[Tuple[float, str]] = None
    for o in obs_today:
        t = o.get("temperature_f")
        if t is None:
            continue
        if best is None or t > best[0]:
            best = (float(t), o["observed_at"])
    return (best[0], best[1]) if best else (None, None)


def high_confirmed(obs_today: List[Dict[str, Any]]) -> bool:
    now = _now_eastern()
    if now.hour < 14:
        return False
    if len(obs_today) < 3:
        return False

    temps = [(o["observed_at"], o.get("temperature_f")) for o in obs_today
             if o.get("temperature_f") is not None]
    if not temps:
        return False

    max_f = max(t for _, t in temps)
    max_idx = max(range(len(temps)), key=lambda i: temps[i][1])
    after_max = temps[max_idx + 1:]
    if not after_max:
        return False

    last_temp = after_max[-1][1]
    if max_f - last_temp < 2.0:
        return False

    two_hours_ago = now - timedelta(hours=2)
    recent = [t for ts, t in after_max
              if datetime.fromisoformat(ts) >= two_hours_ago]
    if len(recent) < 2:
        return False
    return all(t < max_f for t in recent)
