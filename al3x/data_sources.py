"""Async fetchers for every forecasting data source described in the directive.

Each fetcher returns a normalized dict. All fetchers catch transport errors
and return a structure with `error` populated so the ensemble can continue
without that source.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
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

    async def close(self) -> None:
        await self._client.aclose()

    async def _nws_endpoints(self) -> Dict[str, str]:
        """Resolve the dynamic /points/{lat},{lon} endpoint (once)."""
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
                    # Highest coverage among layers (BKN/OVC > SCT > FEW > SKC)
                    best = 0.0
                    for layer in layers:
                        v = _sky_to_pct(layer.get("amount"))
                        if v is not None and v > best:
                            best = v
                    sky_pct = best
                elif p.get("textDescription"):
                    # crude fallback
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
            # Return chronologically ascending
            obs.sort(key=lambda o: o["observed_at"])
            return obs
        except Exception as e:  # noqa: BLE001
            log.warning("asos fetch failed: %s", e)
            return []

    # ---- GFS-MOS / NAM-MOS ---------------------------------------------
    async def mos(self, product_url: str, source_name: str,
                  target_date: date) -> SourceResult:
        try:
            r = await self._client.get(
                product_url,
                params={"stationId": cfg.STATION_ID},
            )
            r.raise_for_status()
            txt = r.text
            val = _parse_mos_max(txt, target_date)
            if val is None:
                return SourceResult(source_name, error="no max field parsed")
            return SourceResult(source_name, value=val, meta={"raw_len": len(txt)})
        except Exception as e:  # noqa: BLE001
            log.warning("%s fetch failed: %s", source_name, e)
            return SourceResult(source_name, error=str(e))

    async def gfs_mos(self, target_date: date) -> SourceResult:
        return await self.mos(cfg.MDL_MOS, "gfs_mos", target_date)

    async def nam_mos(self, target_date: date) -> SourceResult:
        return await self.mos(cfg.MDL_NAMMOS, "nam_mos", target_date)

    # ---- Open-Meteo: HRRR + ECMWF --------------------------------------
    async def open_meteo(self, target_date: date, model: str,
                         name: str) -> SourceResult:
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
            vals = [t for ts, t in zip(times, temps)
                    if t is not None and _parse_dt(ts).date() == target_date]
            if not vals:
                return SourceResult(name, error="no target-date hours")
            return SourceResult(name, value=max(vals),
                                meta={"hours_count": len(vals)})
        except Exception as e:  # noqa: BLE001
            log.warning("%s fetch failed: %s", name, e)
            return SourceResult(name, error=str(e))

    async def hrrr(self, target_date: date) -> SourceResult:
        return await self.open_meteo(target_date, "best_match", "hrrr")

    async def ecmwf(self, target_date: date) -> SourceResult:
        return await self.open_meteo(target_date, "ecmwf_ifs04", "ecmwf")

    # ---- CLI verification feed -----------------------------------------
    async def cli_latest(self) -> Optional[Dict[str, Any]]:
        """Fetch the CLI product and parse recorded max + issuance date."""
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


def _parse_mos_max(text: str, target_date: date) -> Optional[float]:
    """Parse a GFS-MOS / NAM-MOS text bulletin to pull max temp for target_date.

    MOS bulletins list FHR HH values across multiple days. The simpler,
    more resilient strategy is to scan for lines starting with 'X/N' (max/min)
    or 'TMP' and correlate with the DT header. We take the first MAX value
    whose DT falls on the target day (UTC → local approximation by
    treating DT as the verification date).

    Returns None if parsing fails.
    """
    # KNYC bulletins often use the pattern:
    # "DT /MMDD/.../HH/ .."   with X/N row for daily max/min
    # This is intentionally forgiving; we accept the first X/N value.
    try:
        # Narrow to the KNYC section if present
        m = re.search(rf"{cfg.STATION_ID}[^\n]*(?:\n.+){{0,40}}", text)
        block = m.group(0) if m else text

        # Find DT header to know which columns are max temps
        dt_line = re.search(r"^\s*DT\s+([^\n]+)$", block, re.MULTILINE)
        xn_line = re.search(r"^\s*X/N\s+([^\n]+)$", block, re.MULTILINE)
        if not xn_line:
            return None

        xn_values = re.findall(r"-?\d+", xn_line.group(1))
        if not xn_values:
            return None

        # If we can read DT row, match the value whose date == target_date
        if dt_line:
            dt_tokens = dt_line.group(1).split()
            target_day = target_date.strftime("%d")
            for tok, v in zip(dt_tokens, xn_values):
                if target_day in tok:
                    return float(v)

        # Fallback: first max value (earliest period in the bulletin)
        return float(xn_values[0])
    except Exception as e:  # noqa: BLE001
        log.debug("mos parse error: %s", e)
        return None


def _parse_cli(text: str) -> Optional[Dict[str, Any]]:
    """Parse NWS CLI product HTML / plain text.

    We look for the 'MAXIMUM' row, its observed value, and the 'VALID' date.
    """
    # Strip HTML
    plain = re.sub(r"<[^>]+>", "\n", text)
    plain = re.sub(r"&nbsp;", " ", plain)
    plain = re.sub(r"[ \t]+", " ", plain)

    # Pull the issuance date ("CLIMATE REPORT ... VALID TODAY <MONTH DAY YYYY>" etc.)
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

    # Maximum row: "MAXIMUM  67  1251 PM" – take first integer after MAXIMUM
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
        "posted_at": datetime.utcnow().isoformat(),
        "raw_text": plain[:4000],
    }


def running_max(obs_today: List[Dict[str, Any]]) -> Tuple[Optional[float],
                                                           Optional[str]]:
    """Return (max_f, timestamp) over today's observations."""
    best: Optional[Tuple[float, str]] = None
    for o in obs_today:
        t = o.get("temperature_f")
        if t is None:
            continue
        if best is None or t > best[0]:
            best = (float(t), o["observed_at"])
    return (best[0], best[1]) if best else (None, None)


def high_confirmed(obs_today: List[Dict[str, Any]]) -> bool:
    """High Confirmation Rule.

    Requires temperature to have dropped >=2°F from running max AND stayed
    below max for 2 consecutive hours AND it must be past 2:00 PM local.
    """
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

    # Check drop
    last_temp = after_max[-1][1]
    if max_f - last_temp < 2.0:
        return False

    # Check that the last 2h all stayed below max
    two_hours_ago = now - timedelta(hours=2)
    recent = [t for ts, t in after_max
              if datetime.fromisoformat(ts) >= two_hours_ago]
    if len(recent) < 2:
        return False
    return all(t < max_f for t in recent)
