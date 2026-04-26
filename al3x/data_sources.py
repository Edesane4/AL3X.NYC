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
        return await self._station_observations(cfg.STATION_ID)

    async def _station_observations(
            self, station_id: str) -> List[Dict[str, Any]]:
        """Fetch METAR observations for a single NWS station."""
        url = (f"https://api.weather.gov/stations/{station_id}"
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
            log.warning("asos fetch failed for %s: %s", station_id, e)
            return []

    # ---- UPGRADE B: multi-airport cross-validation ---------------------
    async def airport_observations(
            self, stations: Optional[List[str]] = None
            ) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch ASOS observations from the NYC-metro airport ring.

        Returns {station_id: [obs]} keyed by station. Fetches run in
        parallel with per-station error isolation — one failing station
        doesn't poison the rest. Stations default to KLGA/KJFK/KEWR/KTEB
        (not KNYC; that's ``asos_observations`` and goes to its own
        table for backward compatibility).
        """
        import asyncio
        stations = stations or ["KLGA", "KJFK", "KEWR", "KTEB"]
        tasks = [self._station_observations(s) for s in stations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: Dict[str, List[Dict[str, Any]]] = {}
        for st, res in zip(stations, results):
            if isinstance(res, Exception):
                log.info("airport obs %s failed: %s", st, res)
                out[st] = []
            else:
                out[st] = res
        return out

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

    # NAM-MOS intentionally removed: IEM does not archive NAM-MOS for KNYC.

    # ---- Open-Meteo: HRRR + ECMWF --------------------------------------
    async def _get_with_retry_5xx(self, url: str, params: Dict[str, Any],
                                   source_name: str,
                                   max_attempts: int = 2,
                                   backoff_seconds: float = 2.0
                                   ) -> httpx.Response:
        """GET with one retry on transient upstream failure.

        Retries only on:
          - HTTP 5xx responses (server-side, often transient)
          - httpx.ConnectError, ReadTimeout, RemoteProtocolError

        Does NOT retry on 4xx (caller bug) or parse failures (caller's job).

        Raises the final httpx exception or returns the final Response.
        Caller is responsible for raise_for_status() if it wants to treat
        non-2xx as errors.

        Added Session 4 Part 3 to fix the 2026-04-26 8:00 AM ECMWF 502
        that killed an entire intraday cycle.
        """
        import asyncio
        last_exc: Optional[Exception] = None
        last_response: Optional[httpx.Response] = None
        for attempt in range(1, max_attempts + 1):
            try:
                r = await self._client.get(url, params=params)
                if 500 <= r.status_code < 600:
                    last_response = r
                    if attempt < max_attempts:
                        log.info("%s upstream %d on attempt %d/%d, retrying "
                                 "after %.1fs backoff",
                                 source_name, r.status_code, attempt,
                                 max_attempts, backoff_seconds)
                        await asyncio.sleep(backoff_seconds)
                        continue
                    return r
                return r
            except (httpx.ConnectError, httpx.ReadTimeout,
                    httpx.RemoteProtocolError) as e:
                last_exc = e
                if attempt < max_attempts:
                    log.info("%s transient error %s on attempt %d/%d, "
                             "retrying after %.1fs backoff",
                             source_name, type(e).__name__, attempt,
                             max_attempts, backoff_seconds)
                    await asyncio.sleep(backoff_seconds)
                    continue
                raise
        if last_response is not None:
            return last_response
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{source_name}: retry loop exited without result")

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
            r = await self._get_with_retry_5xx(cfg.OPEN_METEO, params, name)
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
        # FIX 4 — live availability is 0/200 (every call returns "no
        # target-date hours"). Root cause is not yet confirmed — likely
        # model-id rename (ecmwf_ifs04 → ecmwf_ifs025 or ecmwf) or a
        # timezone/horizon regression upstream. Diagnostic script:
        # tools/diagnose_ecmwf.py. Sandbox in this session had no
        # network egress so no live capture; do not change the model
        # id speculatively. TODO: re-run the diagnostic on the host
        # that owns al3x.db and apply the minimal fix it reveals.
        return await self.open_meteo(target_date, "ecmwf_ifs04", "ecmwf",
                                     also_obs=also_obs)

    # ---- UPGRADE A: GEFS ensemble spread -------------------------------
    async def gfs_ensemble_spread(self, target_date: date) -> SourceResult:
        """Fetch all GEFS members from Open-Meteo and return the daily-
        max distribution: mean, sigma, p10/p50/p90.

        GEFS runs the GFS 30+ times with perturbed initial conditions.
        The spread of the member daily maxes IS the natively-calibrated
        forecast sigma — no historical data required. Use this as a
        cold-start uncertainty source when QRF and BMA lack training.

        Open-Meteo serves the ensemble under ``gfs_seamless`` with
        ``ensemble=True`` — each member comes back as
        ``temperature_2m_member01`` ... ``temperature_2m_memberNN``.
        """
        try:
            # Session 5 Part 1 — corrected to use the /v1/ensemble endpoint
            # and the gfs05 model name (the GEFS 0.5° grid that natively
            # serves perturbed members as temperature_2m_memberNN). The
            # previous code called /v1/forecast with `ensemble=true`, which
            # is a parameter that endpoint silently ignores; response always
            # returned just the control series. Verified against live API
            # on 2026-04-26.
            params = {
                "latitude": cfg.LAT, "longitude": cfg.LON,
                "hourly": "temperature_2m",
                "models": "gfs05",
                "temperature_unit": "fahrenheit",
                "timezone": "America/New_York",
                "forecast_days": 3,
            }
            r = await self._client.get(cfg.OPEN_METEO_ENSEMBLE, params=params,
                                       timeout=25.0)
            r.raise_for_status()
            js = r.json()
            hourly = js.get("hourly", {}) or {}
            times = hourly.get("time", []) or []
            if not times:
                return SourceResult("gfs_ensemble", error="empty hourly")

            # Index hours that fall on the target date
            idx_today: List[int] = []
            for i, ts in enumerate(times):
                try:
                    dt = _parse_dt(ts)
                except Exception:
                    continue
                if dt.date() == target_date:
                    idx_today.append(i)
            if not idx_today:
                return SourceResult("gfs_ensemble",
                                    error="no target-date hours")

            # FIX 3 — explicitly separate the control member
            # (``temperature_2m``) from the perturbed-member series
            # (``temperature_2m_memberNN``). The prior ``startswith``
            # collapsed both keys into a single group and then failed
            # to correctly collect them, yielding "only 1 members
            # parsed" on every live call.
            member_maxes = _parse_gfs_ensemble_members(hourly, idx_today)

            if len(member_maxes) < 3:
                return SourceResult(
                    "gfs_ensemble",
                    error=f"only {len(member_maxes)} members parsed",
                )

            member_maxes.sort()
            n = len(member_maxes)
            mean = sum(member_maxes) / n
            # Population std-dev of the ensemble (this IS the forecast sigma)
            var = sum((x - mean) ** 2 for x in member_maxes) / n
            sigma = var ** 0.5
            p10 = member_maxes[max(0, int(0.10 * (n - 1)))]
            p50 = member_maxes[n // 2]
            p90 = member_maxes[min(n - 1, int(0.90 * (n - 1)))]
            return SourceResult(
                "gfs_ensemble",
                value=mean,
                meta={
                    "sigma_f": sigma,
                    "p10": p10,
                    "p50": p50,
                    "p90": p90,
                    "n_members": n,
                    "members": member_maxes,
                },
            )
        except Exception as e:  # noqa: BLE001
            log.info("gfs_ensemble fetch failed: %s", e)
            return SourceResult("gfs_ensemble", error=str(e))

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

_GEFS_MEMBER_KEY_RE = re.compile(r"^temperature_2m_member\d+$")


def _parse_gfs_ensemble_members(hourly: Dict[str, Any],
                                 idx_today: List[int]) -> List[float]:
    """Collect per-member daily-max °F from an Open-Meteo ensemble payload.

    Open-Meteo's ensemble response returns the control run as
    ``temperature_2m`` and each perturbed member as
    ``temperature_2m_memberNN`` (NN is typically 01..30). The prior
    ``startswith("temperature_2m")`` approach grouped both patterns
    together and failed to collect them correctly, leaving only one
    member — the diagnostic showed 0/200 successful live calls.

    Strategy: grab the control explicitly, iterate the perturbed
    members via a regex that excludes the bare ``temperature_2m`` key,
    and dedupe by rounded max value so the control run isn't
    double-counted if a feed also serves it as ``temperature_2m_member00``.
    """
    member_maxes: List[float] = []

    def _max_on_target_date(series: Any) -> Optional[float]:
        if not isinstance(series, list):
            return None
        vals = [series[i] for i in idx_today
                if i < len(series) and series[i] is not None]
        if not vals:
            return None
        return float(max(vals))

    control = _max_on_target_date(hourly.get("temperature_2m"))
    if control is not None:
        member_maxes.append(control)

    for key, series in hourly.items():
        if not _GEFS_MEMBER_KEY_RE.match(key):
            continue
        m = _max_on_target_date(series)
        if m is not None:
            member_maxes.append(m)

    # Deduplicate by rounded max (handles feeds that serve the control
    # run as both ``temperature_2m`` and ``temperature_2m_member00``).
    seen_rounded: set = set()
    unique: List[float] = []
    for v in member_maxes:
        key = round(v, 2)
        if key in seen_rounded:
            continue
        seen_rounded.add(key)
        unique.append(v)
    return unique


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


def _parse_mos_csv(text: str, target_date: date) -> Optional[float]:
    """Parse IEM's CSV MOS response.

    Format (per https://mesonet.agron.iastate.edu/api/ ):
      station,model,runtime,ftime,n_x,tmp,dpt,cld,wdr,wsp, ...
    where `ftime` is an ISO UTC timestamp and `tmp` is the hourly forecast
    temperature in °F. The `n_x` column is the X/N field but is sparsely
    populated (sometimes missing the first X for the in-progress day), so
    we bypass it and compute max(tmp) directly over the target-date's
    local daytime window (6 AM – 11 PM Eastern).
    """
    try:
        import csv, io
        reader = csv.DictReader(io.StringIO(text))
        # Target window in UTC: 6 AM local = 10/11 UTC (depending on DST);
        # use a generous 09–04 UTC window to cover both EST and EDT. The
        # far side of that window rolls into the next UTC date, so we
        # match against the *local* date of ftime.
        candidates: List[float] = []
        for row in reader:
            ft = (row.get("ftime") or "").strip()
            tmp_s = (row.get("tmp") or "").strip()
            if not ft or not tmp_s:
                continue
            # ftime is space-separated UTC timestamp "YYYY-MM-DD HH:MM"
            try:
                dt_utc = datetime.fromisoformat(ft).replace(
                    tzinfo=timezone.utc)
            except Exception:
                continue
            dt_local = dt_utc.astimezone(cfg.EASTERN)
            if dt_local.date() != target_date:
                continue
            # Daytime window only
            if dt_local.hour < 6 or dt_local.hour > 22:
                continue
            try:
                candidates.append(float(tmp_s))
            except ValueError:
                continue
        if not candidates:
            return None
        return float(max(candidates))
    except Exception as e:  # noqa: BLE001
        log.debug("mos csv parse error: %s", e)
        return None


def _parse_mos_max(text: str, target_date: date) -> Optional[float]:
    """Parse a MOS bulletin (either IEM CSV or raw MAV text) for the
    daily max at target_date (local Eastern calendar day).

    Tries CSV first (IEM's format since ~2020), falls back to MAV text
    parsing for any source that still serves the classic NWS bulletin.
    """
    # Detect IEM CSV by the first non-blank line being a header
    head = ""
    for ln in text.splitlines()[:3]:
        s = ln.strip()
        if s:
            head = s.lower()
            break
    if head.startswith("station,model,runtime,ftime"):
        csv_result = _parse_mos_csv(text, target_date)
        if csv_result is not None:
            return csv_result
        # If CSV parse produced nothing, fall through to MAV attempt

    # --- Fallback: legacy MAV/MET text bulletin parser ---
    try:
        block = _extract_knyc_block(text) or text
        lines = block.splitlines()

        # Find the header and parse run time (UTC)
        run_hour_utc: Optional[int] = None
        run_date_utc: Optional[date] = None
        for ln in lines[:3]:
            m = re.search(
                r"(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{3,4})\s*UTC",
                ln,
            )
            if m:
                run_date_utc = date(int(m.group(3)),
                                     int(m.group(1)), int(m.group(2)))
                run_hour_utc = int(m.group(4)) // 100
                break

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

        # Collect every numeric on the X/N line in order
        xn_body = xn_line[len(xn_line) - len(xn_line.lstrip()) + len("X/N"):]
        xn_vals_in_order: List[int] = []
        for m in re.finditer(r"-?\d+", xn_body):
            try:
                xn_vals_in_order.append(int(m.group(0)))
            except ValueError:
                continue

        if not xn_vals_in_order:
            return None

        # Pair consecutive values and take max per pair → sequence of X values
        x_values: List[int] = []
        it = iter(xn_vals_in_order)
        for a in it:
            b = next(it, None)
            if b is None:
                x_values.append(a)
            else:
                x_values.append(max(a, b))

        # Compute first_x_local_date from run time. Fall back to "today
        # local Eastern" if the header couldn't be parsed (legacy tests).
        if run_hour_utc is not None and run_date_utc is not None:
            # Use 12Z as a heuristic local-date anchor: for EST (UTC-5) the
            # run is still on the same local day. For EDT (UTC-4) same.
            local_run_date = run_date_utc
            if run_hour_utc < 4:
                # Late-night UTC = previous local day
                local_run_date = run_date_utc - timedelta(days=1)
            if run_hour_utc in (0, 6):
                first_x_local_date = local_run_date + timedelta(days=1)
            else:  # 12Z, 18Z
                first_x_local_date = local_run_date
        else:
            first_x_local_date = datetime.now(cfg.EASTERN).date()

        days_ahead = (target_date - first_x_local_date).days
        if 0 <= days_ahead < len(x_values):
            return float(x_values[days_ahead])

        # Returns None if target date is outside the bulletin range.
        log.debug("mos target %s out of bulletin range "
                  "(first_x=%s, have %d X values)",
                  target_date, first_x_local_date, len(x_values))
        return None
    except Exception as e:  # noqa: BLE001
        log.debug("mos parse error: %s", e)
        return None


def _parse_cli(text: str) -> Optional[Dict[str, Any]]:
    """Parse an NWS CLI product into {target_date, recorded_high_f, ...}.

    FIX 1 — anchor the MAXIMUM regex to the observed-temperature section.
    CLI products contain multiple sections with a "MAXIMUM" token
    (observed, record, normal, last-year). The naive regex
    ``MAXIMUM\\s+(\\d+)`` matches whichever appears first — on current
    templates that's typically the climatological normal, which had been
    silently poisoning every scored forecast.
    """
    plain = re.sub(r"<[^>]+>", "\n", text)
    plain = re.sub(r"&nbsp;", " ", plain)
    plain = re.sub(r"[ \t]+", " ", plain)

    # Real NWS CLI products state the covered date in their subject line:
    #   "...THE CENTRAL PARK NY CLIMATE SUMMARY FOR APRIL 14 2026..."
    # Older templates used "VALID TODAY <MONTH DAY YYYY>". Try both.
    target_dt: Optional[date] = None
    months = (r"(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|"
              r"SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(\d{1,2})\s+(\d{4})")
    for pat in (
        rf"CLIMATE SUMMARY FOR\s+{months}",
        rf"CLIMATE REPORT.*?VALID[^\n]*?{months}",
    ):
        m = re.search(pat, plain, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                target_dt = datetime.strptime(
                    f"{m.group(1)} {m.group(2)} {m.group(3)}",
                    "%B %d %Y",
                ).date()
                break
            except Exception:
                pass

    recorded = _extract_observed_max(plain)
    if recorded is None:
        return None
    if not (-50.0 <= recorded <= 130.0):
        log.error("CLI observed max %s out of sane °F range; rejecting",
                  recorded)
        return None

    return {
        "target_date": target_dt.isoformat() if target_dt else None,
        "recorded_high_f": recorded,
        "posted_at": datetime.now(timezone.utc).isoformat(),
        "raw_text": plain[:4000],
    }


# Major CLI section headers used to bound the observed-temperature block.
_CLI_NEXT_SECTIONS = (
    "PRECIPITATION (IN)", "PRECIPITATION",
    "SNOWFALL (IN)", "SNOWFALL",
    "DEGREE DAYS",
    "WIND (MPH)", "WIND",
    "SKY COVER",
    "WEATHER CONDITIONS",
    "RELATIVE HUMIDITY",
    "THE FOLLOWING",
)


def _extract_observed_max(plain: str) -> Optional[float]:
    """Return the observed daily-high °F from a CLI plain-text product.

    Strategy:
      1. Find the TEMPERATURE section header (``TEMPERATURE`` optionally
         followed by ``(F)`` / ``(^F)``).
      2. Slice up to the next major section header.
      3. Inside that slice, find a ``MAXIMUM`` line that is NOT
         ``RECORD MAXIMUM``, ``NORMAL MAXIMUM``, or ``MAXIMUM
         TEMPERATURE LAST YEAR``. First integer on that line is the
         observed high.
      4. If the section parse fails, use a defensive global regex that
         explicitly excludes lines preceded by RECORD / NORMAL / LAST
         YEAR within 30 characters. Log a warning when this fires so
         operators notice template drift.
    """
    section = _observed_section(plain)
    if section is not None:
        val = _first_observed_max_in_section(section)
        if val is not None:
            return val
        log.warning("CLI observed-section parsed but no MAXIMUM line found; "
                    "falling back to defensive global regex")
    else:
        log.warning("CLI TEMPERATURE section not found; "
                    "falling back to defensive global regex")

    # Fallback: global regex that excludes RECORD / NORMAL / LAST YEAR
    # within 30 characters preceding the MAXIMUM token.
    for m in re.finditer(r"\bMAXIMUM\s+(-?\d+)\b", plain, re.IGNORECASE):
        start = m.start()
        window = plain[max(0, start - 30):start].upper()
        if "RECORD" in window or "NORMAL" in window or "LAST YEAR" in window:
            continue
        try:
            return float(m.group(1))
        except ValueError:
            continue
    return None


def _observed_section(plain: str) -> Optional[str]:
    """Slice the observed-temperature block out of a CLI product."""
    # Bug C fix — require (F) or (^F) suffix so we match only the
    # observed-temperature section, not prior-records headers that
    # share the TEMPERATURE prefix.
    header = re.search(
        r"(?im)^[ \t]*TEMPERATURE[ \t]*\((?:\^?F)\)[^\n]*$",
        plain,
    )
    if header is None:
        return None
    start = header.end()
    # Earliest next-section header after the TEMPERATURE header.
    # Compare as uppercase so the literal matching is case-insensitive.
    tail = plain[start:]
    tail_upper = tail.upper()
    end = len(tail)
    for marker in _CLI_NEXT_SECTIONS:
        idx = tail_upper.find(marker)
        if 0 <= idx < end:
            end = idx
    return tail[:end]


def _first_observed_max_in_section(section: str) -> Optional[float]:
    """Inside the TEMPERATURE section, find the observed MAXIMUM value."""
    for line in section.splitlines():
        stripped = line.strip()
        up = stripped.upper()
        if not up.startswith("MAXIMUM"):
            continue
        # Exclude qualified lines that are NOT the observed max.
        if ("RECORD" in up
                or "NORMAL" in up
                or up.startswith("MAXIMUM TEMPERATURE LAST YEAR")
                or "LAST YEAR" in up):
            continue
        m = re.search(r"(-?\d+)", stripped)
        if not m:
            continue
        try:
            return float(m.group(1))
        except ValueError:
            continue
    return None


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
