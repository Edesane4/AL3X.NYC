"""Independent NWS official-forecast tracker.

Captures the user-facing NWS forecast for Central Park's daily high
at a regular cadence, completely separate from AL3X.NYC's own forecast
cycle and from the existing 'nws_point' source in data_sources.py.

This exists specifically so we can do clean apples-to-apples
comparisons of AL3X.NYC vs weather.gov's forecast at matched
timestamps — the comparison Kalshi traders care about.

NWS API pattern (https://www.weather.gov/documentation/services-web-api):
  1. GET /points/{lat},{lon} returns metadata with a 'forecast' URL
  2. GET that forecast URL returns 'periods' — typically 3-7 days of
     12-hour forecast windows, with name (e.g. "Today", "Tonight"),
     startTime, endTime, isDaytime, temperature, temperatureUnit.
  3. For each daytime period, the temperature IS the forecasted high
     for that calendar date.

The forecast URL is location-specific and stable; we cache it for a
week to avoid hammering /points on every fetch.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import httpx

from al3x import config as cfg

logger = logging.getLogger(__name__)

_USER_AGENT = "AL3X.NYC weather agent (github.com/Edesane4/AL3X.NYC)"

_POINTS_URL = f"https://api.weather.gov/points/{cfg.LAT},{cfg.LON}"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nws_official_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    target_date TEXT NOT NULL,
    forecasted_high_f REAL NOT NULL,
    source_url TEXT NOT NULL,
    nws_issued_at TEXT,
    period_name TEXT,
    raw_response_excerpt TEXT
);
CREATE INDEX IF NOT EXISTS idx_nws_official_target
    ON nws_official_forecasts(target_date);
CREATE INDEX IF NOT EXISTS idx_nws_official_captured
    ON nws_official_forecasts(captured_at);

CREATE TABLE IF NOT EXISTS nws_gridpoint_cache (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    forecast_url TEXT NOT NULL,
    cached_at TEXT NOT NULL
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    conn.commit()


async def _resolve_gridpoint_url(conn: sqlite3.Connection,
                                 client: httpx.AsyncClient) -> Optional[str]:
    """Look up Central Park's gridpoint forecast URL. Cached for 7 days."""
    row = conn.execute(
        "SELECT forecast_url, cached_at FROM nws_gridpoint_cache WHERE id = 1"
    ).fetchone()

    if row:
        try:
            cached_at = datetime.fromisoformat(row[1])
            age_days = (datetime.now(timezone.utc) - cached_at).total_seconds() / 86400
            if age_days < 7.0:
                return row[0]
        except (ValueError, TypeError):
            pass

    try:
        r = await client.get(_POINTS_URL,
                             headers={"User-Agent": _USER_AGENT,
                                      "Accept": "application/geo+json"},
                             timeout=15.0)
        r.raise_for_status()
        data = r.json()
        url = data.get("properties", {}).get("forecast")
        if not url:
            logger.warning("nws_tracker: /points response missing forecast URL")
            return None

        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO nws_gridpoint_cache (id, forecast_url, cached_at)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                forecast_url = excluded.forecast_url,
                cached_at = excluded.cached_at
            """,
            (url, now_iso),
        )
        conn.commit()
        logger.info("nws_tracker: cached gridpoint URL %s", url)
        return url
    except Exception as e:
        logger.warning("nws_tracker: gridpoint URL lookup failed: %s", e)
        return None


def _extract_daily_high_periods(periods: list) -> list:
    """From a list of NWS forecast periods, extract daytime periods.

    Returns list of (target_date_str, period_name, temperature_f, raw_period_dict).
    """
    out = []
    for p in periods:
        if not p.get("isDaytime"):
            continue
        if p.get("temperatureUnit") != "F":
            logger.warning("nws_tracker: unexpected temperatureUnit %s",
                           p.get("temperatureUnit"))
            continue
        try:
            start = p.get("startTime", "")
            target_date = start[:10]
            temp_f = float(p.get("temperature"))
            name = p.get("name", "")
            out.append((target_date, name, temp_f, p))
        except (ValueError, TypeError) as e:
            logger.warning("nws_tracker: skip period due to parse error: %s", e)
            continue
    return out


async def fetch_and_store(conn: sqlite3.Connection,
                          client: httpx.AsyncClient) -> dict:
    """Fetch the latest NWS forecast and store all daytime periods.

    Returns a summary dict for logging/observability. Intended to be
    called every 30 minutes from the scheduler.
    """
    ensure_schema(conn)

    forecast_url = await _resolve_gridpoint_url(conn, client)
    if not forecast_url:
        return {"status": "error", "reason": "no gridpoint URL"}

    try:
        r = await client.get(forecast_url,
                             headers={"User-Agent": _USER_AGENT,
                                      "Accept": "application/geo+json"},
                             timeout=15.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("nws_tracker: forecast fetch failed: %s", e)
        return {"status": "error", "reason": str(e)}

    properties = data.get("properties", {}) or {}
    periods = properties.get("periods", []) or []
    nws_issued = properties.get("updated") or properties.get("updateTime")

    daytime = _extract_daily_high_periods(periods)
    if not daytime:
        return {"status": "error", "reason": "no daytime periods"}

    captured_at = datetime.now(timezone.utc).isoformat()
    excerpt = json.dumps(properties)[:2000]
    rows_written = 0

    for target_date, period_name, temp_f, _raw in daytime:
        conn.execute(
            """
            INSERT INTO nws_official_forecasts
                (captured_at, target_date, forecasted_high_f,
                 source_url, nws_issued_at, period_name,
                 raw_response_excerpt)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (captured_at, target_date, temp_f, forecast_url,
             nws_issued, period_name, excerpt),
        )
        rows_written += 1

    conn.commit()

    summary = {
        "status": "ok",
        "captured_at": captured_at,
        "rows_written": rows_written,
        "today_period": daytime[0] if daytime else None,
        "nws_issued_at": nws_issued,
    }
    logger.info("nws_tracker: wrote %d periods, today=%s°F",
                rows_written,
                daytime[0][2] if daytime else "n/a")
    return summary


def latest_for_target(conn: sqlite3.Connection,
                      target_date: str) -> Optional[dict]:
    """Return the most recent NWS forecast row for a given target date."""
    row = conn.execute(
        """
        SELECT captured_at, target_date, forecasted_high_f,
               nws_issued_at, period_name
          FROM nws_official_forecasts
         WHERE target_date = ?
         ORDER BY captured_at DESC
         LIMIT 1
        """,
        (target_date,),
    ).fetchone()
    if not row:
        return None
    return {
        "captured_at": row[0],
        "target_date": row[1],
        "forecasted_high_f": row[2],
        "nws_issued_at": row[3],
        "period_name": row[4],
    }
