"""Session 8 Part 1 — NWS official-forecast tracker tests.

Shape and parsing tests. Network calls are mocked. Async test bodies
use asyncio.run() to match the existing project pattern (see
test_data_sources_retry.py).
"""
import asyncio
import sqlite3
from unittest.mock import AsyncMock, MagicMock

from al3x.nws_official_tracker import (
    ensure_schema,
    _extract_daily_high_periods,
    fetch_and_store,
    latest_for_target,
)


def _setup_db():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    return conn


def test_schema_creates_table_and_indexes():
    conn = _setup_db()
    cols = {row[1] for row in conn.execute(
        "PRAGMA table_info(nws_official_forecasts)").fetchall()}
    assert "captured_at" in cols
    assert "target_date" in cols
    assert "forecasted_high_f" in cols
    assert "source_url" in cols
    assert "nws_issued_at" in cols
    assert "period_name" in cols


def test_schema_idempotent():
    conn = _setup_db()
    ensure_schema(conn)
    ensure_schema(conn)


def test_extract_daily_high_keeps_only_daytime_periods():
    periods = [
        {"name": "Tonight", "isDaytime": False, "temperature": 55,
         "temperatureUnit": "F", "startTime": "2026-05-06T18:00:00-04:00"},
        {"name": "Tuesday", "isDaytime": True, "temperature": 75,
         "temperatureUnit": "F", "startTime": "2026-05-07T06:00:00-04:00"},
        {"name": "Tuesday Night", "isDaytime": False, "temperature": 60,
         "temperatureUnit": "F", "startTime": "2026-05-07T18:00:00-04:00"},
        {"name": "Wednesday", "isDaytime": True, "temperature": 78,
         "temperatureUnit": "F", "startTime": "2026-05-08T06:00:00-04:00"},
    ]
    result = _extract_daily_high_periods(periods)
    assert len(result) == 2
    assert result[0][0] == "2026-05-07"
    assert result[0][1] == "Tuesday"
    assert result[0][2] == 75.0
    assert result[1][0] == "2026-05-08"
    assert result[1][1] == "Wednesday"
    assert result[1][2] == 78.0


def test_extract_skips_non_fahrenheit():
    periods = [
        {"name": "Today", "isDaytime": True, "temperature": 24,
         "temperatureUnit": "C", "startTime": "2026-05-06T06:00:00-04:00"},
    ]
    result = _extract_daily_high_periods(periods)
    assert result == []


def test_extract_handles_missing_fields():
    periods = [
        {"name": "Bad", "isDaytime": True, "temperature": None,
         "temperatureUnit": "F", "startTime": "2026-05-06T06:00:00-04:00"},
    ]
    assert _extract_daily_high_periods(periods) == []


def test_latest_for_target_returns_most_recent():
    conn = _setup_db()
    conn.execute(
        """INSERT INTO nws_official_forecasts
           (captured_at, target_date, forecasted_high_f, source_url)
           VALUES (?, ?, ?, ?)""",
        ("2026-05-06T10:00:00+00:00", "2026-05-06", 70.0, "http://x"))
    conn.execute(
        """INSERT INTO nws_official_forecasts
           (captured_at, target_date, forecasted_high_f, source_url)
           VALUES (?, ?, ?, ?)""",
        ("2026-05-06T14:00:00+00:00", "2026-05-06", 72.0, "http://x"))
    conn.commit()

    result = latest_for_target(conn, "2026-05-06")
    assert result is not None
    assert result["forecasted_high_f"] == 72.0
    assert result["captured_at"] == "2026-05-06T14:00:00+00:00"


def test_latest_for_target_returns_none_when_no_data():
    conn = _setup_db()
    assert latest_for_target(conn, "2026-05-06") is None


def test_fetch_and_store_writes_periods():
    """Mock the HTTP client to return a synthetic NWS response and
    verify periods are written to the table."""
    async def _go():
        conn = _setup_db()
        conn.execute(
            """INSERT INTO nws_gridpoint_cache (id, forecast_url, cached_at)
               VALUES (1, ?, ?)""",
            ("http://test.example.com/forecast",
             "2026-05-06T10:00:00+00:00"))
        conn.commit()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "properties": {
                "updated": "2026-05-06T15:00:00+00:00",
                "periods": [
                    {"name": "Today", "isDaytime": True, "temperature": 73,
                     "temperatureUnit": "F",
                     "startTime": "2026-05-06T06:00:00-04:00"},
                    {"name": "Tonight", "isDaytime": False, "temperature": 58,
                     "temperatureUnit": "F",
                     "startTime": "2026-05-06T18:00:00-04:00"},
                ],
            },
        })

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await fetch_and_store(conn, mock_client)

        assert result["status"] == "ok"
        assert result["rows_written"] == 1
        rows = conn.execute(
            "SELECT target_date, forecasted_high_f, period_name "
            "FROM nws_official_forecasts").fetchall()
        assert len(rows) == 1
        assert rows[0] == ("2026-05-06", 73.0, "Today")

    asyncio.run(_go())


def test_fetch_and_store_handles_fetch_failure():
    """If the gridpoint URL cannot be resolved (no cache, no points
    response), fetch_and_store returns an error result rather than
    raising."""
    async def _go():
        conn = _setup_db()
        # Mock both /points and the forecast call to fail
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("network down"))
        result = await fetch_and_store(conn, mock_client)
        assert result["status"] == "error"

    asyncio.run(_go())
