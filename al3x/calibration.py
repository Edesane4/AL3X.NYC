"""Calibration recorder.

For each verified CLI day, records whether realized truth fell inside
the agent's stated 80% band. This is the data feed for the Session 7
calibration display on the dashboard.

Schema:
  calibration_records (target_date UNIQUE, realized_high_f, forecast_id,
                       forecast_p10_f, forecast_p50_f, forecast_p90_f,
                       forecast_sigma_f, inside_80, error_signed_f,
                       recorded_at)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS calibration_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_date TEXT NOT NULL UNIQUE,
    realized_high_f REAL NOT NULL,
    forecast_id INTEGER,
    forecast_p50_f REAL,
    forecast_p10_f REAL,
    forecast_p90_f REAL,
    forecast_sigma_f REAL,
    inside_80 INTEGER,
    error_signed_f REAL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calibration_target_date
    ON calibration_records(target_date);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    conn.commit()


def _last_forecast_for_date(conn: sqlite3.Connection, target_date: str
                            ) -> Optional[tuple]:
    """Return (id, extras_json, top_final_f, top_uncertainty_f) for the
    last forecast issued before midnight on target_date.

    Live forecasts schema stores final_f / uncertainty_f as top-level
    columns; the simplified test schema may stuff them into extras_json
    instead. We probe pragma to handle both shapes.

    Returns None if no forecast exists for that day.
    """
    cols = {row[1] for row in conn.execute(
        "PRAGMA table_info(forecasts)").fetchall()}
    final_expr = "final_f" if "final_f" in cols else "NULL AS final_f"
    unc_expr = ("uncertainty_f" if "uncertainty_f" in cols
                else "NULL AS uncertainty_f")
    row = conn.execute(
        f"""
        SELECT id, extras_json, {final_expr}, {unc_expr}
          FROM forecasts
         WHERE target_date = ?
           AND issued_at < ? || 'T23:59:59'
         ORDER BY id DESC
         LIMIT 1
        """,
        (target_date, target_date),
    ).fetchone()
    return row


def record_calibration(conn: sqlite3.Connection, target_date: str,
                       realized_high_f: float) -> Optional[dict]:
    """Record a calibration row for a verified day.

    Idempotent — if a row already exists for target_date it is replaced
    (CLI corrections do happen and we want them reflected).

    Returns the inserted row as a dict, or None if no forecast was found
    for the day (in which case we silently skip — there's nothing to
    calibrate against).
    """
    ensure_schema(conn)

    fc = _last_forecast_for_date(conn, target_date)
    if fc is None:
        logger.info("calibration: no forecast for %s, skipping", target_date)
        return None
    forecast_id, extras_raw, top_final_f, top_uncertainty_f = fc

    try:
        extras = json.loads(extras_raw) if extras_raw else {}
    except json.JSONDecodeError:
        extras = {}

    bands = extras.get("bands") or {}
    p10 = bands.get("p10")
    p50 = bands.get("p50")
    p90 = bands.get("p90")
    sigma = bands.get("sigma_f")

    # Backfill path: forecasts pre-Session-6 don't have bands, but they
    # do have uncertainty_f. Synthesize a Gaussian band so the
    # calibration trajectory has historical data. Prefer extras-embedded
    # values (test fixtures / future shapes); fall back to top-level
    # columns (live prod schema).
    if p10 is None or p90 is None:
        u = extras.get("uncertainty_f")
        if u is None:
            u = top_uncertainty_f
        f = extras.get("final_f")
        if f is None:
            f = top_final_f
        if u is not None and f is not None:
            p50 = float(f)
            p10 = float(f) - 1.2816 * float(u)
            p90 = float(f) + 1.2816 * float(u)
            sigma = float(u)

    inside_80 = None
    error_signed = None
    if p10 is not None and p90 is not None and p50 is not None:
        inside_80 = 1 if (p10 <= realized_high_f <= p90) else 0
        error_signed = realized_high_f - p50

    now_iso = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT INTO calibration_records
            (target_date, realized_high_f, forecast_id,
             forecast_p50_f, forecast_p10_f, forecast_p90_f,
             forecast_sigma_f, inside_80, error_signed_f,
             recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(target_date) DO UPDATE SET
            realized_high_f = excluded.realized_high_f,
            forecast_id = excluded.forecast_id,
            forecast_p50_f = excluded.forecast_p50_f,
            forecast_p10_f = excluded.forecast_p10_f,
            forecast_p90_f = excluded.forecast_p90_f,
            forecast_sigma_f = excluded.forecast_sigma_f,
            inside_80 = excluded.inside_80,
            error_signed_f = excluded.error_signed_f,
            recorded_at = excluded.recorded_at
        """,
        (target_date, realized_high_f, forecast_id,
         p50, p10, p90, sigma, inside_80, error_signed, now_iso),
    )
    conn.commit()

    return {
        "target_date": target_date,
        "realized_high_f": realized_high_f,
        "p10": p10, "p50": p50, "p90": p90,
        "sigma_f": sigma,
        "inside_80": inside_80,
        "error_signed_f": error_signed,
    }
