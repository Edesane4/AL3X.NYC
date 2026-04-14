"""SQLite persistence for forecasts, observations, CLI truth, and bias ledger."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issued_at TEXT NOT NULL,            -- ISO8601 Eastern
    target_date TEXT NOT NULL,          -- YYYY-MM-DD
    mode TEXT NOT NULL,                 -- 'night_before' | 'intraday'
    revision INTEGER NOT NULL DEFAULT 0,
    final_f REAL NOT NULL,
    raw_ensemble_f REAL NOT NULL,
    uncertainty_f REAL NOT NULL,
    running_asos_max_f REAL,
    sources_json TEXT NOT NULL,         -- {source: {value, weight}}
    corrections_json TEXT NOT NULL,     -- {name: {delta, reason}}
    delta_prior_f REAL,
    extras_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_forecasts_target ON forecasts(target_date);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at TEXT NOT NULL,
    temperature_f REAL,
    wind_dir_deg REAL,
    wind_speed_kt REAL,
    dewpoint_f REAL,
    sky_cover_pct REAL,
    raw_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_time ON observations(observed_at);

CREATE TABLE IF NOT EXISTS cli_truth (
    target_date TEXT PRIMARY KEY,
    recorded_high_f REAL NOT NULL,
    posted_at TEXT NOT NULL,
    raw_text TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_date TEXT NOT NULL,
    forecast_id INTEGER NOT NULL,
    mode TEXT NOT NULL,
    error_f REAL NOT NULL,               -- CLI - forecast
    abs_error_f REAL NOT NULL,
    lead_hours REAL NOT NULL,
    regime TEXT,
    FOREIGN KEY (forecast_id) REFERENCES forecasts(id)
);
CREATE INDEX IF NOT EXISTS idx_scores_target ON scores(target_date);

CREATE TABLE IF NOT EXISTS bias_state (
    key TEXT PRIMARY KEY,
    value REAL NOT NULL,
    updated_at TEXT NOT NULL,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS source_weights (
    key TEXT PRIMARY KEY,
    value REAL NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_date TEXT NOT NULL,
    forecast_id INTEGER,
    error_f REAL NOT NULL,
    root_cause TEXT,
    detail_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS log_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    source TEXT,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_log_ts ON log_events(ts);
"""


class Storage:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=15)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)

    # -- forecasts ---------------------------------------------------------
    def save_forecast(self, row: Dict[str, Any]) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO forecasts
                   (issued_at, target_date, mode, revision, final_f, raw_ensemble_f,
                    uncertainty_f, running_asos_max_f, sources_json, corrections_json,
                    delta_prior_f, extras_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row["issued_at"],
                    row["target_date"],
                    row["mode"],
                    row.get("revision", 0),
                    row["final_f"],
                    row["raw_ensemble_f"],
                    row["uncertainty_f"],
                    row.get("running_asos_max_f"),
                    json.dumps(row["sources"]),
                    json.dumps(row["corrections"]),
                    row.get("delta_prior_f"),
                    json.dumps(row.get("extras", {})),
                ),
            )
            return cur.lastrowid

    def latest_forecast(self, target_date: Optional[str] = None) -> Optional[Dict]:
        with self._conn() as c:
            if target_date:
                r = c.execute(
                    "SELECT * FROM forecasts WHERE target_date=? "
                    "ORDER BY id DESC LIMIT 1",
                    (target_date,),
                ).fetchone()
            else:
                r = c.execute(
                    "SELECT * FROM forecasts ORDER BY id DESC LIMIT 1"
                ).fetchone()
            return _forecast_row(r)

    def recent_forecasts(self, limit: int = 50) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM forecasts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [_forecast_row(r) for r in rows]

    def forecasts_for_date(self, target_date: str) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM forecasts WHERE target_date=? ORDER BY id ASC",
                (target_date,),
            ).fetchall()
            return [_forecast_row(r) for r in rows]

    def last_revision(self, target_date: str, mode: str) -> int:
        with self._conn() as c:
            r = c.execute(
                "SELECT MAX(revision) AS m FROM forecasts "
                "WHERE target_date=? AND mode=?",
                (target_date, mode),
            ).fetchone()
            return int(r["m"] or 0)

    # -- observations ------------------------------------------------------
    def save_observation(self, row: Dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO observations
                   (observed_at, temperature_f, wind_dir_deg, wind_speed_kt,
                    dewpoint_f, sky_cover_pct, raw_json)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    row["observed_at"],
                    row.get("temperature_f"),
                    row.get("wind_dir_deg"),
                    row.get("wind_speed_kt"),
                    row.get("dewpoint_f"),
                    row.get("sky_cover_pct"),
                    json.dumps(row.get("raw", {})),
                ),
            )

    def observations_today(self, date_str: str) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT observed_at, temperature_f FROM observations "
                "WHERE substr(observed_at,1,10)=? ORDER BY observed_at ASC",
                (date_str,),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- CLI truth ---------------------------------------------------------
    def save_cli_truth(self, target_date: str, high_f: float,
                       posted_at: str, raw_text: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO cli_truth
                   (target_date, recorded_high_f, posted_at, raw_text)
                   VALUES (?,?,?,?)""",
                (target_date, high_f, posted_at, raw_text),
            )

    def get_cli_truth(self, target_date: str) -> Optional[Dict]:
        with self._conn() as c:
            r = c.execute(
                "SELECT * FROM cli_truth WHERE target_date=?", (target_date,)
            ).fetchone()
            return dict(r) if r else None

    # -- scores ------------------------------------------------------------
    def save_score(self, row: Dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO scores
                   (target_date, forecast_id, mode, error_f, abs_error_f,
                    lead_hours, regime)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    row["target_date"],
                    row["forecast_id"],
                    row["mode"],
                    row["error_f"],
                    row["abs_error_f"],
                    row["lead_hours"],
                    row.get("regime"),
                ),
            )

    def recent_scores(self, days: int = 30) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM scores WHERE target_date >= date('now', ?) "
                "ORDER BY target_date DESC",
                (f"-{days} days",),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- bias / weights state ---------------------------------------------
    def set_bias(self, key: str, value: float, reason: str = "") -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO bias_state
                   (key, value, updated_at, reason) VALUES (?,?,?,?)""",
                (key, value, datetime.utcnow().isoformat(), reason),
            )

    def get_biases(self) -> Dict[str, float]:
        with self._conn() as c:
            rows = c.execute("SELECT key,value FROM bias_state").fetchall()
            return {r["key"]: r["value"] for r in rows}

    def set_weight(self, key: str, value: float) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO source_weights
                   (key,value,updated_at) VALUES (?,?,?)""",
                (key, value, datetime.utcnow().isoformat()),
            )

    def get_weights(self) -> Dict[str, float]:
        with self._conn() as c:
            rows = c.execute("SELECT key,value FROM source_weights").fetchall()
            return {r["key"]: r["value"] for r in rows}

    # -- logs & anomalies --------------------------------------------------
    def log_event(self, level: str, source: str, message: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO log_events (ts, level, source, message) VALUES (?,?,?,?)",
                (datetime.utcnow().isoformat(), level, source, message[:4000]),
            )

    def recent_logs(self, limit: int = 100) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT ts, level, source, message FROM log_events "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def save_anomaly(self, row: Dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO anomalies
                   (target_date, forecast_id, error_f, root_cause,
                    detail_json, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    row["target_date"],
                    row.get("forecast_id"),
                    row["error_f"],
                    row.get("root_cause"),
                    json.dumps(row.get("detail", {})),
                    datetime.utcnow().isoformat(),
                ),
            )


def _forecast_row(r: Optional[sqlite3.Row]) -> Optional[Dict]:
    if r is None:
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
