"""SQLite persistence for forecasts, observations, CLI truth, and bias ledger."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


def _utc_now_iso() -> str:
    """Return current UTC time as a timezone-aware ISO-8601 string.

    Replaces deprecated ``datetime.utcnow()`` (naive) with an aware
    timestamp like ``2026-04-14T18:30:00.123+00:00``. Old naive rows
    stored previously in SQLite remain readable: ``fromisoformat()``
    parses both naive and aware strings since Python 3.11.
    """
    return datetime.now(timezone.utc).isoformat()


log = logging.getLogger("al3x.storage")

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
CREATE UNIQUE INDEX IF NOT EXISTS ux_obs_observed_at ON observations(observed_at);

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
CREATE UNIQUE INDEX IF NOT EXISTS ux_scores_fc_mode ON scores(forecast_id, mode);

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

-- Correction attribution (Superior Quality 3): per-day, per-correction ledger
-- of whether the correction moved the forecast toward truth or away from it.
CREATE TABLE IF NOT EXISTS correction_attribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_date TEXT NOT NULL,
    correction_name TEXT NOT NULL,
    delta_applied_f REAL NOT NULL,
    regime_label TEXT,
    error_f REAL NOT NULL,
    was_helpful INTEGER NOT NULL,        -- 1 helped, -1 hurt, 0 neutral
    forecast_id INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attr_date ON correction_attribution(target_date);
CREATE INDEX IF NOT EXISTS idx_attr_name ON correction_attribution(correction_name);

-- QRF predictions (Quant Upgrade 1): p10/p50/p90 per forecast.
CREATE TABLE IF NOT EXISTS qrf_predictions (
    forecast_id INTEGER PRIMARY KEY,
    p10_delta REAL,
    p50_delta REAL,
    p90_delta REAL,
    interval_width REAL,
    n_training INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (forecast_id) REFERENCES forecasts(id)
);

-- Regime-shift ledger (Quant Upgrade 3).
CREATE TABLE IF NOT EXISTS regime_shifts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT NOT NULL,
    target_date TEXT NOT NULL,
    shift_type TEXT NOT NULL,
    prev_state TEXT,
    new_state TEXT,
    weight_reset_applied INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_shift_date ON regime_shifts(target_date);
"""


class Storage:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._init_schema()
        self._migrate()

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

    def _migrate(self) -> None:
        """Schema migrations for existing databases.

        BUG 3 — the UNIQUE index on observations.observed_at is declared in
        _SCHEMA so new DBs get it for free. For pre-existing databases that
        may already have duplicates, this method dedupes first then retries
        the index creation.
        """
        with self._conn() as c:
            try:
                c.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_obs_observed_at "
                    "ON observations(observed_at)"
                )
            except sqlite3.IntegrityError:
                log.warning("observations has duplicate observed_at rows; "
                            "deduplicating before enforcing UNIQUE index")
                c.execute(
                    "DELETE FROM observations WHERE id NOT IN ("
                    " SELECT MIN(id) FROM observations GROUP BY observed_at"
                    ")"
                )
                c.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_obs_observed_at "
                    "ON observations(observed_at)"
                )
            except sqlite3.OperationalError:
                # observations table doesn't exist yet (e.g. in-memory DB
                # race). _init_schema will handle it on a real file DB.
                pass

            # BUG 1 migration — split any legacy "intraday:{source}" weights
            # into three unique prefixes (id06/id612/id1224) if the new keys
            # don't already exist. Safe to run repeatedly.
            try:
                rows = c.execute(
                    "SELECT key, value FROM source_weights "
                    "WHERE key LIKE 'intraday:%'"
                ).fetchall()
                for r in rows:
                    src = r["key"].split(":", 1)[1]
                    for new_prefix in ("id06", "id612", "id1224"):
                        new_key = f"{new_prefix}:{src}"
                        exists = c.execute(
                            "SELECT 1 FROM source_weights WHERE key=?",
                            (new_key,),
                        ).fetchone()
                        if not exists:
                            c.execute(
                                "INSERT INTO source_weights "
                                "(key, value, updated_at) VALUES (?,?,?)",
                                (new_key, float(r["value"]),
                                 _utc_now_iso()),
                            )
                # Also promote legacy "night_before:{src}" → "nb:{src}"
                rows = c.execute(
                    "SELECT key, value FROM source_weights "
                    "WHERE key LIKE 'night_before:%'"
                ).fetchall()
                for r in rows:
                    src = r["key"].split(":", 1)[1]
                    new_key = f"nb:{src}"
                    exists = c.execute(
                        "SELECT 1 FROM source_weights WHERE key=?",
                        (new_key,),
                    ).fetchone()
                    if not exists:
                        c.execute(
                            "INSERT INTO source_weights "
                            "(key, value, updated_at) VALUES (?,?,?)",
                            (new_key, float(r["value"]), _utc_now_iso()),
                        )
            except sqlite3.OperationalError:
                pass

            # BUGS 1+3 — dedupe scores + enforce UNIQUE(forecast_id, mode)
            # so restarts can't re-score an already-scored forecast.
            try:
                c.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_scores_fc_mode "
                    "ON scores(forecast_id, mode)"
                )
            except sqlite3.IntegrityError:
                log.warning("scores has duplicate (forecast_id, mode) rows; "
                            "deduplicating before enforcing UNIQUE index")
                c.execute(
                    "DELETE FROM scores WHERE id NOT IN ("
                    " SELECT MIN(id) FROM scores GROUP BY forecast_id, mode"
                    ")"
                )
                try:
                    c.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS ux_scores_fc_mode "
                        "ON scores(forecast_id, mode)"
                    )
                except sqlite3.IntegrityError:
                    pass
            except sqlite3.OperationalError:
                pass

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
        """BUG 3 — INSERT OR IGNORE prevents duplicate observed_at rows from
        corrupting the running-max calculation."""
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO observations
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
        """BUG 4 — return the full row so downstream consumers
        (inversion detection, Kalman filter, diurnal fit) can use
        wind_dir_deg, wind_speed_kt, dewpoint_f, sky_cover_pct."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM observations "
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

    def get_cli_truth_recent(self, days: int = 2) -> List[Dict]:
        """Return CLI truth rows from the last ``days`` days. Used at
        boot to pre-populate the scheduler's in-memory
        ``_cli_verified_for`` set so that restarts don't re-score."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT target_date FROM cli_truth "
                "WHERE target_date >= date('now', ?)",
                (f"-{days} days",),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- scores ------------------------------------------------------------
    def save_score(self, row: Dict[str, Any]) -> None:
        """BUGS 1+3 — INSERT OR IGNORE skips re-scoring on double-verify
        (e.g. restarts that lose the in-memory _cli_verified_for guard).
        """
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO scores
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
        # BUG 4 — target_date is always a bare YYYY-MM-DD string, so
        # comparing against date('now', ...) is already correct here;
        # leave as-is but ensure consistency with the other methods.
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM scores "
                "WHERE substr(target_date, 1, 10) >= date('now', ?) "
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
                (key, value, _utc_now_iso(), reason),
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
                (key, value, _utc_now_iso()),
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
                (_utc_now_iso(), level, source, message[:4000]),
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
                    _utc_now_iso(),
                ),
            )

    # -- correction attribution (Superior 3) -----------------------------
    def save_attribution(self, row: Dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO correction_attribution
                   (target_date, correction_name, delta_applied_f, regime_label,
                    error_f, was_helpful, forecast_id, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    row["target_date"],
                    row["correction_name"],
                    row["delta_applied_f"],
                    row.get("regime_label"),
                    row["error_f"],
                    int(row["was_helpful"]),
                    row.get("forecast_id"),
                    _utc_now_iso(),
                ),
            )

    # -- helpers for BMA / climatology / analog engine -------------------
    def get_source_history(self, source_name: str,
                            days: int = 30) -> List[Dict]:
        """Return (target_date, predicted_f, cli_f) for one source across
        the last `days` verified days.

        Joins forecasts.sources_json with cli_truth. Only returns rows
        where we have both a predicted value and a CLI truth.
        """
        rows: List[Dict] = []
        with self._conn() as c:
            r = c.execute(
                "SELECT f.target_date AS target_date, "
                "       f.mode AS mode, "
                "       f.sources_json AS sources_json, "
                "       t.recorded_high_f AS cli_f "
                "FROM forecasts f "
                "JOIN cli_truth t ON t.target_date = f.target_date "
                "WHERE f.target_date >= date('now', ?) "
                "ORDER BY f.target_date DESC, f.id DESC",
                (f"-{days} days",),
            ).fetchall()
        for row in r:
            try:
                srcs = json.loads(row["sources_json"] or "{}")
            except Exception:
                continue
            payload = srcs.get(source_name) or {}
            v = payload.get("value")
            if v is None:
                continue
            rows.append({
                "target_date": row["target_date"],
                "mode": row["mode"],
                "predicted_f": float(v),
                "cli_f": float(row["cli_f"]),
            })
        # BUG 6 — one row per target_date for honest bias/variance.
        # Prefer the intraday row if both modes exist for a date.
        seen: Dict[str, Dict] = {}
        for row in rows:
            d = row["target_date"]
            if d not in seen or row["mode"] == "intraday":
                seen[d] = row
        return list(seen.values())

    def observations_all_days(self, limit_days: int = 365) -> List[Dict]:
        """Return every observation from the last N days (no date filter
        beyond the cutoff). Used by RateClimatology.

        BUG 4 — compare the first 10 chars (YYYY-MM-DD) of observed_at
        against date('now', ...). The stored string is a tz-aware ISO
        timestamp like 2026-04-14T14:30:00-04:00, which SQLite's
        datetime() can't compare correctly against its UTC clock.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM observations "
                "WHERE substr(observed_at, 1, 10) >= date('now', ?) "
                "ORDER BY observed_at ASC",
                (f"-{limit_days} days",),
            ).fetchall()
            return [dict(r) for r in rows]

    def forecasts_with_truth(self, days: int = 365,
                              prefer_mode: str = "night_before") -> List[Dict]:
        """Return forecast rows joined with their CLI truth, for the
        analog engine.

        BUG 5 — for each target_date, prefer the row whose mode matches
        ``prefer_mode`` (default 'night_before'). This avoids training the
        analog library on end-of-day "locked" intraday forecasts (issued
        minutes before CLI posts when running_max has already pinned the
        forecast close to truth), which understated real forecast errors.
        If no matching-mode row exists for a date, fall back to the
        latest row for that date.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT f.*, t.recorded_high_f AS cli_f "
                "FROM forecasts f "
                "JOIN cli_truth t ON t.target_date = f.target_date "
                "WHERE f.target_date >= date('now', ?) "
                "ORDER BY f.target_date DESC, f.id DESC",
                (f"-{days} days",),
            ).fetchall()

        # Group all rows per target_date
        by_date: Dict[str, List[Dict]] = {}
        for r in rows:
            d = dict(r)
            for k in ("sources_json", "corrections_json", "extras_json"):
                if d.get(k):
                    try:
                        d[k.replace("_json", "")] = json.loads(d[k])
                    except Exception:
                        d[k.replace("_json", "")] = {}
                    d.pop(k, None)
            by_date.setdefault(d["target_date"], []).append(d)

        out: List[Dict] = []
        for date_key, candidates in by_date.items():
            # Candidates are already ordered id DESC (newest first)
            preferred = next(
                (c for c in candidates if c.get("mode") == prefer_mode),
                None,
            )
            out.append(preferred if preferred is not None else candidates[0])
        # Preserve target_date DESC ordering
        out.sort(key=lambda d: d["target_date"], reverse=True)
        return out

    # -- QRF predictions --------------------------------------------------
    def save_qrf_prediction(self, forecast_id: int,
                             p10: float, p50: float, p90: float,
                             interval_width: Optional[float] = None,
                             n_training: Optional[int] = None) -> None:
        if interval_width is None:
            interval_width = p90 - p10
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO qrf_predictions
                   (forecast_id, p10_delta, p50_delta, p90_delta,
                    interval_width, n_training, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (forecast_id, float(p10), float(p50), float(p90),
                 float(interval_width),
                 int(n_training) if n_training is not None else None,
                 _utc_now_iso()),
            )

    def get_qrf_prediction(self, forecast_id: int) -> Optional[Dict]:
        with self._conn() as c:
            r = c.execute(
                "SELECT * FROM qrf_predictions WHERE forecast_id=?",
                (forecast_id,),
            ).fetchone()
            return dict(r) if r else None

    # -- Regime shifts ----------------------------------------------------
    def save_regime_shift(self, row: Dict[str, Any]) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO regime_shifts
                   (detected_at, target_date, shift_type,
                    prev_state, new_state, weight_reset_applied)
                   VALUES (?,?,?,?,?,?)""",
                (
                    row.get("detected_at") or _utc_now_iso(),
                    row["target_date"],
                    row["shift_type"],
                    row.get("prev_state"),
                    row.get("new_state"),
                    int(row.get("weight_reset_applied") or 0),
                ),
            )
            return cur.lastrowid

    def recent_regime_shifts(self, days: int = 7) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM regime_shifts "
                "WHERE substr(target_date, 1, 10) >= date('now', ?) "
                "ORDER BY id DESC",
                (f"-{days} days",),
            ).fetchall()
            return [dict(r) for r in rows]

    def attribution_rows(self, days: int = 30) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM correction_attribution "
                "WHERE substr(target_date, 1, 10) >= date('now', ?) "
                "ORDER BY target_date DESC",
                (f"-{days} days",),
            ).fetchall()
            return [dict(r) for r in rows]


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
