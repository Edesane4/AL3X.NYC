import json
import sqlite3
from al3x.calibration import record_calibration, ensure_schema


def _setup_in_memory_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_date TEXT NOT NULL,
            issued_at TEXT NOT NULL,
            extras_json TEXT
        );
    """)
    ensure_schema(conn)
    return conn


def test_records_inside_band_when_truth_is_within_p10_p90():
    conn = _setup_in_memory_db()
    extras = json.dumps({
        "bands": {"p10": 50.0, "p50": 55.0, "p90": 60.0, "sigma_f": 3.9},
        "final_f": 55.0,
    })
    conn.execute(
        "INSERT INTO forecasts (target_date, issued_at, extras_json) VALUES (?, ?, ?)",
        ("2026-04-25", "2026-04-25T20:00:00", extras),
    )
    conn.commit()

    result = record_calibration(conn, "2026-04-25", realized_high_f=56.0)
    assert result is not None
    assert result["inside_80"] == 1
    assert result["error_signed_f"] == 1.0


def test_records_outside_band_when_truth_misses():
    conn = _setup_in_memory_db()
    extras = json.dumps({"bands": {"p10": 50.0, "p50": 55.0, "p90": 60.0, "sigma_f": 3.9}})
    conn.execute(
        "INSERT INTO forecasts (target_date, issued_at, extras_json) VALUES (?, ?, ?)",
        ("2026-04-25", "2026-04-25T20:00:00", extras),
    )
    conn.commit()

    result = record_calibration(conn, "2026-04-25", realized_high_f=65.0)
    assert result["inside_80"] == 0
    assert result["error_signed_f"] == 10.0


def test_backfills_from_uncertainty_when_bands_missing():
    """Pre-Session-6 forecasts have uncertainty_f but no bands. We synthesize."""
    conn = _setup_in_memory_db()
    extras = json.dumps({"final_f": 55.0, "uncertainty_f": 2.0})  # no bands
    conn.execute(
        "INSERT INTO forecasts (target_date, issued_at, extras_json) VALUES (?, ?, ?)",
        ("2026-04-20", "2026-04-20T20:00:00", extras),
    )
    conn.commit()

    result = record_calibration(conn, "2026-04-20", realized_high_f=55.0)
    assert result["p50"] == 55.0
    # Synthesized band: 55 ± 1.2816 * 2.0 ≈ [52.44, 57.56]
    assert abs(result["p10"] - (55.0 - 1.2816 * 2.0)) < 0.01
    assert result["inside_80"] == 1


def test_skips_when_no_forecast_exists():
    conn = _setup_in_memory_db()
    result = record_calibration(conn, "2026-04-19", realized_high_f=50.0)
    assert result is None


def _setup_db_with_mode():
    """Session 7 — in-memory DB with the same columns the live forecasts
    table has (mode, final_f), so night_before logic exercises end-to-end."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_date TEXT NOT NULL,
            issued_at TEXT NOT NULL,
            mode TEXT NOT NULL,
            final_f REAL NOT NULL,
            sources_json TEXT NOT NULL DEFAULT '{}',
            extras_json TEXT
        );
    """)
    ensure_schema(conn)
    return conn


def test_records_night_before_forecast_when_present():
    conn = _setup_db_with_mode()
    nb_extras = json.dumps({
        "bands": {"p10": 50.0, "p50": 55.0, "p90": 60.0, "sigma_f": 3.9},
        "final_f": 55.0,
    })
    conn.execute(
        "INSERT INTO forecasts (target_date, issued_at, mode, final_f, extras_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("2026-04-25", "2026-04-24T18:00:00", "night_before", 55.0, nb_extras),
    )
    final_extras = json.dumps({
        "bands": {"p10": 53.0, "p50": 55.5, "p90": 58.0, "sigma_f": 1.95},
    })
    conn.execute(
        "INSERT INTO forecasts (target_date, issued_at, mode, final_f, extras_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("2026-04-25", "2026-04-25T20:00:00", "intraday", 55.5, final_extras),
    )
    conn.commit()

    result = record_calibration(conn, "2026-04-25", realized_high_f=56.0)
    assert result is not None
    assert result["night_before"]["inside_80"] == 1
    assert result["night_before"]["error_signed_f"] == 1.0  # 56 - 55
    assert result["inside_80"] == 1


def test_handles_missing_night_before_forecast():
    """Some target dates have no night_before forecast — fields should be None."""
    conn = _setup_db_with_mode()
    final_extras = json.dumps({
        "bands": {"p10": 50.0, "p50": 55.0, "p90": 60.0, "sigma_f": 3.9}})
    conn.execute(
        "INSERT INTO forecasts (target_date, issued_at, mode, final_f, extras_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("2026-04-25", "2026-04-25T20:00:00", "intraday", 55.0, final_extras),
    )
    conn.commit()

    result = record_calibration(conn, "2026-04-25", 56.0)
    assert result["night_before"]["p50"] is None
    assert result["night_before"]["inside_80"] is None
    assert result["inside_80"] == 1


def test_picks_first_night_before_when_multiple_revisions_exist():
    """The night-before forecast may be revised before midnight — the
    HONEST measurement is the FIRST issuance, not the last."""
    conn = _setup_db_with_mode()
    early = json.dumps({"bands": {"p10": 48.0, "p50": 53.0, "p90": 58.0, "sigma_f": 3.9}})
    late = json.dumps({"bands": {"p10": 51.0, "p50": 55.0, "p90": 59.0, "sigma_f": 3.1}})
    conn.execute(
        "INSERT INTO forecasts (target_date, issued_at, mode, final_f, extras_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("2026-04-25", "2026-04-24T18:00:00", "night_before", 53.0, early),
    )
    conn.execute(
        "INSERT INTO forecasts (target_date, issued_at, mode, final_f, extras_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("2026-04-25", "2026-04-24T22:30:00", "night_before", 55.0, late),
    )
    final = json.dumps({"bands": {"p10": 53.0, "p50": 55.5, "p90": 58.0, "sigma_f": 1.95}})
    conn.execute(
        "INSERT INTO forecasts (target_date, issued_at, mode, final_f, extras_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("2026-04-25", "2026-04-25T20:00:00", "intraday", 55.5, final),
    )
    conn.commit()

    result = record_calibration(conn, "2026-04-25", 56.0)
    assert result["night_before"]["p50"] == 53.0
    assert result["night_before"]["error_signed_f"] == 3.0


def test_backfills_night_before_from_uncertainty_f_when_bands_missing():
    """Pre-Session-6 night_before forecasts have no bands — synthesize."""
    conn = _setup_db_with_mode()
    nb_extras = json.dumps({"final_f": 55.0, "uncertainty_f": 2.0})  # no bands
    conn.execute(
        "INSERT INTO forecasts (target_date, issued_at, mode, final_f, extras_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("2026-04-20", "2026-04-19T18:00:00", "night_before", 55.0, nb_extras),
    )
    final_extras = json.dumps({
        "bands": {"p10": 53.0, "p50": 55.0, "p90": 57.0, "sigma_f": 1.56}})
    conn.execute(
        "INSERT INTO forecasts (target_date, issued_at, mode, final_f, extras_json) "
        "VALUES (?, ?, ?, ?, ?)",
        ("2026-04-20", "2026-04-20T20:00:00", "intraday", 55.0, final_extras),
    )
    conn.commit()

    result = record_calibration(conn, "2026-04-20", 55.0)
    assert abs(result["night_before"]["p10"] - (55.0 - 1.2816 * 2.0)) < 0.01
    assert result["night_before"]["inside_80"] == 1


def test_schema_migration_is_idempotent():
    """ensure_schema() can run multiple times without raising."""
    conn = _setup_in_memory_db()  # already calls ensure_schema once
    ensure_schema(conn)
    ensure_schema(conn)
    cols = {row[1] for row in conn.execute(
        "PRAGMA table_info(calibration_records)").fetchall()}
    assert "night_before_p50_f" in cols
    assert "night_before_inside_80" in cols
    assert "night_before_error_signed_f" in cols


def test_idempotent_upsert_on_repeated_calls():
    """CLI corrections do happen — a second call updates the row, not duplicates."""
    conn = _setup_in_memory_db()
    extras = json.dumps({"bands": {"p10": 50.0, "p50": 55.0, "p90": 60.0, "sigma_f": 3.9}})
    conn.execute(
        "INSERT INTO forecasts (target_date, issued_at, extras_json) VALUES (?, ?, ?)",
        ("2026-04-25", "2026-04-25T20:00:00", extras),
    )
    conn.commit()

    record_calibration(conn, "2026-04-25", 56.0)
    record_calibration(conn, "2026-04-25", 57.5)  # corrected

    rows = conn.execute(
        "SELECT realized_high_f FROM calibration_records WHERE target_date = ?",
        ("2026-04-25",),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 57.5
