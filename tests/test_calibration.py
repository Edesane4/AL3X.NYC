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
