"""Session 7 Part 2 — /api/trajectory endpoint structure tests.

Shape tests, not value tests. The endpoint reads from the live DB; we
verify it returns the expected JSON contract so the dashboard never
breaks if the data is sparse.
"""
import json
import os
import sqlite3
import asyncio

import pytest

from al3x.calibration import record_calibration, ensure_schema


def _seed_db(path):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issued_at TEXT NOT NULL,
            target_date TEXT NOT NULL,
            mode TEXT NOT NULL,
            final_f REAL NOT NULL,
            sources_json TEXT NOT NULL DEFAULT '{}',
            extras_json TEXT
        );
    """)
    ensure_schema(conn)
    return conn


@pytest.fixture
def patched_db(tmp_path, monkeypatch):
    """Point the endpoint at a fresh tmp DB via AL3X_DB_PATH."""
    db_path = tmp_path / "test.db"
    conn = _seed_db(str(db_path))
    monkeypatch.setenv("AL3X_DB_PATH", str(db_path))
    yield conn, db_path
    try:
        conn.close()
    except Exception:
        pass


def test_endpoint_returns_three_datasets_and_headline(patched_db):
    conn, _ = patched_db

    bands = json.dumps({"bands": {"p10": 50.0, "p50": 55.0, "p90": 60.0, "sigma_f": 3.9}})
    sources = json.dumps({"hrrr": {"value": 56.0, "weight": 0.5},
                          "kalman": {"value": 55.5, "weight": 0.5}})

    conn.execute("INSERT INTO forecasts (issued_at, target_date, mode, final_f, sources_json, extras_json) "
                 "VALUES (?, ?, ?, ?, ?, ?)",
                 ("2026-04-24T18:00:00", "2026-04-25", "night_before", 55.0, sources, bands))
    conn.execute("INSERT INTO forecasts (issued_at, target_date, mode, final_f, sources_json, extras_json) "
                 "VALUES (?, ?, ?, ?, ?, ?)",
                 ("2026-04-25T20:00:00", "2026-04-25", "intraday", 55.5, sources, bands))
    conn.commit()
    record_calibration(conn, "2026-04-25", 56.0)

    from app import trajectory
    result = asyncio.run(trajectory())

    assert "per_source_mae" in result
    assert "ensemble_mae" in result
    assert "lead_time_matrix" in result
    assert "headline" in result
    assert "buckets" in result

    sources_seen = {r["source"] for r in result["per_source_mae"]}
    assert "hrrr" in sources_seen
    assert "kalman" in sources_seen

    em = result["ensemble_mae"][0]
    assert "final_error_f" in em
    assert "night_before_error_f" in em
    assert em["final_error_f"] is not None
    assert em["night_before_error_f"] is not None

    h = result["headline"]
    assert h["verified_days"] == 1
    assert h["final_mae_cumulative_f"] is not None
    assert h["night_before_mae_cumulative_f"] is not None


def test_endpoint_handles_empty_db(patched_db):
    from app import trajectory
    result = asyncio.run(trajectory())

    assert result["per_source_mae"] == []
    assert result["ensemble_mae"] == []
    assert result["lead_time_matrix"] == []
    assert result["headline"]["verified_days"] == 0
    assert result["headline"]["final_mae_cumulative_f"] is None
    assert result["headline"]["night_before_mae_cumulative_f"] is None


def test_head_to_head_returns_expected_shape(patched_db):
    """Session 8 Part 3 — /api/head_to_head returns aggregate +
    timestamped per-forecaster lists, with MAE computed when truth
    is verified."""
    conn, _ = patched_db
    # Add the NWS table — patched_db's _seed_db doesn't include it
    # because it predates Session 8.
    conn.executescript("""
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
    """)

    conn.execute(
        "INSERT INTO forecasts (issued_at, target_date, mode, final_f) "
        "VALUES (?, ?, ?, ?)",
        ("2026-05-05T18:00:00", "2026-05-06", "night_before", 75.0))
    conn.execute(
        "INSERT INTO forecasts (issued_at, target_date, mode, final_f) "
        "VALUES (?, ?, ?, ?)",
        ("2026-05-06T14:00:00", "2026-05-06", "intraday", 73.0))
    conn.execute(
        "INSERT INTO nws_official_forecasts "
        "(captured_at, target_date, forecasted_high_f, source_url, period_name) "
        "VALUES (?, ?, ?, ?, ?)",
        ("2026-05-06T08:00:00", "2026-05-06", 76.0, "http://x", "Today"))
    conn.commit()
    record_calibration(conn, "2026-05-06", 74.0)

    from app import head_to_head
    result = asyncio.run(head_to_head(target_date="2026-05-06"))

    agg = result["aggregate"]
    assert agg["target_date"] == "2026-05-06"
    assert agg["realized_high_f"] == 74.0
    assert agg["al3x_count"] == 2
    assert agg["nws_count"] == 1
    assert "al3x_mae_f" in agg
    assert "nws_mae_f" in agg
    # AL3X errors: |74-75|=1, |74-73|=1 → MAE 1.0
    assert agg["al3x_mae_f"] == 1.0
    # NWS errors: |74-76|=2 → MAE 2.0
    assert agg["nws_mae_f"] == 2.0
    # Final-revision errors are last entry
    assert agg["al3x_final_error_f"] == 1.0
    assert agg["nws_final_error_f"] == 2.0
    assert len(result["al3x_forecasts"]) == 2
    assert len(result["nws_forecasts"]) == 1


def test_head_to_head_handles_no_truth(patched_db):
    """When the day has no calibration row yet, MAE keys are absent
    but lists still come back."""
    conn, _ = patched_db
    conn.executescript("""
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
    """)
    conn.execute(
        "INSERT INTO forecasts (issued_at, target_date, mode, final_f) "
        "VALUES (?, ?, ?, ?)",
        ("2026-05-07T08:00:00", "2026-05-07", "intraday", 70.0))
    conn.commit()

    from app import head_to_head
    result = asyncio.run(head_to_head(target_date="2026-05-07"))

    agg = result["aggregate"]
    assert agg["realized_high_f"] is None
    assert agg["al3x_count"] == 1
    assert "al3x_mae_f" not in agg
    assert "nws_mae_f" not in agg


def test_lead_time_buckets_classify_correctly(patched_db):
    conn, _ = patched_db
    bands = json.dumps({"bands": {"p10": 50.0, "p50": 55.0, "p90": 60.0, "sigma_f": 3.9}})
    sources = '{}'

    rows = [
        ("2026-04-24T18:00:00", "night_before", 53.0),
        ("2026-04-25T03:00:00", "intraday", 54.0),
        ("2026-04-25T09:00:00", "intraday", 55.0),
        ("2026-04-25T14:00:00", "intraday", 55.5),
        ("2026-04-25T20:00:00", "intraday", 56.0),
    ]
    for issued_at, mode, final_f in rows:
        conn.execute(
            "INSERT INTO forecasts (issued_at, target_date, mode, final_f, sources_json, extras_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (issued_at, "2026-04-25", mode, final_f, sources, bands),
        )
    conn.commit()
    record_calibration(conn, "2026-04-25", 56.0)

    from app import trajectory
    result = asyncio.run(trajectory())

    matrix = result["lead_time_matrix"]
    assert len(matrix) == 1
    row = matrix[0]
    for bucket in ["night_before", "pre_dawn", "morning", "pre_peak", "post_peak"]:
        assert row[bucket]["n_days"] == 1, f"bucket {bucket} missing"
        assert row[bucket]["mae_f"] is not None
