"""Daily diagnostic runner for AL3X.NYC.

Prints a plain-English morning check-in block at the top, then
delegates to whatever existing diagnostic sections live below. When
run on a machine without al3x.db present, the SQL-backed sections
are skipped with a clear message.

Session 2 G11 — the morning check-in is what the user reads first,
so it's intentionally short (4 lines) and uses INFO-level phrasing
rather than a field dump.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make the al3x package importable when this is run as a script.
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from al3x import config as cfg  # noqa: E402


def _open_db() -> sqlite3.Connection | None:
    """Open al3x.db read-only. Returns None if the file is absent."""
    db_path = os.environ.get("AL3X_DB_PATH", str(BASE / "al3x.db"))
    if not Path(db_path).exists():
        print(f"(al3x.db not found at {db_path} — DB sections skipped)")
        return None
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _print_morning_checkin(conn: sqlite3.Connection | None) -> None:
    """Session 2 — top-of-output plain-English health summary."""
    today = datetime.now(cfg.EASTERN).date()
    yesterday = today - timedelta(days=1)

    # 1. Correction cap fires overnight (best-effort grep of al3x.log)
    cap_fired_count = None
    log_path = BASE / "al3x.log"
    if log_path.exists():
        cap_fired_count = 0
        try:
            with log_path.open("r", errors="replace") as f:
                for line in f:
                    if "cap fired" in line.lower():
                        cap_fired_count += 1
        except OSError:
            cap_fired_count = None

    # 2. Regime shifts yesterday
    regime_shifts_yday = None
    if conn is not None:
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM regime_shifts "
                "WHERE target_date = ?",
                (yesterday.isoformat(),),
            ).fetchone()
            regime_shifts_yday = int(row["n"]) if row else 0
        except sqlite3.OperationalError:
            regime_shifts_yday = None

    # 3. BMA Kalman bias — pulled from the most recent forecast's
    #    extras_json rather than a non-existent bma_source_state table
    #    (the current schema stores BMA diagnostics inline on each
    #    forecast). Reports "N/A" if no row or no kalman bias present.
    bma_kalman_bias = None
    if conn is not None:
        try:
            import json
            row = conn.execute(
                "SELECT extras_json FROM forecasts "
                "WHERE extras_json LIKE '%\"bma\":%' "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                extras = json.loads(row["extras_json"] or "{}")
                bma = extras.get("bma") or {}
                biases = bma.get("source_biases") or {}
                if "kalman" in biases:
                    bma_kalman_bias = float(biases["kalman"])
        except Exception:  # noqa: BLE001
            bma_kalman_bias = None

    # 4. ECMWF cycles fresh (last 24h) — JSON path depends on schema.
    #    sources_json in forecasts.sources is shaped
    #    {"ecmwf": {"value": <float|null>, ...}, ...}.
    ecmwf_fresh = None
    if conn is not None:
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM forecasts WHERE "
                "json_extract(sources_json, '$.ecmwf.value') IS NOT NULL "
                "AND issued_at > datetime('now', '-1 day')"
            ).fetchone()
            ecmwf_fresh = int(row["n"]) if row else 0
        except sqlite3.OperationalError:
            ecmwf_fresh = None

    bar = "=" * 78
    print(bar)
    print("MORNING CHECK-IN")
    print(bar)
    if cap_fired_count is None:
        cap_line = "LOG NOT FOUND"
    else:
        cap_line = f"fired {cap_fired_count}x"
    print(f"  [1] Correction cap: {cap_line}")
    if regime_shifts_yday is None:
        print("  [2] Regime shifts yesterday: N/A")
    else:
        marker = "✓" if regime_shifts_yday <= 3 else "⚠"
        print(f"  [2] Regime shifts yesterday: {regime_shifts_yday} "
              f"(target ≤3) {marker}")
    if bma_kalman_bias is None:
        print("  [3] BMA Kalman bias: N/A")
    else:
        print(f"  [3] BMA Kalman bias: {bma_kalman_bias:+.2f}°F "
              "(disabled in output)")
    if ecmwf_fresh is None:
        print("  [4] ECMWF cycles fresh (24h): N/A")
    else:
        print(f"  [4] ECMWF cycles fresh (24h): {ecmwf_fresh}")
    print()


def main() -> int:
    conn = _open_db()
    _print_morning_checkin(conn)

    # Existing sections — placeholder stubs. Wire up to the real
    # diagnostic sections as they land. When nothing is wired, the
    # morning check-in is the whole output.
    if conn is None:
        return 0

    # Section 1+ would go here.

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
