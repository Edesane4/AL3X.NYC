"""One-shot backfill: walk every verified CLI day already in the DB
and populate calibration_records. Safe to re-run — uses upsert.

Usage:
    python3 tools/backfill_calibration.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from al3x.calibration import record_calibration, ensure_schema  # noqa: E402


def main() -> int:
    conn = sqlite3.connect("al3x.db")
    ensure_schema(conn)

    rows = conn.execute(
        """
        SELECT target_date, recorded_high_f
          FROM cli_truth
         ORDER BY target_date ASC
        """
    ).fetchall()

    print(f"Backfilling {len(rows)} calibration records...")
    written = 0
    skipped = 0
    for target_date, realized in rows:
        result = record_calibration(conn, target_date, float(realized))
        if result is None:
            skipped += 1
        else:
            written += 1

    print(f"Wrote {written}, skipped {skipped} (no forecast found).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
