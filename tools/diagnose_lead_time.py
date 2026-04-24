"""Lead-time stratified error analysis.

Answers: does forecast accuracy actually improve as lead time shrinks?
This is the key empirical question for Kalshi integration — if sigma
doesn't narrow as lead time decreases, bucket probabilities will be
systematically wrong.

For every forecast that has a verified CLI truth, compute:
  - lead_hours (from issued_at to 3 PM Eastern on target_date)
  - error_f (cli_high - final_f)

Then bin by lead-hour buckets and report MAE + mean error per bucket.
Also report how stated uncertainty_f compares to actual observed error.

Run:
    python3 tools/diagnose_lead_time.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path


def _connect() -> sqlite3.Connection:
    db_path = os.environ.get("AL3X_DB_PATH", "./al3x.db")
    if not Path(db_path).exists():
        print(f"ERROR: database not found at {db_path}")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _lead_hours(issued_at: str, target_date: str) -> float | None:
    """Hours from issue time to 3 PM Eastern on target_date (peak heating)."""
    try:
        issued = datetime.fromisoformat(issued_at)
        target_peak = datetime.fromisoformat(f"{target_date}T15:00:00-04:00")
        if issued.tzinfo is None:
            issued = issued.replace(tzinfo=target_peak.tzinfo)
        hours = (target_peak - issued).total_seconds() / 3600.0
        return max(0.0, hours)
    except Exception:
        return None


def _bucket_for(lead_h: float) -> str:
    if lead_h >= 20:
        return "24h+ (night-before)"
    if lead_h >= 12:
        return "12-20h"
    if lead_h >= 6:
        return "6-12h"
    if lead_h >= 3:
        return "3-6h"
    if lead_h >= 1:
        return "1-3h"
    return "0-1h (final)"


_BUCKET_ORDER = [
    "24h+ (night-before)", "12-20h", "6-12h",
    "3-6h", "1-3h", "0-1h (final)",
]


def main() -> None:
    conn = _connect()
    try:
        print("\nLEAD-TIME STRATIFIED ERROR ANALYSIS")
        print("=" * 78)

        rows = conn.execute("""
            SELECT f.id, f.issued_at, f.target_date, f.mode,
                   f.final_f, f.uncertainty_f, f.running_asos_max_f,
                   t.recorded_high_f as cli_f
            FROM forecasts f
            JOIN cli_truth t ON t.target_date = f.target_date
            ORDER BY f.target_date ASC, f.id ASC
        """).fetchall()

        if not rows:
            print("No verified forecasts found. Need CLI-verified days first.")
            return

        print(f"\nTotal forecast+CLI pairs: {len(rows)}")
        print(f"Unique verified days:     "
              f"{len(set(r['target_date'] for r in rows))}")
        print()

        # Bucketize
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            lead = _lead_hours(r["issued_at"], r["target_date"])
            if lead is None:
                continue
            bucket = _bucket_for(lead)
            err = float(r["cli_f"]) - float(r["final_f"])
            by_bucket[bucket].append({
                "lead_h": lead,
                "error_f": err,
                "abs_err": abs(err),
                "final_f": float(r["final_f"]),
                "cli_f": float(r["cli_f"]),
                "sigma": float(r["uncertainty_f"] or 0),
                "mode": r["mode"],
                "date": r["target_date"],
            })

        print(f"{'Lead bucket':<24} {'N':>5} {'MAE':>7} {'MeanErr':>9} "
              f"{'MedianSigma':>12} {'InSigma%':>10} {'In2σ%':>8}")
        print(f"{'-'*24} {'-'*5} {'-'*7} {'-'*9} {'-'*12} {'-'*10} {'-'*8}")

        for bucket in _BUCKET_ORDER:
            entries = by_bucket.get(bucket, [])
            if not entries:
                print(f"{bucket:<24} {0:>5} {'—':>7} {'—':>9} "
                      f"{'—':>12} {'—':>10} {'—':>8}")
                continue
            n = len(entries)
            mae = sum(e["abs_err"] for e in entries) / n
            mean_err = sum(e["error_f"] for e in entries) / n
            sigmas = sorted(e["sigma"] for e in entries)
            median_sigma = sigmas[n // 2] if sigmas else 0
            within_1sigma = sum(
                1 for e in entries if e["abs_err"] <= e["sigma"]
            )
            within_2sigma = sum(
                1 for e in entries if e["abs_err"] <= 2 * e["sigma"]
            )
            in_pct = 100.0 * within_1sigma / n if n else 0
            in2_pct = 100.0 * within_2sigma / n if n else 0
            print(f"{bucket:<24} {n:>5} {mae:>6.2f}°F {mean_err:>+8.2f}°F "
                  f"{median_sigma:>10.2f}°F {in_pct:>9.0f}% {in2_pct:>7.0f}%")

        # Calibration summary
        print()
        print("CALIBRATION INTERPRETATION:")
        print("  InSigma%:  Should be ~68% if sigma is well-calibrated.")
        print("             Higher = too conservative. Lower = overconfident.")
        print("  In2σ%:     Should be ~95% if sigma is well-calibrated.")
        print()

        # Does MAE actually shrink with lead time?
        print("LEAD-TIME MAE PROGRESSION:")
        prev_mae = None
        for bucket in reversed(_BUCKET_ORDER):
            entries = by_bucket.get(bucket, [])
            if not entries:
                continue
            mae = sum(e["abs_err"] for e in entries) / len(entries)
            arrow = ""
            if prev_mae is not None:
                if mae < prev_mae - 0.1:
                    arrow = "  ✓ improving"
                elif mae > prev_mae + 0.1:
                    arrow = "  ⚠ DEGRADING"
                else:
                    arrow = "  ≈ flat"
            print(f"  {bucket:<24}  MAE={mae:.2f}°F{arrow}")
            prev_mae = mae

        # Per-day latest vs earliest
        print()
        print("PER-DAY: night-before → final intraday error reduction:")
        by_day: dict[str, list] = defaultdict(list)
        for entries in by_bucket.values():
            for e in entries:
                by_day[e["date"]].append(e)

        print(f"  {'Date':<12} {'NB err':>8} {'Final err':>10} "
              f"{'Improvement':>12}")
        print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*12}")
        total_improvement = 0.0
        improvements = 0
        for date in sorted(by_day.keys()):
            day_entries = sorted(by_day[date], key=lambda e: -e["lead_h"])
            nb = next((e for e in day_entries if e["lead_h"] >= 18), None)
            final = next(
                (e for e in reversed(day_entries) if e["lead_h"] < 4), None)
            if nb and final:
                improvement = abs(nb["error_f"]) - abs(final["error_f"])
                total_improvement += improvement
                improvements += 1
                arrow = "✓" if improvement > 0 else "✗"
                print(f"  {date:<12} {nb['error_f']:>+7.1f}°F "
                      f"{final['error_f']:>+9.1f}°F "
                      f"{improvement:>+10.1f}°F  {arrow}")

        if improvements:
            avg = total_improvement / improvements
            print(f"\n  Average improvement NB → final: {avg:+.2f}°F")
            if avg > 0.5:
                print("  ✓ Intraday process measurably improves on night-before.")
            elif avg > 0:
                print("  ≈ Intraday slightly better than night-before.")
            else:
                print("  ⚠ Intraday NOT improving on night-before — "
                      "something is wrong with the intraday cycle.")

        print()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
