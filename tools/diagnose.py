"""AL3X.NYC diagnostic — read-only health check.

Answers the three questions:
  1. Has the Kalman bias been consistently high, or is it a recent drift?
  2. When did ECMWF last return a value, and which sources are reliable?
  3. Is the regime-shift detector catching real shifts or noise?

Plus a few bonus checks:
  4. BMA source-bias and source-variance trends over time.
  5. QRF training status — why has it never fired?
  6. Analog engine status — why has it never fired?
  7. Correction attribution summary — which corrections are helping?

Run:
    python3 tools/diagnose.py

Reads from the DB path in AL3X_DB_PATH env var, or ./al3x.db.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def _connect() -> sqlite3.Connection:
    db_path = os.environ.get("AL3X_DB_PATH", "./al3x.db")
    if not Path(db_path).exists():
        print(f"ERROR: database not found at {db_path}")
        print("Set AL3X_DB_PATH env var or run from the al3x.nyc repo root.")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _hr(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _sub(title: str) -> None:
    print()
    print(f"--- {title} ---")


# ----------------------------------------------------------------------------
# Check 1: Kalman bias history
# ----------------------------------------------------------------------------

def check_kalman_bias(conn: sqlite3.Connection) -> None:
    _hr("1. KALMAN BIAS HISTORY")
    print("Question: is the +6.99°F Kalman bias consistent or drifting?")
    print("Method: for each scored day, compute kalman_avg - cli_high.")
    print()

    rows = conn.execute("""
        SELECT f.target_date as date,
               json_extract(f.extras_json, '$.kalman.projected_max_f') as kalman,
               f.final_f as final,
               t.recorded_high_f as cli,
               f.mode as mode
        FROM forecasts f
        JOIN cli_truth t ON t.target_date = f.target_date
        WHERE json_extract(f.extras_json, '$.kalman.projected_max_f') IS NOT NULL
          AND f.mode = 'intraday'
        ORDER BY f.target_date ASC, f.id DESC
    """).fetchall()

    if not rows:
        print("  NO KALMAN-with-CLI pairs found. Either Kalman has not")
        print("  been firing, or no intraday days have been verified.")
        return

    # Group: keep last intraday per day
    by_day: dict[str, dict] = {}
    for r in rows:
        d = r["date"]
        if d not in by_day:
            by_day[d] = dict(r)

    print(f"  {'Date':<12} {'Kalman':>8} {'Final':>8} {'CLI':>8} "
          f"{'K-CLI':>8} {'Final-CLI':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    k_errs, f_errs = [], []
    for d in sorted(by_day.keys()):
        r = by_day[d]
        k_err = float(r["kalman"]) - float(r["cli"])
        f_err = float(r["final"]) - float(r["cli"])
        k_errs.append(k_err)
        f_errs.append(f_err)
        print(f"  {d:<12} {float(r['kalman']):>8.1f} {float(r['final']):>8.1f} "
              f"{float(r['cli']):>8.1f} {k_err:>+8.2f} {f_err:>+10.2f}")

    if k_errs:
        mean_k = sum(k_errs) / len(k_errs)
        mean_f = sum(f_errs) / len(f_errs)
        k_mae = sum(abs(e) for e in k_errs) / len(k_errs)
        f_mae = sum(abs(e) for e in f_errs) / len(f_errs)
        print()
        print(f"  Kalman mean error:  {mean_k:+.2f}°F   MAE: {k_mae:.2f}°F")
        print(f"  Final  mean error:  {mean_f:+.2f}°F   MAE: {f_mae:.2f}°F")
        print()
        if mean_k > 3.0:
            print(f"  ⚠  Kalman systematically OVER-predicts by {mean_k:.1f}°F.")
            print("     BMA is compensating with a +bias, but the raw")
            print("     Kalman input to the ensemble is still 55% weighted.")
        elif mean_k < -3.0:
            print(f"  ⚠  Kalman systematically UNDER-predicts by {abs(mean_k):.1f}°F.")


# ----------------------------------------------------------------------------
# Check 2: Source availability
# ----------------------------------------------------------------------------

def check_source_availability(conn: sqlite3.Connection) -> None:
    _hr("2. SOURCE AVAILABILITY (last 200 forecasts)")
    print("Question: which sources are returning values vs. failing?")
    print()

    rows = conn.execute("""
        SELECT id, issued_at, sources_json
        FROM forecasts
        ORDER BY id DESC
        LIMIT 200
    """).fetchall()

    sources = ["hrrr", "nws_point", "gfs_mos", "ecmwf", "nbm",
               "asos_trend", "kalman", "gfs_ensemble"]
    counts = {s: {"value": 0, "error": 0, "missing": 0} for s in sources}
    last_value_ts: dict[str, str] = {}
    last_error: dict[str, str] = {}

    for r in rows:
        try:
            srcs = json.loads(r["sources_json"] or "{}")
        except Exception:
            continue
        for s in sources:
            payload = srcs.get(s)
            if payload is None:
                counts[s]["missing"] += 1
                continue
            v = payload.get("value")
            err = payload.get("error")
            if v is not None:
                counts[s]["value"] += 1
                if s not in last_value_ts:
                    last_value_ts[s] = r["issued_at"]
            elif err:
                counts[s]["error"] += 1
                if s not in last_error:
                    last_error[s] = err
            else:
                counts[s]["missing"] += 1

    print(f"  {'Source':<15} {'ValueOK':>8} {'Errors':>8} {'Missing':>8} "
          f"{'Last value':<22} {'Last error':<40}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*22} {'-'*40}")
    for s in sources:
        c = counts[s]
        total = c["value"] + c["error"] + c["missing"]
        lv = last_value_ts.get(s, "NEVER")
        le = last_error.get(s, "—")
        if lv != "NEVER":
            lv = lv[:19]
        print(f"  {s:<15} {c['value']:>8} {c['error']:>8} {c['missing']:>8} "
              f"{lv:<22} {le[:40]:<40}")

    print()
    for s in sources:
        c = counts[s]
        total = c["value"] + c["error"] + c["missing"]
        if total == 0:
            continue
        ok_pct = 100.0 * c["value"] / total
        if ok_pct < 20 and c["value"] == 0:
            print(f"  ⚠  {s}: 0% availability across last {total} forecasts — "
                  f"source is DEAD.")
        elif ok_pct < 50:
            print(f"  ⚠  {s}: only {ok_pct:.0f}% availability — unreliable.")


# ----------------------------------------------------------------------------
# Check 3: Regime-shift detector
# ----------------------------------------------------------------------------

def check_regime_detector(conn: sqlite3.Connection) -> None:
    _hr("3. REGIME-SHIFT DETECTOR")
    print("Question: is the detector catching real regime changes or firing")
    print("on noise? Expected: 0-2 shifts per day under stable weather.")
    print()

    rows = conn.execute("""
        SELECT shift_type, target_date, detected_at,
               prev_state, new_state
        FROM regime_shifts
        ORDER BY id DESC
    """).fetchall()

    if not rows:
        print("  No regime shifts recorded.")
        return

    by_type = Counter(r["shift_type"] for r in rows)
    by_day = defaultdict(int)
    for r in rows:
        by_day[r["target_date"]] += 1

    print(f"  Total shifts recorded: {len(rows)}")
    print(f"  Distinct days with shifts: {len(by_day)}")
    print(f"  Avg shifts per day: {len(rows)/max(1,len(by_day)):.1f}")
    print()
    print(f"  By type:")
    for st, n in by_type.most_common():
        print(f"    {st:<20} {n:>4}")
    print()
    print(f"  Top 5 busiest days:")
    for day, n in sorted(by_day.items(), key=lambda x: -x[1])[:5]:
        print(f"    {day}  {n} shifts")

    # Count how many have empty prev/new state (spread shifts are these)
    empty_state = sum(1 for r in rows
                      if not r["prev_state"] and not r["new_state"])
    if empty_state:
        print()
        print(f"  {empty_state} shifts ({100*empty_state/len(rows):.0f}%) "
              f"have empty prev/new state.")
        print(f"  These are mostly 'spread' shifts — noise from small model")
        print(f"  disagreement fluctuations.")

    if len(rows) / max(1, len(by_day)) > 3:
        print()
        print(f"  ⚠  Detector is firing excessively. Threshold is too loose.")


# ----------------------------------------------------------------------------
# Check 4: BMA bias/variance trends
# ----------------------------------------------------------------------------

def check_bma_trends(conn: sqlite3.Connection) -> None:
    _hr("4. BMA SOURCE BIAS / VARIANCE (latest state)")
    print("Question: what has BMA learned about each source?")
    print()

    r = conn.execute("""
        SELECT extras_json
        FROM forecasts
        WHERE json_extract(extras_json, '$.bma') IS NOT NULL
        ORDER BY id DESC
        LIMIT 1
    """).fetchone()

    if not r:
        print("  No BMA data found.")
        return

    extras = json.loads(r["extras_json"])
    bma = extras.get("bma") or {}
    biases = bma.get("source_biases") or {}
    variances = bma.get("source_variances") or {}

    print(f"  {'Source':<15} {'Bias (°F)':>12} {'Variance':>12} {'Sigma':>8}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*8}")
    for s in sorted(biases.keys()):
        b = biases[s]
        v = variances.get(s, 0.0)
        sigma = v ** 0.5 if v > 0 else 0
        flag = ""
        if abs(b) > 3.0:
            flag = "  ⚠ HIGH BIAS"
        elif v > 10.0:
            flag = "  ⚠ HIGH VAR"
        print(f"  {s:<15} {b:>+12.2f} {v:>12.2f} {sigma:>8.2f}{flag}")

    print()
    print(f"  History days in BMA: {bma.get('history_days')}")


# ----------------------------------------------------------------------------
# Check 5: QRF training status
# ----------------------------------------------------------------------------

def check_qrf_status(conn: sqlite3.Connection) -> None:
    _hr("5. QRF (QUANTILE REGRESSION FOREST) STATUS")
    print("Question: has QRF ever trained? If not, why?")
    print()

    qrf_rows = conn.execute("SELECT COUNT(*) as n FROM qrf_predictions").fetchone()
    print(f"  QRF predictions stored: {qrf_rows['n']}")

    verified = conn.execute("""
        SELECT COUNT(DISTINCT target_date) as n
        FROM cli_truth
    """).fetchone()
    print(f"  Verified days available for training: {verified['n']}")
    print(f"  QRF training threshold: 30 samples (min_samples=30)")
    print()

    if verified["n"] < 30:
        print(f"  → QRF CANNOT train yet. Need {30 - verified['n']} more")
        print(f"    verified days. This is expected at {verified['n']} days.")
    else:
        print(f"  → QRF should train but has not. Check logs for errors")
        print(f"    ('quantile-forest' library may be missing — check")
        print(f"    `pip list | grep -i quantile` on the host).")

    # Also check how many forecasts_with_truth would yield
    with_truth = conn.execute("""
        SELECT COUNT(*) as n
        FROM forecasts f JOIN cli_truth t ON t.target_date = f.target_date
        WHERE f.mode = 'night_before'
    """).fetchone()
    print(f"  Night-before forecasts with CLI truth: {with_truth['n']}")


# ----------------------------------------------------------------------------
# Check 6: Analog engine status
# ----------------------------------------------------------------------------

def check_analog_status(conn: sqlite3.Connection) -> None:
    _hr("6. ANALOG ENGINE STATUS")
    print("Question: has analog matching ever fired? If not, why?")
    print()

    with_analog = conn.execute("""
        SELECT COUNT(*) as n
        FROM forecasts
        WHERE json_extract(extras_json, '$.analog') IS NOT NULL
          AND json_extract(extras_json, '$.analog.analog_bias_f') IS NOT NULL
    """).fetchone()
    print(f"  Forecasts with analog matches: {with_analog['n']}")

    verified = conn.execute("""
        SELECT COUNT(DISTINCT target_date) as n FROM cli_truth
    """).fetchone()
    print(f"  Historical verified days available: {verified['n']}")
    print(f"  Analog engine threshold: _MIN_HISTORY = 14 days")
    print()

    if verified["n"] < 14:
        print(f"  → Analog engine should not fire yet (need 14+ verified days).")
    elif with_analog["n"] == 0:
        print(f"  → Analog engine should fire but has not. Possible causes:")
        print(f"    - Feature vector builder returning None for too many rows")
        print(f"    - Library lookup finding <14 valid (fv, err) pairs")
        print(f"    - Exception being swallowed silently")
        print(f"    Check the 'analog: unavailable' log lines.")


# ----------------------------------------------------------------------------
# Check 7: Correction attribution
# ----------------------------------------------------------------------------

def check_attribution(conn: sqlite3.Connection) -> None:
    _hr("7. CORRECTION ATTRIBUTION")
    print("Question: which corrections are helping vs hurting?")
    print()

    rows = conn.execute("""
        SELECT correction_name,
               COUNT(*) as n,
               SUM(CASE WHEN was_helpful = 1 THEN 1 ELSE 0 END) as helpful,
               SUM(CASE WHEN was_helpful = -1 THEN 1 ELSE 0 END) as harmful,
               AVG(delta_applied_f) as mean_delta,
               AVG(error_f) as mean_err
        FROM correction_attribution
        WHERE was_suppressed = 0
        GROUP BY correction_name
    """).fetchall()

    if not rows:
        print("  No attribution rows recorded.")
        return

    print(f"  {'Correction':<20} {'N':>4} {'Helpful':>8} {'Harmful':>8} "
          f"{'Harm %':>8} {'MeanΔ':>8} {'MeanErr':>9}")
    print(f"  {'-'*20} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
    for r in rows:
        n = r["n"]
        h_pct = 100.0 * r["harmful"] / n if n else 0
        flag = ""
        if h_pct > 55 and n >= 4:
            flag = "  ⚠ HARMFUL"
        print(f"  {r['correction_name']:<20} {n:>4} {r['helpful']:>8} "
              f"{r['harmful']:>8} {h_pct:>7.0f}% "
              f"{r['mean_delta']:>+8.2f} {r['mean_err']:>+9.2f}{flag}")


# ----------------------------------------------------------------------------
# Check 8: Headline stats for grounding
# ----------------------------------------------------------------------------

def check_headline(conn: sqlite3.Connection) -> None:
    _hr("8. HEADLINE STATS")

    totals = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM forecasts) as forecasts,
            (SELECT COUNT(*) FROM observations) as observations,
            (SELECT COUNT(*) FROM cli_truth) as cli_verified,
            (SELECT COUNT(*) FROM scores) as scores,
            (SELECT COUNT(*) FROM correction_attribution) as attributions,
            (SELECT COUNT(*) FROM regime_shifts) as regime_shifts
    """).fetchone()

    print()
    print(f"  Forecasts:            {totals['forecasts']:>6}")
    print(f"  Observations:         {totals['observations']:>6}")
    print(f"  CLI verified days:    {totals['cli_verified']:>6}")
    print(f"  Scores:               {totals['scores']:>6}")
    print(f"  Attribution rows:     {totals['attributions']:>6}")
    print(f"  Regime shifts:        {totals['regime_shifts']:>6}")

    maes = conn.execute("""
        SELECT mode, ROUND(AVG(abs_error_f), 2) as mae, COUNT(*) as n
        FROM scores
        GROUP BY mode
    """).fetchall()
    print()
    for r in maes:
        print(f"  {r['mode']:<15} MAE: {r['mae']}°F  (n={r['n']})")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    conn = _connect()
    try:
        print()
        print("AL3X.NYC DIAGNOSTIC — read-only health check")
        print(f"Run at: {datetime.now().isoformat(timespec='seconds')}")
        print(f"DB: {os.environ.get('AL3X_DB_PATH', './al3x.db')}")

        check_headline(conn)
        check_source_availability(conn)
        check_kalman_bias(conn)
        check_bma_trends(conn)
        check_regime_detector(conn)
        check_qrf_status(conn)
        check_analog_status(conn)
        check_attribution(conn)

        print()
        print("=" * 78)
        print("  Done. Paste this entire output back to me.")
        print("=" * 78)
        print()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
