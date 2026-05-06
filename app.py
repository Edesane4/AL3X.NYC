"""AL3X.NYC — main FastAPI + scheduler entrypoint.

Run:
    pip install -r requirements.txt
    cp .env.example .env   # fill in TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
    python app.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from al3x import config as cfg
from al3x.ai_calibration import AICalibrator
from al3x.health_summary import build_health_summary
from al3x.logging_setup import configure as configure_logging
from al3x.scheduler import AgentScheduler
from al3x.storage import Storage
from al3x.telegram_bot import TelegramNotifier, format_forecast_for_humans

log = logging.getLogger("al3x.app")

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"


# Session 2 G9 — port scanners and malformed localhost probes emit
# "Invalid HTTP request" via uvicorn.error. Harmless but noisy. Filter
# them out before any log handlers see the record.
class _InvalidHttpFilter(logging.Filter):
    """Suppress uvicorn 'Invalid HTTP request' access-log noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "Invalid HTTP request" not in msg


logging.getLogger("uvicorn.error").addFilter(_InvalidHttpFilter())


# Session 2 G7 — 5s TTL cache for /api/state. The dashboard polls this
# endpoint every ~2s; without a cache that's ~51k DB queries/day. TTL
# keeps the UI responsive while cutting load to ~17k/day.
_STATE_CACHE: dict = {"ts": 0.0, "payload": None}
_STATE_CACHE_TTL = 5.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Wire up
    db_path = os.environ.get("AL3X_DB_PATH", str(BASE / "al3x.db"))
    storage = Storage(db_path)
    notifier = TelegramNotifier()
    configure_logging(notifier)
    await notifier.start()

    ai_calibrator = AICalibrator()
    agent = AgentScheduler(storage, notifier, ai_calibrator=ai_calibrator)
    app.state.storage = storage
    app.state.notifier = notifier
    app.state.agent = agent
    app.state.ai_calibrator = ai_calibrator

    log.info("AL3X.NYC booting — DB=%s, Telegram=%s, AI=%s",
             db_path, "enabled" if notifier.configured else "disabled",
             "enabled" if ai_calibrator.enabled else "disabled")
    if notifier.configured and os.environ.get(
            "AL3X_TELEGRAM_TEST_ON_START", "true").lower() == "true":
        notifier.enqueue(
            "🟢 <b>AL3X.NYC online</b>\n"
            "Central Park high-temp agent has started. "
            "Watching KNYC 24/7."
        )

    agent.start()
    try:
        yield
    finally:
        log.info("AL3X.NYC shutting down.")
        await agent.stop()
        await notifier.stop()


app = FastAPI(title="AL3X.NYC", lifespan=lifespan)


# ---- Routes ---------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/state")
async def state():
    # Session 2 G7 — 5s TTL cache (see module-level _STATE_CACHE).
    import time
    now_ts = time.time()
    cached_payload = _STATE_CACHE["payload"]
    if cached_payload is not None and (
            now_ts - _STATE_CACHE["ts"] < _STATE_CACHE_TTL):
        return JSONResponse(cached_payload)

    storage: Storage = app.state.storage
    today = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    latest_today = storage.latest_forecast(today)
    latest_tomorrow = storage.latest_forecast(tomorrow)
    recent = storage.recent_forecasts(limit=30)
    scores = storage.recent_scores(days=30)
    truth_today = storage.get_cli_truth(today)
    logs = storage.recent_logs(limit=40)
    obs = storage.observations_today(today)

    # Stats + correction attribution
    from al3x.learning import Learning
    learning = Learning(storage)
    stats = learning.headline_stats()
    attribution = learning.attribution_stats(days=30)
    # Regime-shift history (Quant Upgrade 3)
    try:
        regime_shifts = storage.recent_regime_shifts(days=7)
    except Exception:
        regime_shifts = []
    # QRF calibration diagnostics (Quant Upgrade 1)
    qrf_stats = None
    try:
        qrf = getattr(app.state.agent.forecaster, "_qrf", None)
        if qrf is not None:
            qrf_stats = qrf.calibration_stats(storage)
    except Exception:
        qrf_stats = None

    # Session 4 Part 4 — surface climatology readiness so we can tell
    # whether Kalman's prior_rate is engaged. The 2026-04-26 5:42 AM bug
    # had prior_rate_used=false; without surfacing readiness state we
    # can't distinguish "not built yet" from "built but no bin matches
    # today's weather conditions".
    clim_state = None
    try:
        clim = getattr(app.state.agent.forecaster, "_clim", None)
        built_date = getattr(app.state.agent.forecaster,
                              "_clim_built_date", None)
        if clim is not None:
            s = clim.stats()
            clim_state = {
                "ready": bool(s.get("ready")),
                "num_bins": s.get("num_bins"),
                "fallback_mean_f_per_hr": s.get("fallback_mean"),
                "last_built_date": (built_date.isoformat()
                                     if built_date else None),
                "sample_counts": s.get("sample_counts") or {},
            }
    except Exception as e:  # noqa: BLE001
        log.info("climatology readiness probe failed: %s", e)
        clim_state = {"ready": False, "error": str(e)}

    # Running max from obs
    running_max_f = None
    if obs:
        temps = [o["temperature_f"] for o in obs
                 if o.get("temperature_f") is not None]
        running_max_f = max(temps) if temps else None

    payload = {
        "now": datetime.now(cfg.EASTERN).isoformat(),
        "target_today": today,
        "target_tomorrow": tomorrow,
        "latest_today": latest_today,
        "latest_tomorrow": latest_tomorrow,
        "recent_forecasts": recent,
        "cli_truth_today": truth_today,
        "scores_30d": scores,
        "logs": logs,
        "running_max_today_f": running_max_f,
        "stats": stats,
        "attribution": attribution,
        "regime_shifts": regime_shifts,
        "qrf_calibration": qrf_stats,
        "climatology": clim_state,
        "ai_enabled": bool(getattr(app.state, "ai_calibrator", None)
                           and app.state.ai_calibrator.enabled),
        "weights_override": storage.get_weights(),
        "biases_override": storage.get_biases(),
    }
    # Session 3 fix — alias keys so build_health_summary reads the
    # same payload that's persisted to /api/state. The /api/state
    # payload uses descriptive names; the health builder was written
    # against shorter names. Aliasing here keeps health_summary.py
    # pure and untouched. These aliases are additive — original keys
    # remain available for any other consumer.
    payload["anchor"] = payload.get("latest_today") or payload.get("latest_tomorrow")
    payload["truth_today"] = payload.get("cli_truth_today")
    payload["scores"] = payload.get("scores_30d") or []
    payload["running_max_f"] = payload.get("running_max_today_f")
    payload["now_et"] = payload.get("now")

    # Session 3 — plain-English health summary derived from the
    # payload we just built. No extra DB queries; build_health_summary
    # is pure over payload. If the builder raises, degrade to a red
    # status dict rather than breaking /api/state.
    try:
        payload["health"] = build_health_summary(payload)
    except Exception as e:  # noqa: BLE001
        log.exception("health_summary build failed: %s", e)
        payload["health"] = {
            "status": "red",
            "headline": f"health_summary build failed: {e}",
            "error": str(e),
        }
    _STATE_CACHE["ts"] = now_ts
    _STATE_CACHE["payload"] = payload
    return JSONResponse(payload)


@app.get("/api/health_summary")
async def health_summary():
    """Session 3 — standalone plain-English health summary.

    Mirrors state["health"] so tools and terminal users can curl
    it directly without parsing the much larger /api/state payload.
    """
    resp = await state()
    import json as _json
    body = resp.body
    data = _json.loads(body) if isinstance(body, (bytes, bytearray)) else body
    return JSONResponse(data.get("health") or {})


@app.get("/api/history")
async def history(days: int = 30):
    storage: Storage = app.state.storage
    scores = storage.recent_scores(days=days)
    # Build daily series
    by_day = {}
    for s in scores:
        d = s["target_date"]
        by_day.setdefault(d, {"date": d})
        by_day[d][f"err_{s['mode']}"] = s["error_f"]
    return JSONResponse({"series": sorted(by_day.values(),
                                           key=lambda x: x["date"])})


@app.get("/api/trajectory")
async def trajectory():
    """Session 7 — trajectory data for the three dashboard panels.

    Returns three datasets:
      per_source_mae    — per-source cumulative + rolling 7d MAE per
                          verified day (anchored on the LAST forecast
                          for each target_date so the per-source numbers
                          stay consistent with the ensemble final number).
      ensemble_mae      — DUAL-METRIC trajectory. For each verified day,
                          night_before AND final error are carried, plus
                          their cumulative + rolling MAE and inside-80
                          rates. The honest "forecasting vs nowcasting"
                          chart.
      lead_time_matrix  — Per-month MAE per lead-time bucket. Buckets:
                          night_before, pre_dawn (00-06), morning (06-12),
                          pre_peak (12-16), post_peak (16+).
    """
    import sqlite3
    import json
    from collections import defaultdict
    from statistics import median

    db_path = os.environ.get("AL3X_DB_PATH", "al3x.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # ----- Dataset 1: per_source_mae -----
    rows = conn.execute("""
        SELECT c.target_date, c.realized_high_f,
               (SELECT f.sources_json FROM forecasts f
                 WHERE f.target_date = c.target_date
                 ORDER BY f.id DESC LIMIT 1) AS sources_json
          FROM calibration_records c
         ORDER BY c.target_date ASC
    """).fetchall()

    per_source_errors: dict = defaultdict(list)
    for r in rows:
        if not r["sources_json"]:
            continue
        try:
            srcs = json.loads(r["sources_json"])
        except json.JSONDecodeError:
            continue
        for name, payload in srcs.items():
            if not isinstance(payload, dict):
                continue
            v = payload.get("value")
            if v is None:
                continue
            err = abs(float(r["realized_high_f"]) - float(v))
            per_source_errors[name].append((r["target_date"], err))

    per_source_mae = []
    for name, day_errors in per_source_errors.items():
        day_errors.sort()
        cumulative_sum = 0.0
        for i, (date, err) in enumerate(day_errors):
            cumulative_sum += err
            cumulative_mae = round(cumulative_sum / (i + 1), 3)
            window = day_errors[max(0, i - 6):i + 1]
            rolling_mae = round(sum(e for _, e in window) / len(window), 3)
            per_source_mae.append({
                "source": name,
                "target_date": date,
                "cumulative_mae_f": cumulative_mae,
                "rolling_7d_mae_f": rolling_mae,
                "n_days": i + 1,
            })

    # ----- Dataset 2: ensemble_mae (dual-metric) -----
    rows2 = conn.execute("""
        SELECT target_date, realized_high_f,
               forecast_p50_f, inside_80,
               night_before_p50_f, night_before_inside_80
          FROM calibration_records
         ORDER BY target_date ASC
    """).fetchall()

    def _mae(vals):
        return round(sum(vals) / len(vals), 3) if vals else None

    def _rolling(vals, k=7):
        window = vals[-k:]
        return round(sum(window) / len(window), 3) if window else None

    ensemble_mae = []
    final_errs = []
    nb_errs = []
    final_inside_count = 0
    nb_inside_count = 0
    final_total = 0
    nb_total = 0
    for r in rows2:
        truth = float(r["realized_high_f"])

        final_err = (abs(truth - float(r["forecast_p50_f"]))
                     if r["forecast_p50_f"] is not None else None)
        if final_err is not None:
            final_errs.append(final_err)
            final_total += 1
            if r["inside_80"]:
                final_inside_count += 1

        nb_err = (abs(truth - float(r["night_before_p50_f"]))
                  if r["night_before_p50_f"] is not None else None)
        if nb_err is not None:
            nb_errs.append(nb_err)
            nb_total += 1
            if r["night_before_inside_80"]:
                nb_inside_count += 1

        ensemble_mae.append({
            "target_date": r["target_date"],
            "realized_high_f": truth,
            "final_error_f": round(final_err, 3) if final_err is not None else None,
            "final_mae_cumulative_f": _mae(final_errs),
            "final_mae_rolling_7d_f": _rolling(final_errs),
            "final_inside_80_rate": (round(final_inside_count / final_total, 3)
                                     if final_total else None),
            "night_before_error_f": round(nb_err, 3) if nb_err is not None else None,
            "night_before_mae_cumulative_f": _mae(nb_errs),
            "night_before_mae_rolling_7d_f": _rolling(nb_errs),
            "night_before_inside_80_rate": (round(nb_inside_count / nb_total, 3)
                                            if nb_total else None),
        })

    # ----- Dataset 3: lead_time_matrix -----
    bucket_rows = conn.execute("""
        SELECT
          c.target_date,
          c.realized_high_f,
          f.mode,
          f.final_f,
          f.issued_at
        FROM calibration_records c
        JOIN forecasts f ON f.target_date = c.target_date
        ORDER BY c.target_date ASC, f.issued_at ASC
    """).fetchall()

    def _bucket(mode, issued_at, target_date):
        if mode == "night_before":
            return "night_before"
        try:
            issued_date = issued_at[:10]
            issued_hour = int(issued_at[11:13])
        except (ValueError, IndexError):
            return "post_peak"
        if issued_date < target_date:
            return "night_before"
        if 0 <= issued_hour < 6:
            return "pre_dawn"
        if 6 <= issued_hour < 12:
            return "morning"
        if 12 <= issued_hour < 16:
            return "pre_peak"
        return "post_peak"

    by_day_bucket: dict = defaultdict(list)
    realized: dict = {}
    for r in bucket_rows:
        td = r["target_date"]
        realized[td] = float(r["realized_high_f"])
        b = _bucket(r["mode"], r["issued_at"], td)
        by_day_bucket[(td, b)].append(float(r["final_f"]))

    by_month_bucket: dict = defaultdict(list)
    for (td, b), vals in by_day_bucket.items():
        if td not in realized:
            continue
        med = median(vals)
        err = abs(med - realized[td])
        month = td[:7]
        by_month_bucket[(month, b)].append(err)

    BUCKETS = ["night_before", "pre_dawn", "morning", "pre_peak", "post_peak"]
    months_seen = sorted({m for (m, _) in by_month_bucket.keys()})
    lead_time_matrix = []
    for month in months_seen:
        row = {"month": month}
        for b in BUCKETS:
            errs = by_month_bucket.get((month, b), [])
            row[b] = {
                "mae_f": round(sum(errs) / len(errs), 3) if errs else None,
                "n_days": len(errs),
            }
        lead_time_matrix.append(row)

    conn.close()

    return {
        "per_source_mae": per_source_mae,
        "ensemble_mae": ensemble_mae,
        "lead_time_matrix": lead_time_matrix,
        "buckets": BUCKETS,
        "headline": {
            "verified_days": len(rows2),
            "final_mae_cumulative_f": _mae(final_errs),
            "final_mae_rolling_7d_f": _rolling(final_errs),
            "night_before_mae_cumulative_f": _mae(nb_errs),
            "night_before_mae_rolling_7d_f": _rolling(nb_errs),
            "final_inside_80_rate": (round(final_inside_count / final_total, 3)
                                     if final_total else None),
            "night_before_inside_80_rate": (round(nb_inside_count / nb_total, 3)
                                            if nb_total else None),
        },
    }


@app.get("/api/nws_official")
async def nws_official():
    """Session 8 Part 2 — latest weather.gov NWS official-forecast captures.

    Used by the dashboard's NWS comparison card, sourced from the
    independent tracker that runs every 30 minutes.
    """
    import sqlite3
    from al3x.nws_official_tracker import latest_for_target, ensure_schema

    db_path = os.environ.get("AL3X_DB_PATH", str(BASE / "al3x.db"))
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)

        now = datetime.now(cfg.EASTERN)
        today_str = now.strftime("%Y-%m-%d")
        tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")

        today_nws = latest_for_target(conn, today_str)
        tomorrow_nws = latest_for_target(conn, tomorrow_str)

        capture_count_24h = conn.execute(
            """SELECT COUNT(*) FROM nws_official_forecasts
               WHERE captured_at > datetime('now', '-24 hours')"""
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "today": today_nws,
        "tomorrow": tomorrow_nws,
        "captures_last_24h": capture_count_24h,
    }


@app.post("/api/force/night_before")
async def force_night_before():
    # BUG 2 — manual API triggers must bypass the 3-run cap
    await app.state.agent.night_before_cycle(forced=True)
    return {"ok": True}


@app.post("/api/force/intraday")
async def force_intraday():
    # intraday_cycle guards on high_locked; this always runs a refresh
    await app.state.agent.intraday_cycle()
    return {"ok": True}


@app.post("/api/force/cli_check")
async def force_cli_check():
    await app.state.agent.cli_verification_cycle()
    return {"ok": True}


@app.get("/api/kalshi/state")
async def kalshi_state():
    storage: Storage = app.state.storage
    bankroll = storage.get_or_create_bankroll(cfg.KALSHI_PAPER_MODE)
    open_positions = storage.open_kalshi_positions()
    pattern_log = storage.recent_pattern_log(hours=24)
    recent_snaps = {}
    # Get the latest snapshot per market for the dashboard
    try:
        from datetime import date as _date
        today = _date.today().isoformat()
        # Pull unique tickers from recent patterns
        tickers = list({p["market_ticker"] for p in pattern_log
                        if p.get("market_ticker")})
        for t in tickers[:20]:
            snaps = storage.recent_snapshots(t, limit=1)
            if snaps:
                recent_snaps[t] = snaps[0]
    except Exception:
        pass
    return JSONResponse({
        "bankroll": bankroll,
        "open_positions": open_positions,
        "pattern_log_24h": pattern_log[:50],
        "latest_snapshots": recent_snaps,
        "paper_mode": cfg.KALSHI_PAPER_MODE,
        "live_trading": cfg.KALSHI_LIVE_TRADING,
    })


@app.post("/api/kalshi/backtest")
async def run_backtest(days: int = 30):
    """Run the historical backtest over the past `days` days.

    Returns the full summary dict including win rate, total P&L,
    ROI, per-strategy breakdown, and a go/no-go recommendation.

    This endpoint may take 30-120 seconds to complete depending on
    how many days are requested and Kalshi API response times.
    """
    try:
        from al3x.kalshi_backtest import KalshiBacktest
        storage: Storage = app.state.storage
        feed = app.state.agent._kalshi_feed
        backtest = KalshiBacktest(storage, feed)
        result = await backtest.run(
            days_back=min(days, 90),
            starting_bankroll=cfg.KALSHI_STARTING_BANKROLL,
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/kalshi/backtest/history")
async def backtest_history():
    """Return all stored backtest results from the DB."""
    storage: Storage = app.state.storage
    try:
        with storage._conn() as c:  # noqa: SLF001
            rows = c.execute(
                "SELECT run_at, target_date, strategy, simulated_side, "
                "simulated_entry_price, simulated_contracts, "
                "simulated_cost_basis, simulated_pnl, edge_cents, ev, "
                "settlement_result, cli_high_f, al3x_fair_value "
                "FROM kalshi_backtest_results "
                "ORDER BY run_at DESC, target_date DESC LIMIT 500"
            ).fetchall()
        return JSONResponse({"results": [dict(r) for r in rows]})
    except Exception as e:
        return JSONResponse({"results": [], "error": str(e)})


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("AL3X_HOST", "0.0.0.0")
    port = int(os.environ.get("AL3X_PORT", "8090"))
    uvicorn.run("app:app", host=host, port=port, reload=False, log_level="info")
