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
from al3x.logging_setup import configure as configure_logging
from al3x.scheduler import AgentScheduler
from al3x.storage import Storage
from al3x.telegram_bot import TelegramNotifier, format_forecast_for_humans

log = logging.getLogger("al3x.app")

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"


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

    # Running max from obs
    running_max_f = None
    if obs:
        temps = [o["temperature_f"] for o in obs
                 if o.get("temperature_f") is not None]
        running_max_f = max(temps) if temps else None

    return JSONResponse({
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
        "ai_enabled": bool(getattr(app.state, "ai_calibrator", None)
                           and app.state.ai_calibrator.enabled),
        "weights_override": storage.get_weights(),
        "biases_override": storage.get_biases(),
    })


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


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("AL3X_HOST", "0.0.0.0")
    port = int(os.environ.get("AL3X_PORT", "8090"))
    uvicorn.run("app:app", host=host, port=port, reload=False, log_level="info")
