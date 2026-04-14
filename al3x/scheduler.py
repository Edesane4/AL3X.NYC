"""Operational schedule — orchestrates cycles for AL3X.NYC.

Runs:
  • ASOS observations + intraday revision every 15 minutes (24/7)
  • Night-Before forecast once after 6 PM local (re-computable)
  • CLI verification polling every 10 minutes 5:30-7:30 PM local
  • Weight retune weekly, bias retune biweekly
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import date, datetime, timedelta
from typing import Dict, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from . import config as cfg
from .data_sources import DataSources, high_confirmed
from .forecaster import Forecaster
from .learning import Learning
from .storage import Storage
from .telegram_bot import (TelegramNotifier, format_cli_confirmation,
                           format_forecast_for_humans)

log = logging.getLogger("al3x.scheduler")


class AgentScheduler:
    def __init__(self, storage: Storage, notifier: TelegramNotifier,
                 ai_calibrator=None) -> None:
        self.storage = storage
        self.notifier = notifier
        self.sources = DataSources()
        self.forecaster = Forecaster(storage, self.sources,
                                      ai_calibrator=ai_calibrator)
        self.learning = Learning(storage)
        self.scheduler = AsyncIOScheduler(timezone=str(cfg.EASTERN))
        # BUG 6 — up to 3 night-before runs per evening, tracked by date.
        self._night_before_runs: Dict[str, int] = {}
        self._cli_verified_for: set[str] = set()
        self._high_locked_for: set[str] = set()
        self._last_forecast_finalf: Optional[float] = None

    def start(self) -> None:
        # Intraday cycle — every 15 min
        self.scheduler.add_job(
            self._safe(self.intraday_cycle),
            IntervalTrigger(minutes=15, jitter=30),
            id="intraday",
            next_run_time=datetime.now(cfg.EASTERN) + timedelta(seconds=5),
        )
        # Hourly data refresh — we just reuse intraday cycle
        # Night-before check — every 30 min between 6 PM and midnight
        self.scheduler.add_job(
            self._safe(self.night_before_cycle),
            CronTrigger(hour="18-23", minute="0,30", timezone=str(cfg.EASTERN)),
            id="night_before",
        )
        # GAP 4 — CLI polling 5 PM – 9:59 PM Eastern every 10 min to catch
        # late postings on complex-weather evenings. Already-verified dates
        # are skipped via _cli_verified_for so extra polls are cheap.
        self.scheduler.add_job(
            self._safe(self.cli_verification_cycle),
            CronTrigger(hour="17-21", minute="0,10,20,30,40,50",
                        timezone=str(cfg.EASTERN)),
            id="cli_verify",
        )
        # Weight retune — weekly Sunday 02:00 Eastern
        self.scheduler.add_job(
            self._safe(self.retune_weights),
            CronTrigger(day_of_week="sun", hour=2, minute=0,
                        timezone=str(cfg.EASTERN)),
            id="retune_weights",
        )
        # Bias retune — every 14 days at 02:15
        self.scheduler.add_job(
            self._safe(self.retune_biases),
            CronTrigger(day="1,15", hour=2, minute=15,
                        timezone=str(cfg.EASTERN)),
            id="retune_biases",
        )
        self.scheduler.start()
        log.info("AL3X.NYC scheduler started (all jobs armed).")

    async def stop(self) -> None:
        self.scheduler.shutdown(wait=False)
        await self.sources.close()

    # ---- Job wrappers ---------------------------------------------------

    def _safe(self, coro_fn):
        async def runner():
            try:
                await coro_fn()
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                log.error("Job %s failed: %s\n%s", coro_fn.__name__, e, tb)
        return runner

    # ---- Intraday -------------------------------------------------------
    async def intraday_cycle(self) -> None:
        now = datetime.now(cfg.EASTERN)
        today = now.date()
        today_str = today.isoformat()

        # If today's high is already locked, just refresh observations
        if today_str in self._high_locked_for:
            obs = await self.sources.asos_observations()
            for o in obs:
                if o["observed_at"][:10] == today_str:
                    self.storage.save_observation(o)
            return

        fc = await self.forecaster.produce("intraday", today)
        fid = self.storage.save_forecast(fc)
        fc["id"] = fid

        log.info("Intraday forecast %s: %.1f°F (raw %.2f, Δ %s, asos_max %s)",
                 fc["target_date"], fc["final_f"], fc["raw_ensemble_f"],
                 fc.get("delta_prior_f"), fc.get("running_asos_max_f"))

        # Notify only when it's meaningfully different
        delta = fc.get("delta_prior_f")
        should_notify = (delta is None or abs(delta) >= 1.0
                         or fc["revision"] in (1, 5, 10, 20))
        if should_notify and self.notifier.configured:
            self.notifier.enqueue(format_forecast_for_humans(fc))

        # High confirmation lock
        obs_today = self.storage.observations_today(today_str)
        if high_confirmed(obs_today):
            self._high_locked_for.add(today_str)
            log.info("High confirmed for %s at %.1f°F — locking forecast.",
                     today_str, fc.get("running_asos_max_f") or fc["final_f"])
            if self.notifier.configured:
                self.notifier.enqueue(
                    f"🔒 <b>High confirmed for {today_str}</b>\n"
                    f"Central Park ran up to "
                    f"{fc.get('running_asos_max_f', fc['final_f']):.1f}°F and "
                    "has been falling for 2+ hours. Forecast locked."
                )

    # ---- Night-before ---------------------------------------------------
    async def night_before_cycle(self, forced: bool = False) -> None:
        """BUG 6 — allow up to 3 night-before runs per evening:
          run #1: any time after 6 PM
          run #2: after 8:30 PM Eastern (18Z ECMWF + later MOS in)
          run #3: only after CLI has verified for today (fresh weights)
        The forced=True bypass is used by API /force endpoints.
        """
        now = datetime.now(cfg.EASTERN)
        tomorrow = (now + timedelta(days=1)).date()
        key = tomorrow.isoformat()
        # BUG 7 — prune entries older than 2 days so this dict never grows
        cutoff = (now - timedelta(days=2)).date().isoformat()
        self._night_before_runs = {
            k: v for k, v in self._night_before_runs.items() if k >= cutoff
        }
        runs_done = self._night_before_runs.get(key, 0)

        if not forced:
            if runs_done >= 3:
                return
            today_key = now.date().isoformat()
            cli_ready = today_key in self._cli_verified_for
            # Gate runs by time-of-day / CLI readiness
            if runs_done == 0:
                if now.hour < 18:
                    return
            elif runs_done == 1:
                if now.hour < 20 or (now.hour == 20 and now.minute < 30):
                    return
            elif runs_done == 2:
                if not cli_ready:
                    return

        run_number = runs_done + 1
        fc = await self.forecaster.produce("night_before", tomorrow)
        fc["extras"] = dict(fc.get("extras", {}))
        fc["extras"]["night_before_run"] = run_number
        fid = self.storage.save_forecast(fc)
        fc["id"] = fid
        self._night_before_runs[key] = run_number
        log.info("Night-before forecast (run %d/3) %s: %.1f°F",
                 run_number, fc["target_date"], fc["final_f"])
        if self.notifier.configured:
            self.notifier.enqueue(format_forecast_for_humans(fc))

    # ---- CLI verification ----------------------------------------------
    async def cli_verification_cycle(self) -> None:
        now = datetime.now(cfg.EASTERN)
        today = now.date()
        today_str = today.isoformat()
        if today_str in self._cli_verified_for:
            return
        cli = await self.sources.cli_latest()
        if not cli:
            return
        # Guard: CLI must be for today
        if cli.get("target_date") and cli["target_date"] != today_str:
            return
        self.storage.save_cli_truth(today_str, cli["recorded_high_f"],
                                    cli["posted_at"], cli["raw_text"])
        self._cli_verified_for.add(today_str)

        # Score forecasts
        result = self.learning.score_day(today_str, cli["recorded_high_f"])
        latest = self.storage.latest_forecast(today_str)
        best_f = latest["final_f"] if latest else cli["recorded_high_f"]
        err = cli["recorded_high_f"] - best_f
        log.info("CLI verified %s: %.1f°F (err %+.2f, scored=%s)",
                 today_str, cli["recorded_high_f"], err, result.get("scored"))
        if self.notifier.configured:
            self.notifier.enqueue(
                format_cli_confirmation(today_str, cli["recorded_high_f"],
                                         best_f, err)
            )
        # Trigger night-before for tomorrow now that truth is in (run #3
        # gate is "CLI verified", which is now true).
        await self.night_before_cycle()

    # ---- Auto-tune ------------------------------------------------------
    async def retune_weights(self) -> None:
        res = self.learning.retune_weights()
        log.info("Weekly weight retune: %s", res)
        if res.get("ok") and self.notifier.configured:
            self.notifier.enqueue(
                f"⚖️ <b>Weights retuned (7-day window)</b>\n"
                f"Source MAE: <code>{res['mae']}</code>\n"
                f"Changes: <code>{res['changes']}</code>"
            )

    async def retune_biases(self) -> None:
        res = self.learning.retune_biases()
        log.info("Biweekly bias retune: %s", res)
        if res.get("ok") and res.get("adjustments") and self.notifier.configured:
            lines = []
            for k, v in res["adjustments"].items():
                lines.append(f"  • {k}: {v['old']:+.1f} → {v['new']:+.1f}°F "
                             f"({v['reason']})")
            self.notifier.enqueue(
                "🧠 <b>Bias ledger auto-tuned (14-day window)</b>\n"
                + "\n".join(lines)
            )
