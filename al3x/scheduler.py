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
import os
import traceback
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from . import config as cfg
from .data_sources import DataSources, high_confirmed
from .forecaster import Forecaster
from .learning import Learning
from .sheets_exporter import SheetsExporter
from .storage import Storage
from .telegram_bot import (TelegramNotifier, format_cli_confirmation,
                           format_forecast_for_humans)
from .kalshi_engine import KalshiEngine
from .kalshi_feed import KalshiFeed


def _regime_label_from_flags(regime: Dict[str, Any]) -> str:
    """Same regime-label taxonomy as learning._regime_label, duplicated
    here to avoid a cross-module import cycle."""
    if regime.get("precip_heavy"):
        return "rain_heavy"
    if regime.get("any_precip_peak"):
        return "rain_light"
    if regime.get("sea_breeze_full") or regime.get("sea_breeze_shift"):
        return "sea_breeze"
    if regime.get("calm_clear"):
        return "clear_calm"
    if regime.get("sustained_windy"):
        return "windy"
    return "mixed"

log = logging.getLogger("al3x.scheduler")


class AgentScheduler:
    def __init__(self, storage: Storage, notifier: TelegramNotifier,
                 ai_calibrator=None) -> None:
        self.storage = storage
        self.notifier = notifier
        self.sources = DataSources()
        self.forecaster = Forecaster(storage, self.sources,
                                      ai_calibrator=ai_calibrator,
                                      notifier=notifier)
        self.learning = Learning(storage)
        self.scheduler = AsyncIOScheduler(timezone=str(cfg.EASTERN))
        # FIX 5 — run counts now live in the DB (see
        # storage.night_before_runs). The in-memory dict was lost on
        # every restart, producing phantom repeat runs.
        self._cli_verified_for: set[str] = set()
        # BUGS 1+3 — survive restarts without re-scoring already-verified
        # days. Seed the in-memory guard from the DB at boot.
        try:
            for r in storage.get_cli_truth_recent(days=2):
                self._cli_verified_for.add(r["target_date"])
        except Exception:
            pass
        self._high_locked_for: set[str] = set()
        self._last_forecast_finalf: Optional[float] = None
        # Quant Upgrade 3 — regime-shift detection state
        self._last_regime: Optional[Dict[str, Any]] = None
        self._last_spread: Optional[float] = None
        # Seed regime-shift state from last stored forecast so the first
        # post-restart cycle has a prior to compare against.
        try:
            last_fc = storage.latest_forecast()
            if last_fc and last_fc.get("mode") == "intraday":
                seed_extras = last_fc.get("extras") or {}
                self._last_regime = seed_extras.get("regime") or None
                self._last_spread = seed_extras.get("spread_f")
        except Exception:
            pass
        # Google Sheets exporter (optional)
        spreadsheet_id = os.environ.get("GOOGLE_SHEETS_SPREADSHEET_ID", "")
        creds_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        self.sheets: Optional[SheetsExporter] = (
            SheetsExporter(spreadsheet_id, creds_path)
            if spreadsheet_id and creds_path else None
        )

        # Kalshi market intelligence engine
        self._kalshi_feed = KalshiFeed()
        self._kalshi_engine = KalshiEngine(storage, self._kalshi_feed)

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
        # NWS OKX posts NYC CLI the same evening it covers (~5:30–7:30 PM
        # Eastern). Poll every 10 min across the 17:00-21:59 window to
        # catch late postings. Already-verified dates are skipped via
        # _cli_verified_for so extra polls are cheap.
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
        # Kalshi orderbook scan — every 30 seconds
        self.scheduler.add_job(
            self._safe(self.kalshi_scan_cycle),
            IntervalTrigger(seconds=cfg.KALSHI_FEED_INTERVAL_SECONDS, jitter=5),
            id="kalshi_scan",
            next_run_time=datetime.now(cfg.EASTERN) + timedelta(seconds=15),
        )
        # Session 2 G12 — daily morning brief at 7:00 AM Eastern.
        # Sends the short summary via Telegram and writes the longer
        # markdown version to the user's Obsidian vault when it's
        # configured (OBSIDIAN_VAULT_PATH).
        self.scheduler.add_job(
            self._safe(self.morning_brief_cycle),
            CronTrigger(hour=7, minute=0, timezone=str(cfg.EASTERN)),
            id="morning_brief",
            replace_existing=True,
        )
        # Daily settlement — runs at 9:30 PM Eastern after markets close
        self.scheduler.add_job(
            self._safe(self.settle_positions_cycle),
            CronTrigger(hour=21, minute=30, timezone=str(cfg.EASTERN)),
            id="settle_positions",
        )
        # Reset daily P&L at midnight Eastern
        self.scheduler.add_job(
            self._safe(self.reset_daily_pnl),
            CronTrigger(hour=0, minute=1, timezone=str(cfg.EASTERN)),
            id="reset_daily_pnl",
        )
        # Session 8 — independent NWS official-forecast tracker. Hits
        # weather.gov's gridpoint forecast endpoint every 30 minutes
        # and stores results in nws_official_forecasts. Completely
        # separate from the existing 'nws_point' source in
        # data_sources.py; this is the apples-to-apples comparison
        # baseline for AL3X-vs-NWS analysis.
        self.scheduler.add_job(
            self._safe(self.nws_official_tracker_cycle),
            IntervalTrigger(minutes=30, jitter=60),
            id="nws_official_tracker",
            next_run_time=datetime.now(cfg.EASTERN) + timedelta(seconds=30),
        )
        self.scheduler.start()
        log.info("AL3X.NYC scheduler started (all jobs armed).")

    async def stop(self) -> None:
        self.scheduler.shutdown(wait=False)
        await self.sources.close()
        try:
            await self._kalshi_feed.close()
        except Exception:
            pass

    # ---- Job wrappers ---------------------------------------------------

    def _safe(self, coro_fn):
        async def runner():
            try:
                await coro_fn()
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                log.error("Job %s failed: %s\n%s", coro_fn.__name__, e, tb)
        return runner

    # ---- Regime-shift detection (Quant Upgrade 3) ----------------------
    def detect_regime_shift(self, current_regime: Dict[str, Any],
                             current_spread: Optional[float],
                             target_date: str) -> List[str]:
        """FIX 2 — tightened thresholds + rate limit.

        Evidence: 41 shifts in 8 days (5.1/day avg; 13 on one day). 51%
        were ``spread`` shifts with empty prev/new_state — noise. Each
        firing triggers an adaptive weight reset at blend_alpha=0.50,
        thrashing learned weights and correlating with the degraded
        3-12h lead MAE. Changes:
          * spread threshold tightened from >3.0°F to >5.0°F
          * skip a candidate shift whose storage-side prev_state and
            new_state would both be empty strings (the noise pattern)
          * rate-limit at 3 shifts/day (skip further detection entirely)
        """
        try:
            already = self.storage.count_regime_shifts_for_date(target_date)
        except Exception:  # noqa: BLE001
            already = 0
        if already >= 3:
            log.info("Regime-shift rate limit: %d shifts already recorded "
                     "for %s; skipping detection", already, target_date)
            return []

        # Mirror intraday_cycle's key_map so we can pre-check whether
        # storage would persist empty prev/new_state strings.
        key_map = {
            "precip_onset": "any_precip_peak",
            "sea_breeze": "sea_breeze_shift",
            "wind": "sustained_windy",
        }

        def _states_empty(stype: str) -> bool:
            ck = key_map.get(stype, stype)
            prev_s = str((self._last_regime or {}).get(ck, ""))
            new_s = str(current_regime.get(ck, ""))
            return not prev_s and not new_s

        shift_flags: List[str] = []
        prev = self._last_regime
        if prev is not None:
            if (prev.get("sea_breeze_shift")
                    != current_regime.get("sea_breeze_shift")):
                if not _states_empty("sea_breeze"):
                    shift_flags.append("sea_breeze")
            if (not prev.get("any_precip_peak")
                    and current_regime.get("any_precip_peak")):
                if not _states_empty("precip_onset"):
                    shift_flags.append("precip_onset")
            if (prev.get("sustained_windy")
                    != current_regime.get("sustained_windy")):
                if not _states_empty("wind"):
                    shift_flags.append("wind")
            if (self._last_spread is not None and current_spread is not None
                    and abs(current_spread - self._last_spread) > 5.0):
                if not _states_empty("spread"):
                    shift_flags.append("spread")
        return shift_flags

    async def _adaptive_weight_reset(self, shift_flags: List[str]) -> None:
        """Kick a more aggressive weight retune (blend_alpha=0.50) after
        a detected regime change so we catch the new regime faster."""
        try:
            res = self.learning.retune_weights(blend_alpha=0.50)
            log.info("Adaptive weight reset (alpha=0.50, shifts=%s): %s",
                     shift_flags, res)
        except Exception as e:  # noqa: BLE001
            log.warning("adaptive weight reset failed: %s", e)

    def _get_night_before_for_date(self, date_str: str) -> Optional[Dict[str, Any]]:
        for f in self.storage.forecasts_for_date(date_str):
            if f["mode"] == "night_before":
                return f
        return None

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

        # Persist QRF prediction (Quant Upgrade 1)
        extras = fc.get("extras") or {}
        qrf = extras.get("qrf")
        if qrf is not None:
            try:
                self.storage.save_qrf_prediction(
                    fid,
                    p10=qrf.get("p10_delta", 0.0),
                    p50=qrf.get("p50_delta", 0.0),
                    p90=qrf.get("p90_delta", 0.0),
                    interval_width=qrf.get("interval_width"),
                    n_training=qrf.get("n_training"),
                )
            except Exception as e:  # noqa: BLE001
                log.info("qrf persistence failed: %s", e)

        log.info("Intraday forecast %s: %.1f°F (raw %.2f, Δ %s, asos_max %s)",
                 fc["target_date"], fc["final_f"], fc["raw_ensemble_f"],
                 fc.get("delta_prior_f"), fc.get("running_asos_max_f"))

        # Regime-shift detection (Quant Upgrade 3) — compare current
        # regime/spread to the last cycle's. Fires only when we have a
        # prior regime to compare against.
        current_regime = extras.get("regime") or {}
        current_spread = extras.get("spread_f")
        shift_flags = self.detect_regime_shift(
            current_regime, current_spread, today_str,
        )
        if shift_flags:
            for stype in shift_flags:
                key_map = {
                    "precip_onset": "any_precip_peak",
                    "sea_breeze": "sea_breeze_shift",
                    "wind": "sustained_windy",
                }
                check_key = key_map.get(stype, stype)
                self.storage.save_regime_shift({
                    "detected_at": datetime.now(cfg.EASTERN).isoformat(),
                    "target_date": today_str,
                    "shift_type": stype,
                    "prev_state": str((self._last_regime or {}).get(check_key, "")),
                    "new_state": str(current_regime.get(check_key, "")),
                    "weight_reset_applied": 1,
                })
            await self._adaptive_weight_reset(shift_flags)
            log.info("Regime shift detected on %s: %s — adaptive weight "
                     "reset fired", today_str, shift_flags)
            if self.notifier.configured:
                self.notifier.enqueue(
                    f"⚡ <b>Regime shift detected</b>: "
                    f"{', '.join(shift_flags)}\n"
                    f"Adaptive weight reset applied for {today_str}"
                )
        self._last_regime = current_regime
        self._last_spread = current_spread

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
        # FIX 5 — read run count from DB (not in-memory), so restarts
        # between 6-11 PM don't re-emit night-before runs.
        runs_done = self.storage.get_night_before_run_count(key)

        if not forced:
            if runs_done >= 3:
                return
            # NWS OKX posts today's CLI same evening (~6:30 PM Eastern),
            # so by the time run #3 fires, today's CLI should already be
            # in the verified set — that's the "fresh truth available"
            # signal for aggressive night-before tuning.
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

        run_number = self.storage.increment_night_before_run_count(key)
        fc = await self.forecaster.produce("night_before", tomorrow)
        fc["extras"] = dict(fc.get("extras", {}))
        fc["extras"]["night_before_run"] = run_number
        fid = self.storage.save_forecast(fc)
        fc["id"] = fid
        # Persist QRF prediction for night-before too
        nb_qrf = (fc.get("extras") or {}).get("qrf")
        if nb_qrf is not None:
            try:
                self.storage.save_qrf_prediction(
                    fid,
                    p10=nb_qrf.get("p10_delta", 0.0),
                    p50=nb_qrf.get("p50_delta", 0.0),
                    p90=nb_qrf.get("p90_delta", 0.0),
                    interval_width=nb_qrf.get("interval_width"),
                    n_training=nb_qrf.get("n_training"),
                )
            except Exception as e:  # noqa: BLE001
                log.info("qrf persistence failed: %s", e)
        log.info("Night-before forecast (run %d/3) %s: %.1f°F",
                 run_number, fc["target_date"], fc["final_f"])
        if self.notifier.configured:
            self.notifier.enqueue(format_forecast_for_humans(fc))
        # Sheets: append a forecast-log row
        if self.sheets and self.sheets.enabled:
            try:
                await asyncio.to_thread(self.sheets.push_forecast_log, fc)
            except Exception as e:  # noqa: BLE001
                log.warning("Sheets forecast log push failed: %s", e)

    # ---- CLI verification ----------------------------------------------
    async def cli_verification_cycle(self) -> None:
        """NWS OKX posts NYC CLI the same evening it covers
        (~5:30-7:30 PM Eastern). The CLI's own parsed valid date is
        always authoritative.
        """
        now = datetime.now(cfg.EASTERN)
        today_str = now.date().isoformat()

        cli = await self.sources.cli_latest()
        if not cli:
            return

        # The CLI's own parsed valid date is always authoritative. Fall
        # back to today only if parsing failed (not yesterday).
        target_date = cli.get("target_date") or today_str

        if target_date in self._cli_verified_for:
            return

        self.storage.save_cli_truth(target_date, cli["recorded_high_f"],
                                    cli["posted_at"], cli["raw_text"])
        self._cli_verified_for.add(target_date)

        # Session 6 Part 3 — record whether realized truth fell inside
        # the agent's stated 80% band. Observability only; never block
        # production verification on calibration recording errors.
        try:
            from al3x.calibration import record_calibration
            with self.storage._conn() as cal_conn:
                record_calibration(cal_conn, target_date,
                                   float(cli["recorded_high_f"]))
        except Exception as e:  # noqa: BLE001
            log.warning("calibration recording failed for %s: %s",
                        target_date, e)

        # Score the forecasts that were issued for this target date
        result = self.learning.score_day(target_date, cli["recorded_high_f"])
        latest = self.storage.latest_forecast(target_date)
        best_f = latest["final_f"] if latest else cli["recorded_high_f"]
        err = cli["recorded_high_f"] - best_f
        log.info("CLI verified %s: %.1f°F (err %+.2f, scored=%s)",
                 target_date, cli["recorded_high_f"], err,
                 result.get("scored"))
        if self.notifier.configured:
            self.notifier.enqueue(
                format_cli_confirmation(target_date, cli["recorded_high_f"],
                                         best_f, err)
            )

        # Sheets: append a daily-result row with all the relevant fields
        if self.sheets and self.sheets.enabled:
            try:
                night_fc = self._get_night_before_for_date(target_date)
                extras = (latest.get("extras") or {}) if latest else {}
                regime = extras.get("regime") or {}
                sources = (latest.get("sources") or {}) if latest else {}
                qrf = extras.get("qrf") or {}
                analog = extras.get("analog") or {}
                ai_cal = extras.get("ai_calibration") or {}
                bma = extras.get("bma")
                data = {
                    "night_before_f": (night_fc.get("final_f")
                                        if night_fc else ""),
                    "final_intraday_f": (latest.get("final_f")
                                          if latest else ""),
                    "cli_f": cli["recorded_high_f"],
                    "nb_error": (round(cli["recorded_high_f"]
                                        - night_fc["final_f"], 2)
                                 if night_fc else ""),
                    "final_error": round(err, 2),
                    "regime": _regime_label_from_flags(regime),
                    "sea_breeze": bool(regime.get("sea_breeze_shift")
                                       or regime.get("sea_breeze_full")),
                    "precip": bool(regime.get("any_precip_peak")
                                   or regime.get("precip_heavy")),
                    "hrrr_f": (sources.get("hrrr") or {}).get("value", ""),
                    "ecmwf_f": (sources.get("ecmwf") or {}).get("value", ""),
                    "gfs_mos_f": (sources.get("gfs_mos") or {}).get(
                        "value", ""),
                    "bma_f": (bma.get("bma_forecast_f") if bma else ""),
                    "analog_bias": analog.get("analog_bias_f", ""),
                    "ai_delta": ai_cal.get("delta_f", ""),
                    "qrf_p10": qrf.get("p10_delta", ""),
                    "qrf_p90": qrf.get("p90_delta", ""),
                    "uncertainty": (latest.get("uncertainty_f")
                                     if latest else ""),
                    "spread": extras.get("spread_f", ""),
                }
                await asyncio.to_thread(
                    self.sheets.push_daily_result, target_date, data,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Sheets daily push failed: %s", e)

        # Trigger night-before for tomorrow. The night-before run-3 gate
        # is "today_local is CLI-verified" — which only matters if we
        # just verified today (unlikely, but harmless either way).
        await self.night_before_cycle()

    # ---- Auto-tune ------------------------------------------------------
    async def retune_weights(self) -> None:
        res = self.learning.retune_weights()
        log.info("Weekly weight retune: %s", res)

        # Session 4 Part 4 — BMA-learned-bias-vs-actual-7d-mean-error log
        # line. Without this, we can't track whether the +5°F phantom
        # Kalman bias BMA learned (per morning.txt 2026-04-26) is
        # shrinking. One greppable line per source per weekly retune is
        # enough to build the trend.
        try:
            last_fc = self.storage.latest_forecast()
            bma = ((last_fc.get("extras") or {}).get("bma")
                   if last_fc else None) or {}
            bma_biases = bma.get("source_biases") or {}

            scores = self.storage.recent_scores(days=7)
            by_source_err: dict = {}
            for s in scores:
                truth = self.storage.get_cli_truth(s["target_date"])
                if not truth:
                    continue
                fcs = self.storage.forecasts_for_date(s["target_date"])
                if not fcs:
                    continue
                fc = fcs[-1]
                cli = float(truth["recorded_high_f"])
                for name, payload in (fc.get("sources") or {}).items():
                    v = payload.get("value")
                    if v is None:
                        continue
                    by_source_err.setdefault(name, []).append(cli - float(v))

            actual_mean_err = {k: round(sum(v) / len(v), 2)
                                for k, v in by_source_err.items() if v}

            for src in sorted(set(bma_biases) | set(actual_mean_err)):
                bma_b = bma_biases.get(src)
                act_b = actual_mean_err.get(src)
                n_actual = len(by_source_err.get(src) or [])
                if bma_b is None and act_b is None:
                    continue
                log.info(
                    "BMA_BIAS_TREND src=%s bma_learned=%s actual_7d=%s n=%d",
                    src,
                    f"{bma_b:+.2f}" if bma_b is not None else "n/a",
                    f"{act_b:+.2f}" if act_b is not None else "n/a",
                    n_actual,
                )
        except Exception as e:  # noqa: BLE001
            log.info("BMA bias trend log failed: %s", e)

        # FIX 5 — piggy-back on the weekly cron to prune old
        # night_before_runs rows (keep 7 days).
        try:
            pruned = self.storage.purge_old_night_before_runs(days=7)
            if pruned:
                log.info("Pruned %d old night_before_runs rows", pruned)
        except Exception as e:  # noqa: BLE001
            log.info("night_before_runs prune failed: %s", e)
        if res.get("ok") and self.notifier.configured:
            self.notifier.enqueue(
                f"⚖️ <b>Weights retuned (7-day window)</b>\n"
                f"Source MAE: <code>{res['mae']}</code>\n"
                f"Changes: <code>{res['changes']}</code>"
            )
        # Sheets: weekly performance snapshot
        if self.sheets and self.sheets.enabled:
            try:
                attribution = self.learning.attribution_stats(days=30)
                stats = self.learning.headline_stats()
                await asyncio.to_thread(
                    self.sheets.push_performance_summary, stats, attribution,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Sheets performance push failed: %s", e)

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

    async def kalshi_scan_cycle(self) -> None:
        """30-second Kalshi intelligence scan.

        Passes the latest AL3X forecast to the engine before each scan
        so fair values are always based on the freshest model output.
        """
        # Session 2 G8 — short-circuit when credentials are unconfigured
        # so we don't spam 401 Unauthorized log lines every 30 seconds.
        # The scheduled job stays registered; it emits at debug level
        # instead of tripping the Kalshi API. Once KALSHI_API_KEY_ID +
        # KALSHI_PRIVATE_KEY_PATH are set in .env, the job resumes.
        if not (os.getenv("KALSHI_API_KEY_ID")
                and os.getenv("KALSHI_PRIVATE_KEY_PATH")):
            log.debug("Kalshi scan skipped — no credentials configured")
            return

        now = datetime.now(cfg.EASTERN)
        today_str = now.date().isoformat()

        # Feed latest AL3X forecast into the engine
        latest_fc = self.storage.latest_forecast(today_str)
        if latest_fc:
            self._kalshi_engine.update_forecast(latest_fc)

        try:
            summary = await self._kalshi_engine.run_cycle(today_str)
            if summary.get("trades_placed", 0) > 0:
                log.info("Kalshi scan: %d markets, %d patterns, "
                         "%d decisions, %d trades placed",
                         summary["markets_scanned"],
                         summary["patterns_detected"],
                         summary["decisions_made"],
                         summary["trades_placed"])
                if self.notifier.configured:
                    bankroll = self.storage.get_or_create_bankroll(
                        cfg.KALSHI_PAPER_MODE)
                    self.notifier.enqueue(
                        f"📈 <b>Kalshi trade(s) logged</b>\n"
                        f"{summary['trades_placed']} position(s) opened\n"
                        f"Bankroll: ${bankroll['current_bankroll']:.2f} | "
                        f"Deployed: ${bankroll['total_deployed']:.2f}"
                    )
        except Exception as e:  # noqa: BLE001
            log.info("Kalshi scan cycle error: %s", e)

    async def settle_positions_cycle(self) -> None:
        """Settle all open Kalshi positions against today's CLI truth."""
        now = datetime.now(cfg.EASTERN)
        today_str = now.date().isoformat()
        try:
            result = await self._kalshi_engine.settle_positions(today_str)
            log.info("Settlement cycle: %d positions settled, "
                     "total P&L %+.4f, errors=%s",
                     result["settled"], result["total_pnl"],
                     result["errors"])
            if result["settled"] > 0 and self.notifier.configured:
                bankroll = self.storage.get_or_create_bankroll(
                    cfg.KALSHI_PAPER_MODE)
                self.notifier.enqueue(
                    f"💰 <b>Settlement complete — {today_str}</b>\n"
                    f"Positions settled: {result['settled']}\n"
                    f"Session P&L: {result['total_pnl']:+.2f}\n"
                    f"Bankroll: ${bankroll['current_bankroll']:.2f}"
                )
        except Exception as e:  # noqa: BLE001
            log.warning("Settlement cycle error: %s", e)

    async def morning_brief_cycle(self) -> None:
        """Session 2 G12 — generate the morning brief and dispatch it.

        Telegram gets the short summary. If ``OBSIDIAN_VAULT_PATH`` is
        set and points to an existing vault, the longer markdown note
        is written under ``05-Daily Check-ins/YYYY-MM-DD.md``.
        """
        try:
            from .morning_brief import generate_morning_brief
            from .obsidian_writer import write_daily_checkin
        except Exception as e:  # noqa: BLE001
            log.warning("morning brief imports failed: %s", e)
            return
        try:
            content = generate_morning_brief(self.storage)
        except Exception as e:  # noqa: BLE001
            log.warning("morning brief generation failed: %s", e)
            return
        if self.notifier and self.notifier.configured:
            self.notifier.enqueue(content["telegram"])
        try:
            write_daily_checkin(content["obsidian"], content["target_date"])
        except Exception as e:  # noqa: BLE001
            log.info("obsidian daily check-in skipped: %s", e)

    async def nws_official_tracker_cycle(self) -> None:
        """Session 8 — fetch weather.gov NWS forecast and store in
        nws_official_forecasts. Independent of forecast cycle.
        """
        try:
            import httpx
            from .nws_official_tracker import fetch_and_store
            with self.storage._conn() as conn:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    result = await fetch_and_store(conn, client)
            if result.get("status") != "ok":
                log.info("nws_official_tracker: %s", result)
        except Exception as e:  # noqa: BLE001
            log.warning("nws_official_tracker cycle error: %s", e)

    async def reset_daily_pnl(self) -> None:
        """Reset daily_pnl to 0.0 at midnight so CB1 does not
        permanently halt trading after a single bad day."""
        try:
            self.storage.update_bankroll(
                cfg.KALSHI_PAPER_MODE, {"daily_pnl": 0.0})
            log.info("Daily P&L counter reset at midnight.")
        except Exception as e:  # noqa: BLE001
            log.warning("daily_pnl reset failed: %s", e)
