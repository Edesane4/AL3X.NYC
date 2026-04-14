"""Telegram notifier.

Sends user-friendly forecast updates and pipes Python errors/warnings from
the logger to the configured chat.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from .config import env

log = logging.getLogger("al3x.telegram")


class TelegramNotifier:
    API = "https://api.telegram.org"

    def __init__(self, token: Optional[str] = None,
                 chat_id: Optional[str] = None) -> None:
        self.token = token or env("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or env("TELEGRAM_CHAT_ID")
        self._client = httpx.AsyncClient(timeout=15.0)
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
        self._worker_task: Optional[asyncio.Task] = None
        # Circuit breaker: disable sending after repeated auth failures so we
        # don't spin into a feedback loop with the logging handler.
        self._disabled_reason: Optional[str] = None
        self._consecutive_fail = 0

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id
                    and self._disabled_reason is None)

    async def start(self) -> None:
        if not (self.token and self.chat_id):
            return
        # Verify token with getMe before starting the worker so a bad token
        # is caught immediately — no retry spam.
        try:
            r = await self._client.get(
                f"{self.API}/bot{self.token}/getMe", timeout=10,
            )
            if r.status_code == 401:
                self._disabled_reason = "401 Unauthorized (bad token)"
                log.warning("Telegram disabled: %s", self._disabled_reason)
                return
            if r.status_code >= 400:
                self._disabled_reason = f"{r.status_code} on getMe"
                log.warning("Telegram disabled: %s", self._disabled_reason)
                return
        except Exception as e:  # noqa: BLE001
            self._disabled_reason = f"getMe failed: {e}"
            log.warning("Telegram disabled: %s", self._disabled_reason)
            return
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        await self._client.aclose()

    def enqueue(self, text: str) -> None:
        """Thread- & loop-safe enqueue. Silently drops if queue is full."""
        try:
            self._queue.put_nowait(text)
        except asyncio.QueueFull:
            log.warning("telegram queue full, dropping message")

    async def send(self, text: str) -> None:
        if not self.configured:
            return
        url = f"{self.API}/bot{self.token}/sendMessage"
        # Telegram hard cap is 4096 chars
        for chunk in _chunks(text, 3800):
            try:
                r = await self._client.post(
                    url,
                    json={
                        "chat_id": self.chat_id,
                        "text": chunk,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
                if r.status_code == 401:
                    # Definitively bad creds — disable permanently this session
                    self._disabled_reason = "401 Unauthorized during send"
                    log.warning("Telegram disabled: bad token (401). "
                                "No further messages will be attempted.")
                    return
                if r.status_code >= 400:
                    self._consecutive_fail += 1
                    if self._consecutive_fail >= 5:
                        self._disabled_reason = (
                            f"{self._consecutive_fail} consecutive failures"
                        )
                        log.warning("Telegram disabled after repeated failures.")
                        return
                    log.warning("telegram error %s: %s",
                                r.status_code, r.text[:200])
                else:
                    self._consecutive_fail = 0
            except Exception as e:  # noqa: BLE001
                self._consecutive_fail += 1
                msg = str(e) or type(e).__name__
                log.warning("telegram send failed (%s): %s",
                            type(e).__name__, msg)

    async def _worker(self) -> None:
        while True:
            text = await self._queue.get()
            try:
                await self.send(text)
            except Exception as e:  # noqa: BLE001
                log.warning("telegram worker error: %s", e)
            await asyncio.sleep(0.2)  # soft rate-limit


def _chunks(text: str, size: int):
    for i in range(0, len(text), size):
        yield text[i : i + size]


class TelegramLogHandler(logging.Handler):
    """Logging handler that forwards WARNING+ records to Telegram.

    Only handles records from AL3X loggers and Python tracebacks — we don't
    want to echo routine chatter.
    """

    def __init__(self, notifier: TelegramNotifier, level: int = logging.WARNING):
        super().__init__(level=level)
        self.notifier = notifier
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Break the feedback loop: never forward logs that came from the
            # Telegram subsystem itself, or from httpx (which logs every
            # outgoing request, including ours to api.telegram.org).
            if (record.name.startswith("al3x.telegram")
                    or record.name == "httpx"
                    or record.name.startswith("httpcore")):
                return
            if not self.notifier.configured:
                return
            msg = self.format(record)
            icon = {"WARNING": "⚠️", "ERROR": "🛑",
                    "CRITICAL": "🚨"}.get(record.levelname, "ℹ️")
            body = f"{icon} <b>{record.levelname}</b>\n<code>{_esc(msg)}</code>"
            if record.exc_info:
                import traceback
                tb = "".join(traceback.format_exception(*record.exc_info))
                body += f"\n<pre>{_esc(tb[-1500:])}</pre>"
            self.notifier.enqueue(body)
        except Exception:  # noqa: BLE001
            # Never let logging break the app
            pass


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


# ---- Message formatters -----------------------------------------------------

def format_forecast_for_humans(fc: dict) -> str:
    """Render a forecast record as a friendly Telegram message."""
    mode = fc["mode"].replace("_", "-").title()
    rev = f" (revision #{fc['revision']})" if fc.get("revision") else ""
    date = fc["target_date"]
    final = round(fc["final_f"])
    unc = fc["uncertainty_f"]
    delta = fc.get("delta_prior_f")
    delta_line = ""
    if delta is not None and fc.get("revision"):
        sign = "▲" if delta > 0 else ("▼" if delta < 0 else "▶")
        delta_line = f"\n{sign} Change from last update: {delta:+.1f}°F"

    running = ""
    if fc.get("running_asos_max_f") is not None:
        running = f"\n🌡 Running Central Park max so far: {fc['running_asos_max_f']:.1f}°F"

    sources = fc.get("sources", {})
    src_lines = []
    for name, payload in sources.items():
        if payload.get("value") is None:
            continue
        src_lines.append(
            f"  • <b>{_pretty_source(name)}:</b> {payload['value']:.1f}°F "
            f"(weight {int(round(payload['weight']*100))}%)"
        )
    src_block = "\n".join(src_lines) if src_lines else "  (no sources)"

    corr = fc.get("corrections", {})
    corr_lines = []
    total = 0.0
    for name, payload in corr.items():
        d = payload.get("delta", 0)
        if d == 0 and not payload.get("reason"):
            continue
        total += d
        corr_lines.append(f"  • {_pretty_correction(name)}: {d:+.1f}°F "
                          f"— {payload.get('reason','')}")
    corr_block = "\n".join(corr_lines) if corr_lines else "  (no adjustments)"

    return (
        f"🗽 <b>AL3X.NYC forecast update</b>\n"
        f"Central Park high for <b>{date}</b>\n"
        f"Mode: {mode}{rev}\n\n"
        f"🔮 <b>Forecast: {final}°F</b> (±{unc:.1f}°F){delta_line}{running}\n\n"
        f"📊 <b>Ensemble</b>\n{src_block}\n"
        f"Raw ensemble: {fc['raw_ensemble_f']:.1f}°F\n\n"
        f"🛠 <b>NYC adjustments</b> (total {total:+.1f}°F)\n{corr_block}"
    )


def format_cli_confirmation(target_date: str, recorded_f: float,
                            best_forecast_f: float, error_f: float) -> str:
    emoji = "🎯" if abs(error_f) <= 1 else ("✅" if abs(error_f) <= 2.5 else "⚠️")
    return (
        f"{emoji} <b>CLI verified — {target_date}</b>\n"
        f"Central Park recorded high: <b>{recorded_f:.0f}°F</b>\n"
        f"Our last forecast: {best_forecast_f:.0f}°F\n"
        f"Error: {error_f:+.1f}°F"
    )


def _pretty_source(s: str) -> str:
    return {
        "hrrr": "HRRR (3 km)",
        "nws_point": "NWS Point Forecast",
        "gfs_mos": "GFS-MOS",
        "nam_mos": "NAM-MOS",
        "ecmwf": "ECMWF",
        "asos_trend": "KNYC ASOS trend",
        "nbm": "NBM",
    }.get(s, s)


def _pretty_correction(s: str) -> str:
    return {
        "sea_breeze": "Sea breeze",
        "uhi": "Urban heat island",
        "cloud_timing": "Afternoon clouds",
        "precip": "Precipitation",
        "inversion": "Morning inversion / fog",
        "spread_penalty": "Model-spread penalty",
    }.get(s, s.replace("_", " ").title())
