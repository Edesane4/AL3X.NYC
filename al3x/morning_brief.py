"""Session 2 G13 — morning brief content generator.

Produces a dict with two fully-formatted strings:

  ``telegram``:  short, emoji-friendly; ≤500 chars
  ``obsidian``:  longer markdown for the daily check-in note

Lives in the ``al3x`` package so the scheduler can import it directly.
Everything here is read-only: it queries the live DB but never writes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from . import config as cfg

log = logging.getLogger("al3x.morning_brief")


def _count_cap_fires_from_log(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    count = 0
    try:
        with log_path.open("r", errors="replace") as f:
            for line in f:
                if "cap fired" in line.lower():
                    count += 1
    except OSError:
        return None
    return count


def _count_regime_shifts(storage, target_date: str) -> int:
    """Read-only: count regime shifts stored for a given date. Prefers
    the Session 1 storage helper; falls back to a fresh read-only
    SELECT if the helper is unavailable (e.g. mocked storage in tests).
    """
    counter = getattr(storage, "count_regime_shifts_for_date", None)
    if callable(counter):
        try:
            return int(counter(target_date))
        except Exception:  # noqa: BLE001
            pass
    try:
        with storage._conn() as c:  # noqa: SLF001
            row = c.execute(
                "SELECT COUNT(*) AS n FROM regime_shifts "
                "WHERE target_date = ?",
                (target_date,),
            ).fetchone()
            return int(row["n"]) if row else 0
    except Exception:  # noqa: BLE001
        return 0


def generate_morning_brief(storage,
                             log_path: Path | None = None) -> Dict[str, Any]:
    """Build Telegram + Obsidian content. ``log_path`` defaults to
    ``./al3x.log`` in the CWD; override for tests."""
    today = datetime.now(cfg.EASTERN).date()
    yesterday = today - timedelta(days=1)

    yday_cli = storage.get_cli_truth(yesterday.isoformat())
    yday_forecast = storage.latest_forecast(yesterday.isoformat())
    today_forecast = storage.latest_forecast(today.isoformat())

    yday_score: Dict[str, float] | None = None
    if yday_cli and yday_forecast:
        yday_score = {
            "actual": float(yday_cli["recorded_high_f"]),
            "forecast": float(yday_forecast["final_f"]),
            "error": float(yday_forecast["final_f"])
            - float(yday_cli["recorded_high_f"]),
        }

    cap_fires = _count_cap_fires_from_log(log_path or Path("al3x.log"))
    shifts = _count_regime_shifts(storage, yesterday.isoformat())

    # --- Telegram (short, emoji-friendly, plain text) ---
    tg_lines = [
        f"☀️ AL3X.NYC morning brief — {today.isoformat()}",
        "",
    ]
    if yday_score is not None:
        sign = "✓" if abs(yday_score["error"]) < 2.0 else "⚠"
        tg_lines.append(
            f"{sign} Yesterday: forecast {yday_score['forecast']:.1f}°F, "
            f"actual {yday_score['actual']:.1f}°F "
            f"(error {yday_score['error']:+.1f}°F)"
        )
    if today_forecast is not None:
        tg_lines.append(
            f"📡 Today: {today_forecast['final_f']:.1f}°F "
            f"(±{today_forecast['uncertainty_f']:.1f}°F)"
        )
    tg_lines.append("")
    cap_line = (f"Cap fires overnight: {cap_fires}"
                if cap_fires is not None else "Cap fires overnight: N/A")
    tg_lines.append(cap_line)
    tg_lines.append(f"Regime shifts yesterday: {shifts}")
    telegram_content = "\n".join(tg_lines)

    # --- Obsidian (markdown, longer) ---
    ob_lines = [
        f"# {today.isoformat()}",
        "",
        "## Overnight Summary",
        "",
    ]
    if yday_score is not None:
        ob_lines.extend([
            "**Yesterday's scoring:**",
            f"- Forecast: {yday_score['forecast']:.1f}°F",
            f"- Actual:   {yday_score['actual']:.1f}°F",
            f"- Error:    {yday_score['error']:+.1f}°F",
            "",
        ])
    ob_lines.extend([
        f"**Correction cap fires overnight:** "
        f"{cap_fires if cap_fires is not None else 'log not found'}",
        f"**Regime shifts for {yesterday.isoformat()}:** {shifts}",
        "",
        "## Today's Forecast",
        "",
    ])
    if today_forecast is not None:
        ob_lines.extend([
            f"- Final: **{today_forecast['final_f']:.1f}°F** "
            f"(±{today_forecast['uncertainty_f']:.1f}°F)",
            f"- Issued: {today_forecast['issued_at']}",
            f"- Mode: {today_forecast['mode']}",
        ])
    ob_lines.extend([
        "",
        "## Related",
        "",
        "- [[Home]]",
        "- [[Roadmap]]",
    ])
    obsidian_content = "\n".join(ob_lines)

    return {
        "telegram": telegram_content,
        "obsidian": obsidian_content,
        "target_date": today,
    }
