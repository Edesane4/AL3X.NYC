"""Google Sheets auto-export (optional).

Wires AL3X.NYC into a Google Sheet with three tabs:
  * "Daily Results"   — one row per day after CLI verifies
  * "Forecast Log"    — one row per night-before forecast
  * "Performance"     — weekly MAE + correction attribution snapshot

Fully optional. If gspread / google-auth aren't installed or the env
variables aren't set, `_enabled` stays False and every public method
silently no-ops. No exception that originates in this module should
ever propagate to the scheduler.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("al3x.sheets")

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

_DAILY_HEADERS = [
    "Date", "Night-Before Forecast (°F)", "Final Intraday Forecast (°F)",
    "CLI Recorded High (°F)", "Night-Before Error (°F)", "Final Error (°F)",
    "Regime", "Sea Breeze", "Precip", "HRRR (°F)", "ECMWF (°F)",
    "GFS-MOS (°F)", "BMA Forecast (°F)", "Analog Bias (°F)",
    "AI Delta (°F)", "QRF p10 (°F)", "QRF p90 (°F)", "Uncertainty ±(°F)",
    "Model Spread (°F)",
]

_FORECAST_HEADERS = [
    "Issued At", "Target Date", "Mode", "Run #", "Forecast (°F)",
    "Raw Ensemble (°F)", "Model Spread (°F)", "Uncertainty ±(°F)",
    "HRRR (°F)", "ECMWF (°F)", "GFS-MOS (°F)", "NWS Point (°F)",
    "Sea Breeze Correction (°F)", "UHI Correction (°F)",
    "Cloud Correction (°F)", "Precip Correction (°F)",
    "BMA Active", "Analog Bias (°F)", "AI Delta (°F)", "Lead Hours",
]

_PERFORMANCE_HEADERS = [
    "Date", "Night-Before MAE", "Intraday MAE", "Sample Count",
    "Sea Breeze Helpful %", "UHI Helpful %",
    "Precip Helpful %", "Cloud Helpful %",
]


def _helpful_pct(attr: Dict[str, Any], name: str) -> str:
    row = attr.get(name) or {}
    n = row.get("times_applied") or 0
    if n <= 0:
        return ""
    h = row.get("times_helpful") or 0
    return f"{(h / n) * 100:.0f}%"


def _pick(d: Optional[Dict[str, Any]], key: str, default: Any = "") -> Any:
    if not d:
        return default
    v = d.get(key)
    return v if v is not None else default


class SheetsExporter:
    def __init__(self, spreadsheet_id: str, credentials_path: str) -> None:
        self.spreadsheet_id = spreadsheet_id
        self.credentials_path = credentials_path
        self._gc = None
        self._sh = None
        self._enabled: bool = False
        self._connect()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _connect(self) -> None:
        if not self.spreadsheet_id or not self.credentials_path:
            return
        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except Exception as e:  # noqa: BLE001
            log.info("gspread/google-auth not installed — Sheets disabled: %s", e)
            return
        try:
            creds = Credentials.from_service_account_file(
                self.credentials_path, scopes=_SCOPES,
            )
            self._gc = gspread.authorize(creds)
            self._sh = self._gc.open_by_key(self.spreadsheet_id)
            self._enabled = True
            log.info("Google Sheets exporter connected to spreadsheet %s",
                     self.spreadsheet_id)
        except Exception as e:  # noqa: BLE001
            log.warning("Google Sheets connect failed: %s", e)
            self._enabled = False

    def _get_or_create_sheet(self, title: str, headers: List[str]):
        if not self._enabled:
            return None
        try:
            ws = self._sh.worksheet(title)
        except Exception:
            # Either not found or another API error — try to create.
            try:
                ws = self._sh.add_worksheet(
                    title=title, rows=5000, cols=len(headers),
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Sheets: could not open or create tab %s: %s",
                            title, e)
                return None
        # Ensure header row exists
        try:
            first_row = ws.row_values(1)
            if not first_row:
                ws.update("A1", [headers])
        except Exception as e:  # noqa: BLE001
            log.debug("Sheets: header write skipped on %s: %s", title, e)
        return ws

    # -- Public push methods ---------------------------------------------

    def push_daily_result(self, date_str: str, data: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            ws = self._get_or_create_sheet("Daily Results", _DAILY_HEADERS)
            if ws is None:
                return
            row = [
                date_str,
                data.get("night_before_f", ""),
                data.get("final_intraday_f", ""),
                data.get("cli_f", ""),
                data.get("nb_error", ""),
                data.get("final_error", ""),
                data.get("regime", ""),
                "yes" if data.get("sea_breeze") else "no",
                "yes" if data.get("precip") else "no",
                data.get("hrrr_f", ""),
                data.get("ecmwf_f", ""),
                data.get("gfs_mos_f", ""),
                data.get("bma_f", ""),
                data.get("analog_bias", ""),
                data.get("ai_delta", ""),
                data.get("qrf_p10", ""),
                data.get("qrf_p90", ""),
                data.get("uncertainty", ""),
                data.get("spread", ""),
            ]
            ws.append_row(row, value_input_option="USER_ENTERED")
        except Exception as e:  # noqa: BLE001
            log.warning("Sheets push_daily_result failed: %s", e)

    def push_forecast_log(self, fc: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            ws = self._get_or_create_sheet("Forecast Log", _FORECAST_HEADERS)
            if ws is None:
                return
            sources = fc.get("sources") or {}
            corrs = fc.get("corrections") or {}
            extras = fc.get("extras") or {}
            bma = extras.get("bma")
            analog = extras.get("analog") or {}
            ai_cal = extras.get("ai_calibration") or {}
            row = [
                fc.get("issued_at", ""),
                fc.get("target_date", ""),
                fc.get("mode", ""),
                extras.get("night_before_run", fc.get("revision", 0)),
                fc.get("final_f", ""),
                fc.get("raw_ensemble_f", ""),
                extras.get("spread_f", ""),
                fc.get("uncertainty_f", ""),
                _pick(sources.get("hrrr"), "value"),
                _pick(sources.get("ecmwf"), "value"),
                _pick(sources.get("gfs_mos"), "value"),
                _pick(sources.get("nws_point"), "value"),
                _pick(corrs.get("sea_breeze"), "delta"),
                _pick(corrs.get("uhi"), "delta"),
                _pick(corrs.get("cloud_timing"), "delta"),
                _pick(corrs.get("precip"), "delta"),
                "yes" if bma else "no",
                analog.get("analog_bias_f", ""),
                ai_cal.get("delta_f", ""),
                extras.get("lead_hours", ""),
            ]
            ws.append_row(row, value_input_option="USER_ENTERED")
        except Exception as e:  # noqa: BLE001
            log.warning("Sheets push_forecast_log failed: %s", e)

    def push_performance_summary(self, stats: Dict[str, Any],
                                  attribution: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            ws = self._get_or_create_sheet("Performance", _PERFORMANCE_HEADERS)
            if ws is None:
                return
            samples = stats.get("samples") or {}
            sample_count = (samples.get("night_before", 0)
                            + samples.get("intraday", 0))
            today = datetime.now(timezone.utc).date().isoformat()
            row = [
                today,
                stats.get("night_before_mae", "") or "",
                stats.get("intraday_mae", "") or "",
                sample_count,
                _helpful_pct(attribution, "sea_breeze"),
                _helpful_pct(attribution, "uhi"),
                _helpful_pct(attribution, "precip"),
                _helpful_pct(attribution, "cloud_timing"),
            ]
            ws.append_row(row, value_input_option="USER_ENTERED")
        except Exception as e:  # noqa: BLE001
            log.warning("Sheets push_performance_summary failed: %s", e)
