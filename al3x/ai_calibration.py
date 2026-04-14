"""Gap 5 — Claude AI calibration layer.

Calls the Anthropic API to produce a final ±delta (°F) on top of the
statistical ensemble + bias-corrected forecast. Strictly optional: if
ANTHROPIC_API_KEY is unset, calibrate() returns delta=0.0 and the rest of
the pipeline is untouched.

Caching: responses keyed by (mode, target_date, round(raw_ensemble, 1),
regime_flags_tuple) are cached for 20 minutes to hold API calls under
~4/day ≈ $1/month.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("al3x.ai_calibration")

_MODEL = "claude-sonnet-4-20250514"
_CACHE_TTL_SECONDS = 20 * 60
_DELTA_CAP = 2.0


class AICalibrator:
    """Thin wrapper around the Anthropic SDK with response caching."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._client = None
        if self.api_key:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except Exception as e:  # noqa: BLE001
                log.warning("anthropic SDK unavailable (%s); AI layer disabled", e)
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def calibrate(self, forecast_dict: Dict[str, Any],
                        obs_today: List[Dict[str, Any]],
                        recent_scores: List[Dict[str, Any]]
                        ) -> Dict[str, Any]:
        """Return {"delta_f": float, "confidence": float, "reasoning": str}.

        Never raises. On any error returns delta_f=0.0.
        """
        if not self.enabled:
            return {"delta_f": 0.0, "confidence": 0.0,
                    "reasoning": "ANTHROPIC_API_KEY not set — AI layer disabled",
                    "source": "disabled"}

        cache_key = _cache_key(forecast_dict)
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            out = dict(cached[1])
            out["source"] = "cache"
            return out

        try:
            prompt = _build_prompt(forecast_dict, obs_today, recent_scores)
        except Exception as e:  # noqa: BLE001
            log.warning("AI prompt build failed: %s", e)
            return {"delta_f": 0.0, "confidence": 0.0,
                    "reasoning": f"prompt build error: {e}",
                    "source": "error"}

        system = (
            "You are a final calibration layer for a Central Park NYC high "
            "temperature forecasting agent. You receive the agent's ensemble "
            "forecast and all context. Your job is to recommend a small final "
            "adjustment in °F to improve accuracy. You must respond ONLY with "
            "valid JSON in this exact format: "
            '{"delta_f": float between -3.0 and 3.0, '
            '"confidence": float 0.0 to 1.0, '
            '"reasoning": string under 200 chars}. '
            "Base your adjustment on: known model biases you can detect in "
            "the source disagreements, correction interactions that may be "
            "double-counting or missing (e.g. sea breeze + morning clouds "
            "applied additively when they partially cancel), patterns you "
            "see in the 14-day error history, and any regime combination "
            "that historically causes systematic error at this specific "
            "station. If you see no reason to adjust, return delta_f: 0.0."
        )

        try:
            import asyncio
            def _call():
                return self._client.messages.create(
                    model=_MODEL,
                    max_tokens=400,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )

            resp = await asyncio.to_thread(_call)
            text = ""
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    text += block.text
            parsed = _parse_response(text)
            parsed["delta_f"] = max(-_DELTA_CAP,
                                     min(_DELTA_CAP, float(parsed["delta_f"])))
            parsed["source"] = "live"
            self._cache[cache_key] = (now, parsed)
            return parsed
        except Exception as e:  # noqa: BLE001
            log.warning("AI calibration call failed: %s", e)
            return {"delta_f": 0.0, "confidence": 0.0,
                    "reasoning": f"API error: {type(e).__name__}: {e}",
                    "source": "error"}


def _cache_key(fc: Dict[str, Any]) -> str:
    regime = fc.get("extras", {}).get("regime", {}) or {}
    flags = tuple(sorted(
        (k, v) for k, v in regime.items()
        if isinstance(v, bool)
    ))
    payload = {
        "mode": fc.get("mode"),
        "target_date": fc.get("target_date"),
        "raw_ensemble": round(fc.get("raw_ensemble_f", 0.0), 1),
        "flags": flags,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _parse_response(text: str) -> Dict[str, Any]:
    """Parse Claude's JSON reply; raise ValueError on malformed data."""
    import re
    t = text.strip()
    # Strip markdown code fence if present
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        raise ValueError("no JSON object found in response")
    obj = json.loads(m.group(0))
    for k in ("delta_f", "confidence", "reasoning"):
        if k not in obj:
            raise ValueError(f"missing key {k}")
    return {
        "delta_f": float(obj["delta_f"]),
        "confidence": float(obj["confidence"]),
        "reasoning": str(obj["reasoning"])[:400],
    }


def _build_prompt(fc: Dict[str, Any], obs_today: List[Dict[str, Any]],
                  recent_scores: List[Dict[str, Any]]) -> str:
    sources = fc.get("sources", {})
    corrs = fc.get("corrections", {})
    extras = fc.get("extras", {})
    regime = extras.get("regime", {}) or {}
    spread = extras.get("spread_f")
    lead = extras.get("lead_hours")

    source_lines = []
    for name, payload in sources.items():
        v = payload.get("value")
        if v is None:
            source_lines.append(f"- {name}: unavailable ({payload.get('error')})")
        else:
            w = payload.get("weight", 0.0)
            source_lines.append(f"- {name}: {v:.1f}°F (weight {w*100:.0f}%)")

    corr_lines = []
    for name, payload in corrs.items():
        d = payload.get("delta", 0)
        corr_lines.append(f"- {name}: {d:+.1f}°F — {payload.get('reason','')}")

    regime_lines = []
    for k in ("sea_breeze_shift", "sea_breeze_full",
              "cloud_morning_increase", "cloud_afternoon_clearing",
              "any_precip_peak", "precip_heavy", "inversion_hint",
              "calm_clear", "sustained_windy", "wind_nw_all_day"):
        regime_lines.append(f"- {k}: {regime.get(k)}")

    history_lines = []
    for s in (recent_scores or [])[:14]:
        history_lines.append(
            f"- {s['target_date']} [{s['mode']}] err={s['error_f']:+.1f}°F "
            f"regime={s.get('regime','?')}"
        )

    from datetime import datetime as _dt
    from . import config as _cfg
    now = _dt.now(_cfg.EASTERN)

    running = fc.get("running_asos_max_f")
    running_line = (f"Running ASOS max so far: {running:.1f}°F"
                    if running is not None else
                    "Running ASOS max: not yet (night-before or early morning)")

    return (
        f"Target date: {fc.get('target_date')}   Mode: {fc.get('mode')}\n"
        f"Current time (Eastern): {now.isoformat()}\n"
        f"Month: {now.month}   Lead hours to 3 PM: {lead}\n\n"
        f"Raw ensemble: {fc.get('raw_ensemble_f')}°F\n"
        f"Post-bias final: {fc.get('final_f')}°F\n"
        f"Model spread: {spread}°F   Uncertainty: ±{fc.get('uncertainty_f')}°F\n"
        f"{running_line}\n\n"
        "Sources (with weights):\n" + "\n".join(source_lines) + "\n\n"
        "Bias corrections applied:\n" + "\n".join(corr_lines) + "\n\n"
        "Regime flags:\n" + "\n".join(regime_lines) + "\n\n"
        "Last 14 days of verified errors:\n"
        + ("\n".join(history_lines) if history_lines else "- (no history yet)")
        + "\n\nRespond with JSON only."
    )
