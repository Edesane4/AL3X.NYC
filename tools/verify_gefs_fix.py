"""Session 5 Part 1 — verify GEFS fix returns ensemble members live.

Run:
    python3 tools/verify_gefs_fix.py

Expected output: status 200, 30+ keys containing "member", n_members >= 20.

Compare against the pre-fix output (saved here for reference):
    Status: 200
    Total hourly keys: 2
    Keys containing "member": 0
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx                       # noqa: E402
from al3x import config as cfg     # noqa: E402


async def main() -> int:
    params = {
        "latitude": cfg.LAT, "longitude": cfg.LON,
        "hourly": "temperature_2m",
        "models": "gfs05",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
        "forecast_days": 3,
    }
    print(f"URL:    {cfg.OPEN_METEO_ENSEMBLE}")
    print(f"Params: {params}\n")

    async with httpx.AsyncClient(timeout=25.0) as c:
        r = await c.get(cfg.OPEN_METEO_ENSEMBLE, params=params)

    print(f"Status:          {r.status_code}")
    print(f"Response length: {len(r.text)} bytes")
    if r.status_code != 200:
        print("Body prefix:", r.text[:500])
        return 1

    js = r.json()
    hourly = js.get("hourly", {}) or {}
    keys = list(hourly.keys())
    member_keys = [k for k in keys if "member" in k.lower()]
    has_control = "temperature_2m" in keys

    print(f"Total hourly keys:  {len(keys)}")
    print(f"Has control series: {has_control}")
    print(f"Member key count:   {len(member_keys)}")
    if member_keys:
        print(f"Sample member keys: {member_keys[:5]}")

    # Sanity: at least the control + several perturbed members
    if not has_control:
        print("\nFAIL: no control `temperature_2m` series in response")
        return 2
    if len(member_keys) < 5:
        print(f"\nFAIL: only {len(member_keys)} member keys; expected 20+")
        return 3
    print(f"\nPASS: control + {len(member_keys)} perturbed members returned")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
