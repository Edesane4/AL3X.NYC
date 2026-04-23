"""FIX 4 — ECMWF availability diagnostic.

Evidence: live availability 0/200. Every call to
``DataSources.ecmwf`` returns "no target-date hours", meaning the
Open-Meteo hourly payload contains zero entries whose .date() matches
the target. Possible causes include model ID rename, timezone shift,
or a parameter change upstream.

This script makes a raw HTTPS call with the exact parameters the app
uses (from al3x/config.py) and prints the response structure so the
root cause can be identified without re-running.

Run with:  python3 tools/diagnose_ecmwf.py

--- Captured output (run 2026-04-22, sandbox environment) ---
    Status code: 403
    Response length: 21 bytes
    Body: "Host not in allowlist"

The session sandbox blocks outbound HTTPS to api.open-meteo.com, so no
live capture was possible. Re-run this script on the MacBook where
al3x.db lives to capture the actual upstream response; record the new
output in this comment so the next session can read it without
re-executing.

Expected response structure when the feed works:
  status_code: 200
  hourly.time: list of ISO-8601 timestamps in America/New_York
  hourly.temperature_2m: parallel list of °F floats
  dates found: today, tomorrow, day-after-tomorrow (forecast_days=3)
                plus yesterday (past_days=1)

Known Open-Meteo gotchas to investigate in the captured output:
  * model id — `ecmwf_ifs04` may have been renamed to `ecmwf_ifs025`
    (0.25° grid) or simply `ecmwf`. Check js.get("reason") and any
    "hourly_units" / "models" echo fields.
  * timezone — if the API ignores `America/New_York` for this model,
    timestamps come back in UTC and the .date() filter in
    data_sources.open_meteo drops every entry for the target local
    date.
  * forecast_days — some model-specific endpoints cap at shorter
    horizons; ECMWF IFS historically serves 10 days.

If the diagnostic shows `hourly.time` is empty or missing,
the Open-Meteo endpoint itself is rejecting our parameters (check
`status_code` and `js.get("reason")`). If `hourly.time` is populated
but no entry matches the target local date, that's a timezone/model
mismatch — fix at al3x/data_sources.py ecmwf() accordingly.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta

import httpx

# Import exactly what the app imports, so parameters match live usage.
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from al3x import config as cfg  # noqa: E402


def main() -> int:
    params = {
        "latitude": cfg.LAT,
        "longitude": cfg.LON,
        "hourly": "temperature_2m",
        "models": "ecmwf_ifs04",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
        "forecast_days": 3,
        "past_days": 1,
    }

    print(f"URL: {cfg.OPEN_METEO}")
    print(f"Params: {json.dumps(params, indent=2)}")
    print()

    try:
        r = httpx.get(cfg.OPEN_METEO, params=params, timeout=20.0,
                      headers={"User-Agent": cfg.USER_AGENT})
    except Exception as e:  # noqa: BLE001
        print(f"HTTP error: {type(e).__name__}: {e}")
        return 2

    print(f"Status code: {r.status_code}")
    print(f"Response length: {len(r.text)} bytes")
    print(f"Final URL: {r.url}")
    print()

    if r.status_code != 200:
        # Open-Meteo returns a JSON error body with a 'reason' field
        # when it rejects the parameters. Print the full body so the
        # reason surfaces in the captured output.
        print("Non-200 response body (first 2KB):")
        print(r.text[:2000])
        return 1

    try:
        js = r.json()
    except Exception as e:  # noqa: BLE001
        print(f"JSON decode error: {e}")
        print("Body prefix:", r.text[:500])
        return 3

    print("Top-level keys:", sorted(js.keys()))
    print("Echoed 'reason' field:", js.get("reason"))
    print("Echoed 'timezone':", js.get("timezone"))
    print("Echoed 'timezone_abbreviation':", js.get("timezone_abbreviation"))
    print("Echoed 'utc_offset_seconds':", js.get("utc_offset_seconds"))
    print()

    hourly = js.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    print(f"hourly.time entries: {len(times)}")
    print(f"hourly.temperature_2m entries: {len(temps)}")
    if times:
        print(f"  first = {times[0]}")
        print(f"  last  = {times[-1]}")

    # Count per-local-date so the caller can see whether the target date
    # survives the parse.
    date_counts: dict[str, int] = {}
    for t in times:
        try:
            d = datetime.fromisoformat(t).date().isoformat()
        except Exception:
            continue
        date_counts[d] = date_counts.get(d, 0) + 1
    print("Entries per local date:", json.dumps(date_counts, indent=2))

    today = date.today()
    target = today + timedelta(days=0)
    print()
    print(f"Today (local):   {today.isoformat()}  entries: "
          f"{date_counts.get(today.isoformat(), 0)}")
    print(f"Target date:     {target.isoformat()}  entries: "
          f"{date_counts.get(target.isoformat(), 0)}")
    print(f"Tomorrow:        {(today + timedelta(days=1)).isoformat()}  "
          f"entries: "
          f"{date_counts.get((today + timedelta(days=1)).isoformat(), 0)}")

    if not date_counts.get(target.isoformat()):
        print()
        print("⚠️  target date has NO entries in the response. This is "
              "the live failure mode. Cross-check:")
        print("    * model id: is 'ecmwf_ifs04' still served?")
        print("    * timezone: did entries come back in UTC rather than "
              "America/New_York?")
        print("    * forecast_days: is the horizon shorter than expected?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
