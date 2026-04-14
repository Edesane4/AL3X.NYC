# AL3X.NYC

**24/7 meteorological agent for the official NWS CLI recorded daily
maximum temperature at Central Park (KNYC).**

AL3X.NYC pulls data from every operational model the NWS publishes for
Central Park, blends them into a weighted ensemble, applies NYC-specific
local-bias corrections (sea breeze, urban heat island, cloud timing,
precipitation, inversions), verifies itself against the evening CLI
report, and auto-tunes its own weights and bias ledger over rolling
7/14-day windows.

* 🔭 **Night-Before forecast** issued each evening for tomorrow's high.
* 🔁 **Intraday revisions** every 15 minutes, with a running ASOS-max
  floor rule once Central Park starts climbing.
* 🎯 **CLI verification** each evening between 5:30–7:30 PM ET; every
  forecast scored, anomalies flagged.
* 🧠 **Learning loop** — source weights and bias corrections retune
  themselves.
* 📲 **Telegram updates** for every new forecast plus WARNING/ERROR log
  events from the Python process.
* 💻 **Live HTML dashboard** with polling updates.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # fill in TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
python app.py
```

Open <http://localhost:8080> for the dashboard.

### Telegram setup

1. Message [@BotFather](https://t.me/BotFather) on Telegram, `/newbot`,
   save the token.
2. Start a chat with your new bot and send any message.
3. Visit
   `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your
   numeric `chat.id`.
4. Paste both values into `.env`.

## How it decides

| Data source | Endpoint | Role |
|---|---|---|
| NWS Point (hourly) | `api.weather.gov/points/.../forecast/hourly` | human-adjusted baseline + regime detection |
| NBM (daily) | `api.weather.gov/points/.../forecast` | consensus sanity-check |
| IEM ASOS live | `mesonet.agron.iastate.edu/.../KNYC/observations.json` | ground truth & running max floor |
| GFS-MOS | `mdl.nws.noaa.gov/api/product/glamos/` | bias-corrected station guidance |
| NAM-MOS | `mdl.nws.noaa.gov/api/product/nammos/` | short-range guidance |
| HRRR (via Open-Meteo) | `api.open-meteo.com/v1/forecast?models=best_match` | 3 km high-res 0–18 h |
| ECMWF (via Open-Meteo) | `api.open-meteo.com/v1/forecast?models=ecmwf_ifs04` | best global for day+1 |
| NWS CLI | `forecast.weather.gov/product.php?...&product=CLI` | evening ground truth |

Default ensemble weights (Night-Before):

```
HRRR         25%
NWS Point    25%
GFS-MOS      20%
NAM-MOS      15%
ECMWF        15%
```

Intraday weights dynamically shift by lead time (see
`al3x/config.py`). NYC local-bias rules live in `al3x/forecaster.py`.

## Performance targets

* Night-Before MAE ≤ 2.5 °F within 30 days.
* Final Intraday MAE ≤ 1.5 °F within 30 days.
* Zero |error| > 6 °F after 60 days (anomalies root-caused and
  corrected).

## Project layout

```
al3x/
  config.py         # site, endpoints, default weights, bias defaults
  data_sources.py   # async fetchers for every source
  forecaster.py     # ensemble + NYC bias corrections
  learning.py       # scoring, bias ledger, auto-tune
  scheduler.py      # APScheduler cron/interval jobs
  storage.py        # SQLite persistence
  telegram_bot.py   # Telegram notifier + log handler
  logging_setup.py  # pipes warnings/errors into Telegram
static/index.html   # live dashboard
app.py              # FastAPI entrypoint
```
