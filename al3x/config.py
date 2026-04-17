"""Static configuration: site, endpoints, default weights, bias defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

import pytz

# Central Park (KNYC) location
LAT = 40.7789
LON = -73.9692
STATION_ID = "KNYC"
WFO = "OKX"

EASTERN = pytz.timezone("America/New_York")

# --- Endpoints ---------------------------------------------------------------
NWS_POINT_HOURLY = f"https://api.weather.gov/points/{LAT},{LON}/forecast/hourly"
NWS_POINT_DAILY = f"https://api.weather.gov/points/{LAT},{LON}/forecast"
IEM_ASOS = (
    "https://mesonet.agron.iastate.edu/api/1/stations/"
    f"{STATION_ID}/observations.json"
)
MDL_MOS = "https://mdl.nws.noaa.gov/api/product/glamos/"
MDL_NAMMOS = "https://mdl.nws.noaa.gov/api/product/nammos/"

# Canonical MOS source — Iowa Environmental Mesonet per-station MAV endpoint.
# NAM-MOS (MET) is intentionally not included: IEM does not archive NAM-MOS
# for KNYC (Central Park is on the GFS full-station list but not the NAM
# cooperative list). We don't substitute a nearby airport station — the
# ensemble re-weights across the remaining sources.
GFS_MOS_URL = ("https://mesonet.agron.iastate.edu/api/1/mos.txt"
               f"?station={STATION_ID}&model=GFS")
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
NWS_CLI = (
    "https://forecast.weather.gov/product.php"
    f"?site={WFO}&issuedby=NYC&product=CLI&format=CI&version=1&glossary=0"
)

# HTTP
USER_AGENT = "AL3X.NYC (CentralPark-high-temp-agent; github.com/edesane4/al3x.nyc)"
HTTP_TIMEOUT = 20.0

# --- Default ensemble weights ------------------------------------------------
# Weights are dicts keyed by source name. They must sum to ~1.0 but the
# forecaster re-normalizes anything missing.
NIGHT_BEFORE_WEIGHTS: Dict[str, float] = {
    # UPGRADE A — gfs_ensemble carved out at 0.10; the remaining 0.90 is
    # distributed proportionally from the pre-GEFS layout (hrrr 30,
    # nws_point 25, gfs_mos 25, ecmwf 20 → each × 0.90).
    "hrrr": 0.27,
    "nws_point": 0.225,
    "gfs_mos": 0.225,
    "ecmwf": 0.18,
    "gfs_ensemble": 0.10,
}

# UPGRADE A — gfs_ensemble weight at 0.05 in intraday (slower-responding
# than HRRR); remaining 0.95 scales the pre-GEFS layouts.
INTRADAY_WEIGHTS_0_6 = {
    "hrrr": 0.475,
    "nws_point": 0.285,
    "asos_trend": 0.19,
    "gfs_ensemble": 0.05,
}
INTRADAY_WEIGHTS_6_12 = {
    "hrrr": 0.3325,
    "nws_point": 0.2375,
    "gfs_mos": 0.19,
    "ecmwf": 0.095,
    "asos_trend": 0.095,
    "gfs_ensemble": 0.05,
}
INTRADAY_WEIGHTS_12_24 = {
    "hrrr": 0.209,
    "nws_point": 0.2375,
    "gfs_mos": 0.19,
    "ecmwf": 0.3135,
    "gfs_ensemble": 0.05,
}

# --- Default bias corrections ------------------------------------------------
@dataclass
class BiasDefaults:
    """Default additive bias corrections (°F). Mutable at runtime via learning."""

    sea_breeze_shift: float = -2.0
    sea_breeze_full: float = -3.0
    sea_breeze_nw_boost: float = 1.0
    uhi_clear_calm: float = 1.0
    cloud_increase_morning: float = -2.0
    cloud_clearing_afternoon: float = 2.0
    precip_light: float = -3.0
    precip_heavy: float = -5.0
    inversion_winter: float = -1.0


BIAS_DEFAULTS = BiasDefaults()


# --- Performance targets (documented reference) ------------------------------
TARGET_NIGHT_BEFORE_MAE = 2.5
TARGET_INTRADAY_MAE = 1.5
ANOMALY_THRESHOLD = 6.0


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# --- Kalshi Market Intelligence -------------------------------------------
KALSHI_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_FEED_INTERVAL_SECONDS = 30       # orderbook poll frequency
KALSHI_NYC_EVENT_PREFIX = "KXHIGHNY"  # Kalshi event ticker prefix for NYC high temp
KALSHI_MIN_EV = 0.04                    # minimum expected value to consider a trade
KALSHI_MIN_EDGE_CENTS = 5               # minimum fair-value divergence in cents
KALSHI_MIN_EDGE_PERSIST_CYCLES = 2      # edge must persist this many cycles before entry
KALSHI_MAX_SPREAD_CENTS = 8             # skip markets where spread exceeds this
KALSHI_FRACTIONAL_KELLY = 0.25          # quarter-Kelly position sizing
KALSHI_MAX_SINGLE_POSITION_PCT = 0.08   # 8% bankroll max per contract
KALSHI_MAX_THRESHOLD_EXPOSURE_PCT = 0.12
KALSHI_MAX_TOTAL_EXPOSURE_PCT = 0.35
KALSHI_DAILY_LOSS_LIMIT_PCT = 0.15
KALSHI_STARTING_BANKROLL = 500.0
KALSHI_LIVE_TRADING = False             # set True only after paper validation
KALSHI_PAPER_MODE = True                # log simulated trades without placing real ones
KALSHI_SPOOF_SIZE_THRESHOLD_PCT = 0.05  # 5% of total depth to flag as spoof candidate
KALSHI_PANIC_VELOCITY_MULTIPLIER = 3.0
KALSHI_SMART_MONEY_MIN_CYCLES = 5       # resting cycles to classify as patient capital
KALSHI_RUNNING_MAX_CERTAINTY_THRESHOLD = 0.95  # buy YES below this when ASOS confirmed
KALSHI_RUNNING_MAX_MIN_AGE_MINUTES = 30
KALSHI_FEE_PER_CONTRACT = 0.009   # ~0.9¢ taker fee on winning side
KALSHI_MIN_NET_EV = 0.04 + KALSHI_FEE_PER_CONTRACT  # net-of-fee EV floor
