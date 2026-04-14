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

# Canonical MOS sources — Iowa Environmental Mesonet (mesonet.agron.iastate.edu)
# serves per-station MAV/MET text bulletins at a simple JSON/text API. IEM has
# been archiving NWS MOS since 2008 and their endpoint is purpose-built for
# scripted access (unlike NOAA's FTPPRD/NOMADS which are either unreachable or
# don't mirror MOS to nomads). The .txt endpoint returns a single-station
# MAV/MET block we can feed straight to _parse_mos_max().
GFS_MOS_URL = ("https://mesonet.agron.iastate.edu/api/1/mos.txt"
               f"?station={STATION_ID}&model=GFS")
NAM_MOS_URL = ("https://mesonet.agron.iastate.edu/api/1/mos.txt"
               f"?station={STATION_ID}&model=NAM")
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
    "hrrr": 0.25,
    "nws_point": 0.25,
    "gfs_mos": 0.20,
    "nam_mos": 0.15,
    "ecmwf": 0.15,
}

INTRADAY_WEIGHTS_0_6 = {
    "hrrr": 0.50,
    "nws_point": 0.30,
    "asos_trend": 0.20,
}
INTRADAY_WEIGHTS_6_12 = {
    "hrrr": 0.35,
    "nws_point": 0.25,
    "gfs_mos": 0.20,
    "ecmwf": 0.10,
    "asos_trend": 0.10,
}
INTRADAY_WEIGHTS_12_24 = {
    "hrrr": 0.20,
    "nws_point": 0.25,
    "gfs_mos": 0.20,
    "ecmwf": 0.25,
    "nam_mos": 0.10,
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
