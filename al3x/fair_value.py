"""Fair value calculator — converts AL3X forecast to Kalshi contract prices.

Takes AL3X's point forecast, BMA sigma, and QRF p10/p90 to compute the
probability that today's NYC high temperature will exceed any given
threshold. This probability IS the fair value of the YES contract on
Kalshi. Comparing it to the market price gives the edge.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfg

log = logging.getLogger("al3x.fair_value")


# ---- Normal distribution helpers ----------------------------------------

def _erf(x: float) -> float:
    """Error function approximation (Abramowitz & Stegun 7.1.26).
    Max error: 1.5e-7. Pure Python — no scipy needed.
    """
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    poly = t * (0.254829592
                + t * (-0.284496736
                       + t * (1.421413741
                              + t * (-1.453152027
                                     + t * 1.061405429))))
    result = 1.0 - poly * math.exp(-(x * x))
    return result if x >= 0 else -result


def _normal_cdf(x: float, mean: float, std: float) -> float:
    """P(X <= x) for X ~ Normal(mean, std)."""
    if std <= 0:
        return 0.0 if x < mean else 1.0
    z = (x - mean) / (std * math.sqrt(2.0))
    return 0.5 * (1.0 + _erf(z))


def _normal_pdf(x: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.0
    z = (x - mean) / std
    return math.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


# ---- Sigma estimation ---------------------------------------------------

def estimate_sigma(bma_result: Optional[Dict[str, Any]],
                    qrf_result: Optional[Dict[str, Any]],
                    fallback_sigma: float = 2.5) -> float:
    """Derive a forecast standard deviation from whichever source is best.

    Priority:
      1. QRF interval: sigma = (p90 - p10) / 2.563  [80% interval ≈ 2.563σ]
      2. BMA variance_f (already a sigma)
      3. Fallback: 2.5°F (typical NWS day-ahead accuracy)
    """
    if qrf_result is not None:
        p10 = qrf_result.get("p10_delta")
        p90 = qrf_result.get("p90_delta")
        if p10 is not None and p90 is not None:
            width = float(p90) - float(p10)
            if width > 0:
                return max(0.5, width / 2.563)
    if bma_result is not None:
        v = bma_result.get("bma_variance_f")
        if v is not None and float(v) > 0:
            return max(0.5, float(v))
    return fallback_sigma


# ---- Core fair value computation ----------------------------------------

def compute_fair_value(threshold_f: float,
                        al3x_forecast_f: float,
                        sigma: float,
                        running_max_f: Optional[float] = None,
                        running_max_age_minutes: Optional[float] = None
                        ) -> Dict[str, Any]:
    """Return the fair YES probability for 'NYC high > threshold_f'.

    If running_max_f >= threshold_f and the reading is at least
    KALSHI_RUNNING_MAX_MIN_AGE_MINUTES old, the contract has
    effectively settled YES — return near-certainty.

    Returns:
        {
          fair_value: float (0-1 probability),
          method: str,
          sigma_used: float,
          z_score: float,    (how many sigmas threshold is from forecast)
        }
    """
    # Running max floor: if ASOS already confirmed the threshold
    min_age = cfg.KALSHI_RUNNING_MAX_MIN_AGE_MINUTES
    if (running_max_f is not None
            and running_max_f >= threshold_f
            and (running_max_age_minutes is None
                 or running_max_age_minutes >= min_age)):
        return {
            "fair_value": 0.99,
            "method": "running_max_confirmed",
            "sigma_used": sigma,
            "z_score": None,
        }

    # Standard normal CDF: P(high > threshold) = 1 - CDF(threshold)
    p_exceed = 1.0 - _normal_cdf(threshold_f, al3x_forecast_f, sigma)
    p_exceed = max(0.01, min(0.99, p_exceed))  # clip to tradeable range
    z = ((threshold_f - al3x_forecast_f) / sigma) if sigma > 0 else 0.0

    return {
        "fair_value": round(p_exceed, 4),
        "method": "normal_cdf",
        "sigma_used": round(sigma, 3),
        "z_score": round(z, 3),
    }


def compute_all_fair_values(markets: List[Dict[str, Any]],
                              al3x_forecast_f: float,
                              sigma: float,
                              running_max_f: Optional[float] = None,
                              running_max_age_minutes: Optional[float] = None
                              ) -> Dict[str, Dict[str, Any]]:
    """Compute fair values for all active markets.

    markets: list of dicts from KalshiFeed.get_nyc_temp_markets()
    Returns: {market_ticker: fair_value_dict}
    """
    result: Dict[str, Dict[str, Any]] = {}
    for m in markets:
        ticker = m.get("market_ticker", "")
        threshold = m.get("threshold_f")
        if ticker and threshold is not None:
            fv = compute_fair_value(
                float(threshold), al3x_forecast_f, sigma,
                running_max_f, running_max_age_minutes,
            )
            fv["threshold_f"] = float(threshold)
            fv["market_ticker"] = ticker
            result[ticker] = fv
    return result


# ---- Edge and EV calculation -------------------------------------------

def compute_edge(fair_value: float,
                  market_price: float,
                  side: str = "yes") -> Dict[str, Any]:
    """Return signed edge in cents and expected value for a given side.

    side: 'yes' — buy YES contract at market_price
          'no'  — buy NO contract (equivalent to selling YES at market_price)

    Edge (cents) = (fair_value - market_price) * 100  [YES side]
                 = ((1 - fair_value) - (1 - market_price)) * 100  [NO side]
                 = (market_price - fair_value) * 100  [NO side simplified]

    EV formula for binary contract:
      EV_YES = fair_value * (1 - entry) - (1 - fair_value) * entry
      EV_NO  = (1 - fair_value) * (1 - no_entry) - fair_value * no_entry
             = (1 - fair_value) * market_price - fair_value * (1 - market_price)
    """
    if side == "yes":
        edge_cents = round((fair_value - market_price) * 100, 2)
        ev = (fair_value * (1 - market_price)
              - (1 - fair_value) * market_price)
    else:
        no_price = 1.0 - market_price
        edge_cents = round(((1 - fair_value) - no_price) * 100, 2)
        ev = ((1 - fair_value) * market_price
              - fair_value * (1 - market_price))

    return {
        "side": side,
        "edge_cents": edge_cents,
        "ev": round(ev, 4),
        "market_price": market_price,
        "fair_value": fair_value,
        "actionable": (abs(edge_cents) >= cfg.KALSHI_MIN_EDGE_CENTS
                       and ev >= cfg.KALSHI_MIN_NET_EV),
    }


def best_side(fair_value: float,
               best_bid: Optional[float],
               best_ask: Optional[float]) -> Optional[Dict[str, Any]]:
    """Determine which side (YES or NO) has the better edge, if any.

    Returns the best edge dict or None if neither side clears the
    minimum thresholds.
    """
    candidates = []
    if best_ask is not None:
        yes_edge = compute_edge(fair_value, best_ask, "yes")
        if yes_edge["actionable"]:
            candidates.append(yes_edge)
    if best_bid is not None:
        no_edge = compute_edge(fair_value, best_bid, "no")
        if no_edge["actionable"]:
            candidates.append(no_edge)
    if not candidates:
        return None
    return max(candidates, key=lambda x: x["ev"])


# ---- Monotonicity arbitrage detection ----------------------------------

def detect_monotonicity_violation(
        fair_values: Dict[str, Dict[str, Any]],
        snapshots: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Find pairs of contracts where the market implies a higher probability
    for a stricter threshold — a provable mispricing.

    fair_values:  {ticker: {threshold_f, fair_value, ...}}
    snapshots:    {ticker: {best_bid, best_ask, ...}}

    Returns list of arbitrage opportunities, each with:
        lower_ticker, upper_ticker, lower_threshold, upper_threshold,
        lower_market_price, upper_market_price, arb_cents
    """
    # Build sorted list of (threshold, ticker, market_price)
    points = []
    for ticker, fv in fair_values.items():
        snap = snapshots.get(ticker, {})
        bid = snap.get("best_bid")
        ask = snap.get("best_ask")
        if bid is None or ask is None:
            continue
        mid = round((bid + ask) / 2, 4)
        points.append((float(fv["threshold_f"]), ticker, mid, bid, ask))
    points.sort(key=lambda x: x[0])  # ascending threshold

    violations = []
    for i in range(len(points) - 1):
        t_low, tk_low, p_low, bid_low, ask_low = points[i]
        t_high, tk_high, p_high, bid_high, ask_high = points[i + 1]
        # Monotonicity: P(high > lower) >= P(high > higher)
        # Violation: p_low < p_high
        if p_low < p_high - 0.02:  # 2-cent tolerance for noise
            # Arb: buy the lower-threshold YES (underpriced), buy the
            # higher-threshold NO (overpriced YES means cheap NO)
            arb_cents = round((p_high - p_low) * 100, 1)
            violations.append({
                "lower_ticker": tk_low,
                "upper_ticker": tk_high,
                "lower_threshold_f": t_low,
                "upper_threshold_f": t_high,
                "lower_market_price": p_low,
                "upper_market_price": p_high,
                "arb_cents": arb_cents,
                "buy_lower_yes_at": ask_low,
                "buy_upper_no_at": round(1.0 - bid_high, 4),
            })
    return violations
