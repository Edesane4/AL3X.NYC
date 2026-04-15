"""Position sizing engine — quarter-Kelly with hard bankroll limits.

Converts a probability edge and current bankroll state into a concrete
number of contracts to buy, respecting all exposure limits from config.
Never raises. Returns 0 contracts if any limit would be breached.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

from . import config as cfg

log = logging.getLogger("al3x.kelly")


def kelly_fraction(p: float, b: float) -> float:
    """Full Kelly fraction for a binary bet.

    p: probability of winning (0-1)
    b: net odds (profit per unit risked). For a binary contract:
       b = (1 - entry_price) / entry_price
       e.g., buying YES at 0.60 → b = 0.40/0.60 = 0.667
    Returns f* in [0, 1]. Negative values (negative edge) return 0.
    """
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0
    f = (p * b - (1 - p)) / b
    return max(0.0, f)


def size_position(fair_value: float,
                   entry_price: float,
                   side: str,
                   current_bankroll: float,
                   current_total_exposure: float,
                   current_threshold_exposure: float,
                   strategy: str = "model_divergence"
                   ) -> Dict[str, Any]:
    """Compute the number of contracts to buy and the total cost.

    Returns:
        {
          contracts: int,
          cost_basis: float,
          kelly_full: float,
          kelly_fraction_used: float,
          bankroll_pct: float,
          reason: str,   # why this size (or why 0)
        }
    """
    zero = lambda reason: {
        "contracts": 0, "cost_basis": 0.0,
        "kelly_full": 0.0, "kelly_fraction_used": 0.0,
        "bankroll_pct": 0.0, "reason": reason,
    }

    if current_bankroll <= 0:
        return zero("bankroll_exhausted")

    # For YES: p = fair_value, entry = ask price
    # For NO:  p = 1 - fair_value, entry = 1 - bid price
    if side == "yes":
        p = float(fair_value)
        entry = float(entry_price)
    else:
        p = 1.0 - float(fair_value)
        entry = 1.0 - float(entry_price)  # NO price = 1 - YES bid

    if entry <= 0 or entry >= 1:
        return zero("invalid_entry_price")

    b = (1.0 - entry) / entry  # net odds
    f_full = kelly_fraction(p, b)
    if f_full <= 0:
        return zero("no_positive_edge")

    # Apply fractional Kelly
    f_used = cfg.KALSHI_FRACTIONAL_KELLY * f_full
    raw_dollars = current_bankroll * f_used

    # Running-max arb gets higher size limit (near-certain outcome)
    if strategy == "running_max":
        max_single = current_bankroll * cfg.KALSHI_MAX_THRESHOLD_EXPOSURE_PCT
    else:
        max_single = current_bankroll * cfg.KALSHI_MAX_SINGLE_POSITION_PCT

    # Cross-market arb: split evenly between two legs
    if strategy == "arb":
        max_single = current_bankroll * cfg.KALSHI_MAX_THRESHOLD_EXPOSURE_PCT / 2

    # Apply all caps
    capped = min(raw_dollars, max_single)

    # Check threshold exposure cap
    remaining_threshold = (
        current_bankroll * cfg.KALSHI_MAX_THRESHOLD_EXPOSURE_PCT
        - current_threshold_exposure
    )
    capped = min(capped, max(0.0, remaining_threshold))

    # Check total exposure cap
    remaining_total = (
        current_bankroll * cfg.KALSHI_MAX_TOTAL_EXPOSURE_PCT
        - current_total_exposure
    )
    capped = min(capped, max(0.0, remaining_total))

    if capped < entry:  # can't even afford one contract
        return zero("exposure_limit_reached")

    # Kalshi contracts are priced per share; 1 contract = entry_price dollars
    contracts = int(math.floor(capped / entry))
    if contracts < 1:
        return zero("below_minimum_size")

    cost_basis = round(contracts * entry, 4)
    bankroll_pct = round(cost_basis / current_bankroll * 100, 2)

    return {
        "contracts": contracts,
        "cost_basis": cost_basis,
        "kelly_full": round(f_full, 4),
        "kelly_fraction_used": round(f_used, 4),
        "bankroll_pct": bankroll_pct,
        "reason": "sized",
    }


def compute_unrealized_pnl(positions: list,
                             current_prices: Dict[str, float]) -> float:
    """Compute total unrealized P&L across all open positions.

    positions: list of open kalshi_positions rows from storage
    current_prices: {market_ticker: current_mid_price}
    """
    total = 0.0
    for pos in positions:
        ticker = pos.get("market_ticker", "")
        current_mid = current_prices.get(ticker)
        if current_mid is None:
            continue
        entry = float(pos.get("entry_price", 0))
        contracts = int(pos.get("contracts", 0))
        side = pos.get("side", "yes")
        if side == "yes":
            unrealized = (current_mid - entry) * contracts
        else:
            # NO position: bought at (1-entry), current value = (1-current_mid)
            unrealized = ((1 - current_mid) - (1 - entry)) * contracts
        total += unrealized
    return round(total, 4)
