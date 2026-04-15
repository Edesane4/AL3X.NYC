"""Circuit breakers and pre-trade risk checks for the Kalshi engine.

All checks are pure functions (no I/O). They take the current system
state and return a RiskStatus indicating whether trading should
proceed, be reduced, or be halted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import config as cfg

log = logging.getLogger("al3x.risk")


@dataclass
class RiskStatus:
    trading_allowed: bool = True
    size_multiplier: float = 1.0   # 1.0 = full size, 0.5 = half size, 0.0 = halt
    active_breakers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def halted(self) -> bool:
        return not self.trading_allowed or self.size_multiplier == 0.0


def check_all(bankroll: Dict[str, Any],
               open_positions: List[Dict[str, Any]],
               feed_healthy: bool,
               last_forecast_error_f: Optional[float],
               current_spread_cents: Optional[float],
               lead_hours_to_settlement: Optional[float],
               al3x_uncertainty_f: Optional[float],
               ) -> RiskStatus:
    """Run all circuit breakers. Returns a RiskStatus.

    Parameters
    ----------
    bankroll               : row from kalshi_bankroll table
    open_positions         : list of open position rows
    feed_healthy           : False if Kalshi feed errored 3+ consecutive cycles
    last_forecast_error_f  : yesterday's AL3X absolute error (°F), or None
    current_spread_cents   : spread in cents for the target market
    lead_hours_to_settlement: hours until market closes
    al3x_uncertainty_f     : current QRF/BMA uncertainty (interval_width/2)
    """
    status = RiskStatus()

    current_bankroll = float(bankroll.get("current_bankroll",
                                           cfg.KALSHI_STARTING_BANKROLL))
    starting_bankroll = float(bankroll.get("starting_bankroll",
                                            cfg.KALSHI_STARTING_BANKROLL))
    daily_pnl = float(bankroll.get("daily_pnl", 0.0))
    realized_pnl = float(bankroll.get("realized_pnl", 0.0))

    # CB1 — Daily loss limit
    daily_loss_limit = starting_bankroll * cfg.KALSHI_DAILY_LOSS_LIMIT_PCT
    if daily_pnl <= -daily_loss_limit:
        status.trading_allowed = False
        status.size_multiplier = 0.0
        status.active_breakers.append(
            f"CB1_daily_loss_limit: daily_pnl={daily_pnl:.2f} "
            f"limit=-{daily_loss_limit:.2f}"
        )
        log.warning("CB1 TRIGGERED: daily loss limit reached "
                    "(%.2f / -%.2f). Halting all new entries.",
                    daily_pnl, daily_loss_limit)

    # CB2 — Model confidence floor
    if al3x_uncertainty_f is not None and al3x_uncertainty_f > 8.0:
        if status.size_multiplier > 0.5:
            status.size_multiplier = 0.5
        status.warnings.append(
            f"CB2_low_confidence: uncertainty={al3x_uncertainty_f:.1f}°F "
            f"(> 8°F threshold) — size halved"
        )
        log.info("CB2: model uncertainty %.1f°F > 8°F — size reduced 50%%",
                 al3x_uncertainty_f)

    # CB3 — Orderbook liquidity
    if (current_spread_cents is not None
            and current_spread_cents > cfg.KALSHI_MAX_SPREAD_CENTS):
        status.trading_allowed = False
        status.active_breakers.append(
            f"CB3_illiquid: spread={current_spread_cents:.1f}¢ "
            f"max={cfg.KALSHI_MAX_SPREAD_CENTS}¢"
        )

    # CB4 — Settlement proximity + model uncertainty
    if (lead_hours_to_settlement is not None
            and lead_hours_to_settlement < (20 / 60)   # 20 min
            and al3x_uncertainty_f is not None
            and al3x_uncertainty_f > 2.0):
        status.trading_allowed = False
        status.active_breakers.append(
            f"CB4_settlement_proximity: "
            f"lead={lead_hours_to_settlement:.2f}h "
            f"uncertainty={al3x_uncertainty_f:.1f}°F"
        )

    # CB5 — API feed health
    if not feed_healthy:
        status.trading_allowed = False
        status.size_multiplier = 0.0
        status.active_breakers.append("CB5_feed_unhealthy: stale orderbook data")
        log.warning("CB5: Kalshi feed unhealthy — halting new entries")

    # CB6 — Prior day model accuracy
    if (last_forecast_error_f is not None
            and abs(last_forecast_error_f) > 4.0):
        if status.size_multiplier > 0.7:
            status.size_multiplier *= 0.7
        status.warnings.append(
            f"CB6_prior_day_error: yesterday error="
            f"{last_forecast_error_f:+.1f}°F — size reduced 30%%"
        )
        log.info("CB6: yesterday AL3X error %+.1f°F > 4°F — "
                 "size reduced 30%%", last_forecast_error_f)

    return status
