"""The three core trading strategies for AL3X.NYC's Kalshi engine.

Strategy 1 — Model Divergence: Enter when AL3X fair value diverges
             from market price by >= min edge for >= N cycles.
Strategy 2 — Running Max Arbitrage: Buy YES when ASOS has confirmed
             the threshold was reached and the contract is below 0.95.
Strategy 3 — Cross-Market Monotonicity Arbitrage: Paired trade when
             adjacent threshold contracts have inverted probabilities.

All strategies return a Decision dict or None. The decision dict
describes WHAT to do (buy/skip) but does NOT execute — the engine
layer executes. This separation allows paper trading to use the
same strategy logic as live trading.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import config as cfg
from .fair_value import best_side, compute_edge, detect_monotonicity_violation
from .kelly_sizer import size_position

log = logging.getLogger("al3x.strategies")


def _get_exposure(open_positions: List[Dict[str, Any]],
                   market_ticker: str,
                   bankroll: Dict[str, Any]) -> Dict[str, float]:
    """Return current exposure totals for position sizing."""
    total = float(bankroll.get("total_deployed", 0))
    threshold_exp = sum(
        float(p.get("cost_basis", 0)) for p in open_positions
        if p.get("market_ticker") == market_ticker
    )
    return {"total": total, "threshold": threshold_exp}


# ---- Strategy 1: Model Divergence ---------------------------------------

def strategy_model_divergence(
        market_ticker: str,
        threshold_f: float,
        snapshot: Dict[str, Any],
        fair_value_result: Dict[str, Any],
        recent_snapshots: List[Dict[str, Any]],
        open_positions: List[Dict[str, Any]],
        bankroll: Dict[str, Any],
        pattern_hits: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Strategy 1: buy when AL3X price diverges from market for N+ cycles.

    Returns a decision dict or None (no trade).
    """
    fv = float(fair_value_result.get("fair_value", 0.5))
    best_bid = snapshot.get("best_bid")
    best_ask = snapshot.get("best_ask")

    if best_bid is None or best_ask is None:
        return None

    # Spread check
    spread_cents = (best_ask - best_bid) * 100
    if spread_cents > cfg.KALSHI_MAX_SPREAD_CENTS:
        return None

    # Determine best side and edge
    edge = best_side(fv, best_bid, best_ask)
    if edge is None:
        return None

    edge_cents = abs(edge["edge_cents"])
    if edge_cents < cfg.KALSHI_MIN_EDGE_CENTS:
        return None

    # Net-of-fee EV check: subtract expected fee cost from EV
    # Fee is paid on winning side only: P_win × fee_per_contract
    p_win = fv if edge["side"] == "yes" else (1.0 - fv)
    net_ev = edge["ev"] - (p_win * cfg.KALSHI_FEE_PER_CONTRACT)
    if net_ev < cfg.KALSHI_MIN_NET_EV:
        return None

    # Check persistence: edge must have existed in prior cycles too
    persist_count = 1
    min_persist = cfg.KALSHI_MIN_EDGE_PERSIST_CYCLES
    for snap in reversed(recent_snapshots[:-1]):
        b = snap.get("best_bid")
        a = snap.get("best_ask")
        if b is None or a is None:
            break
        mid = (b + a) / 2
        if abs(fv - mid) * 100 >= cfg.KALSHI_MIN_EDGE_CENTS:
            persist_count += 1
        else:
            break
        if persist_count >= min_persist:
            break

    if persist_count < min_persist:
        return None

    # Pattern amplifiers — increase confidence
    confidence_boost = 0.0
    amplifiers = []
    for p in pattern_hits:
        ptype = p.get("pattern_type", "")
        pdir = p.get("direction", "") or p.get("implied_direction", "")
        # Panic in our direction increases confidence
        if ptype == "panic":
            if (edge["side"] == "no" and "selling" in pdir) or \
               (edge["side"] == "yes" and "buying" in pdir):
                confidence_boost += 0.05
                amplifiers.append("panic_confirms")
        # Stale price = direct confirmation
        if ptype == "stale_price":
            confidence_boost += 0.03
            amplifiers.append("stale_price")
        # Smart money in our direction
        if ptype == "smart_money":
            if (edge["side"] == "yes" and "accum" in pdir) or \
               (edge["side"] == "no" and "distrib" in pdir):
                confidence_boost += 0.04
                amplifiers.append("smart_money_aligns")

    # Apply confidence boost to Kelly sizing.
    # confidence_boost is in [0, 0.12] from the amplifiers above.
    # We apply it as a mild multiplier on the fractional Kelly:
    # at max boost (0.12), size increases by ~12%.
    # This is bounded so it never exceeds 1.5× normal sizing.
    kelly_boost_multiplier = min(1.5, 1.0 + confidence_boost)

    # Size the position
    exposure = _get_exposure(open_positions, market_ticker, bankroll)
    entry_price = (best_ask if edge["side"] == "yes"
                    else best_bid)
    current_bankroll = float(bankroll.get("current_bankroll",
                                           cfg.KALSHI_STARTING_BANKROLL))
    sizing = size_position(
        fair_value=fv,
        entry_price=entry_price,
        side=edge["side"],
        current_bankroll=current_bankroll,
        current_total_exposure=exposure["total"],
        current_threshold_exposure=exposure["threshold"],
        strategy="model_divergence",
    )
    # Apply confidence boost to contract count (already Kelly-capped)
    if kelly_boost_multiplier > 1.0 and sizing["contracts"] > 0:
        boosted_contracts = int(sizing["contracts"] * kelly_boost_multiplier)
        # Re-check single position cap after boost
        max_contracts_by_cap = int(
            (current_bankroll * cfg.KALSHI_MAX_SINGLE_POSITION_PCT)
            / entry_price
        )
        sizing = dict(sizing)
        sizing["contracts"] = min(boosted_contracts, max_contracts_by_cap)
        sizing["cost_basis"] = round(
            sizing["contracts"] * entry_price, 4)

    if sizing["contracts"] == 0:
        return None

    return {
        "strategy": "model_divergence",
        "action": "buy",
        "market_ticker": market_ticker,
        "threshold_f": threshold_f,
        "side": edge["side"],
        "entry_price": entry_price,
        "contracts": sizing["contracts"],
        "cost_basis": sizing["cost_basis"],
        "al3x_fair_value": fv,
        "edge_cents": edge_cents,
        "ev": edge["ev"],
        "persist_cycles": persist_count,
        "confidence_boost": confidence_boost,
        "amplifiers": amplifiers,
        "sizing_detail": sizing,
    }


# ---- Strategy 2: Running Max Arbitrage ----------------------------------

def strategy_running_max_arb(
        market_ticker: str,
        threshold_f: float,
        snapshot: Dict[str, Any],
        running_max_f: Optional[float],
        running_max_age_minutes: Optional[float],
        open_positions: List[Dict[str, Any]],
        bankroll: Dict[str, Any],
        current_hour_eastern: int,
) -> Optional[Dict[str, Any]]:
    """Strategy 2: buy YES when ASOS has already cleared the threshold.

    Requires:
      - Current hour >= 13 (1 PM Eastern)
      - running_max_f >= threshold_f
      - running_max age >= KALSHI_RUNNING_MAX_MIN_AGE_MINUTES
      - YES ask < KALSHI_RUNNING_MAX_CERTAINTY_THRESHOLD
    """
    if current_hour_eastern < 13:
        return None
    if running_max_f is None or running_max_f < threshold_f:
        return None
    min_age = cfg.KALSHI_RUNNING_MAX_MIN_AGE_MINUTES
    if (running_max_age_minutes is not None
            and running_max_age_minutes < min_age):
        return None

    best_ask = snapshot.get("best_ask")
    if best_ask is None:
        return None
    certainty = cfg.KALSHI_RUNNING_MAX_CERTAINTY_THRESHOLD
    if best_ask >= certainty:
        return None  # already correctly priced

    # This is near-certain: fair value ≈ 0.99
    fair_value = 0.99
    edge_cents = round((certainty - best_ask) * 100, 1)
    ev = compute_edge(fair_value, best_ask, "yes")["ev"]

    # Running max has near-certain win probability
    net_ev_arb = ev - (0.99 * cfg.KALSHI_FEE_PER_CONTRACT)
    if net_ev_arb < cfg.KALSHI_MIN_NET_EV:
        return None

    exposure = _get_exposure(open_positions, market_ticker, bankroll)
    current_bankroll = float(bankroll.get("current_bankroll",
                                           cfg.KALSHI_STARTING_BANKROLL))
    sizing = size_position(
        fair_value=fair_value,
        entry_price=best_ask,
        side="yes",
        current_bankroll=current_bankroll,
        current_total_exposure=exposure["total"],
        current_threshold_exposure=exposure["threshold"],
        strategy="running_max",
    )

    if sizing["contracts"] == 0:
        return None

    return {
        "strategy": "running_max",
        "action": "buy",
        "market_ticker": market_ticker,
        "threshold_f": threshold_f,
        "side": "yes",
        "entry_price": best_ask,
        "contracts": sizing["contracts"],
        "cost_basis": sizing["cost_basis"],
        "al3x_fair_value": fair_value,
        "edge_cents": edge_cents,
        "ev": ev,
        "running_max_f": running_max_f,
        "running_max_age_min": running_max_age_minutes,
        "sizing_detail": sizing,
    }


# ---- Strategy 3: Monotonicity Arbitrage ---------------------------------

def strategy_monotonicity_arb(
        fair_values: Dict[str, Dict[str, Any]],
        snapshots_by_ticker: Dict[str, Dict[str, Any]],
        open_positions: List[Dict[str, Any]],
        bankroll: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Strategy 3: paired trade when adjacent threshold contracts have
    inverted implied probabilities.

    Returns a list of decision dicts (pairs). Each pair has two legs:
    [buy_lower_yes, buy_upper_no].
    """
    violations = detect_monotonicity_violation(fair_values, snapshots_by_ticker)
    if not violations:
        return []

    current_bankroll = float(bankroll.get("current_bankroll",
                                           cfg.KALSHI_STARTING_BANKROLL))
    total_exposure = float(bankroll.get("total_deployed", 0))
    decisions = []

    for v in violations:
        # Leg 1: buy YES on lower threshold (underpriced)
        lower_ask = v.get("buy_lower_yes_at")
        upper_no_price = v.get("buy_upper_no_at")  # 1 - bid_high
        if lower_ask is None or upper_no_price is None:
            continue
        if lower_ask >= 1.0 or upper_no_price >= 1.0:
            continue

        # Spread check on both legs
        lower_snap = snapshots_by_ticker.get(v["lower_ticker"], {})
        upper_snap = snapshots_by_ticker.get(v["upper_ticker"], {})
        lower_spread = ((lower_snap.get("best_ask", 1) -
                         (lower_snap.get("best_bid") or 0)) * 100)
        upper_spread = ((upper_snap.get("best_ask", 1) -
                         (upper_snap.get("best_bid") or 0)) * 100)
        if (lower_spread > cfg.KALSHI_MAX_SPREAD_CENTS
                or upper_spread > cfg.KALSHI_MAX_SPREAD_CENTS):
            continue

        # Use the worst-leg spread for CB3 evaluation in the caller.
        # Store it in the decision so the engine can pass it to check_all.
        worst_spread_cents = max(lower_spread, upper_spread)

        # Size each leg at half the arb allocation
        fv_lower = (fair_values.get(v["lower_ticker"]) or {}).get(
            "fair_value", 0.55)
        fv_upper = (fair_values.get(v["upper_ticker"]) or {}).get(
            "fair_value", 0.45)

        sz_lower = size_position(
            fair_value=float(fv_lower),
            entry_price=lower_ask,
            side="yes",
            current_bankroll=current_bankroll,
            current_total_exposure=total_exposure,
            current_threshold_exposure=0.0,
            strategy="arb",
        )
        sz_upper = size_position(
            fair_value=float(fv_upper),
            entry_price=float(upper_no_price),
            side="no",
            current_bankroll=current_bankroll,
            current_total_exposure=total_exposure + sz_lower["cost_basis"],
            current_threshold_exposure=0.0,
            strategy="arb",
        )

        if sz_lower["contracts"] == 0 or sz_upper["contracts"] == 0:
            continue

        decisions.append({
            "strategy": "arb",
            "arb_cents": v["arb_cents"],
            "worst_spread_cents": worst_spread_cents,
            "legs": [
                {
                    "action": "buy",
                    "side": "yes",
                    "market_ticker": v["lower_ticker"],
                    "threshold_f": v["lower_threshold_f"],
                    "entry_price": lower_ask,
                    "contracts": sz_lower["contracts"],
                    "cost_basis": sz_lower["cost_basis"],
                    "al3x_fair_value": float(fv_lower),
                },
                {
                    "action": "buy",
                    "side": "no",
                    "market_ticker": v["upper_ticker"],
                    "threshold_f": v["upper_threshold_f"],
                    "entry_price": float(upper_no_price),
                    "contracts": sz_upper["contracts"],
                    "cost_basis": sz_upper["cost_basis"],
                    "al3x_fair_value": float(fv_upper),
                },
            ],
        })

    return decisions
