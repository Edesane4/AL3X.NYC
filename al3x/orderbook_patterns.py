"""Orderbook pattern detector — identifies 8 structural market patterns.

Each detection function takes recent snapshots and/or trade history
and returns a pattern result dict or None. All functions are pure
(no I/O, no side effects) — the calling code handles storage.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfg

log = logging.getLogger("al3x.patterns")


def _mid(snap: Dict[str, Any]) -> Optional[float]:
    b = snap.get("best_bid")
    a = snap.get("best_ask")
    if b is not None and a is not None:
        return round((b + a) / 2, 4)
    return None


def _total_bid_qty(snap: Dict[str, Any]) -> int:
    return sum(lv.get("qty", 0) for lv in (snap.get("bid_depth") or []))


def _total_ask_qty(snap: Dict[str, Any]) -> int:
    return sum(lv.get("qty", 0) for lv in (snap.get("ask_depth") or []))


# ---- Pattern 1: Spoofing / False Orders ---------------------------------

def detect_spoofing(snapshots: List[Dict[str, Any]]) -> Optional[Dict]:
    """Detect large resting orders that appear and vanish within 2 cycles.

    Requires at least 3 consecutive snapshots (T-2, T-1, T).
    Returns detection dict or None.
    """
    if len(snapshots) < 3:
        return None
    s0, s1, s2 = snapshots[-3], snapshots[-2], snapshots[-1]
    total_depth_s1 = _total_bid_qty(s1) + _total_ask_qty(s1)
    if total_depth_s1 == 0:
        return None

    threshold = cfg.KALSHI_SPOOF_SIZE_THRESHOLD_PCT * total_depth_s1

    spoof_signals = []
    # Check bid side: order appeared in s1, gone in s2
    for lv1 in (s1.get("bid_depth") or []):
        price1, qty1 = lv1.get("price"), lv1.get("qty", 0)
        if qty1 < threshold:
            continue
        # Was it in s0?
        in_s0 = any(abs(lv.get("price", 0) - price1) < 0.005
                    for lv in (s0.get("bid_depth") or []))
        # Is it in s2?
        in_s2 = any(abs(lv.get("price", 0) - price1) < 0.005
                    for lv in (s2.get("bid_depth") or []))
        if not in_s0 and not in_s2:
            spoof_signals.append({
                "side": "bid", "price": price1, "qty": qty1,
                "pct_of_book": round(qty1 / total_depth_s1, 3),
            })

    # Check ask side
    for lv1 in (s1.get("ask_depth") or []):
        price1, qty1 = lv1.get("price"), lv1.get("qty", 0)
        if qty1 < threshold:
            continue
        in_s0 = any(abs(lv.get("price", 0) - price1) < 0.005
                    for lv in (s0.get("ask_depth") or []))
        in_s2 = any(abs(lv.get("price", 0) - price1) < 0.005
                    for lv in (s2.get("ask_depth") or []))
        if not in_s0 and not in_s2:
            spoof_signals.append({
                "side": "ask", "price": price1, "qty": qty1,
                "pct_of_book": round(qty1 / total_depth_s1, 3),
            })

    if not spoof_signals:
        return None

    bid_spoofs = [s for s in spoof_signals if s["side"] == "bid"]
    ask_spoofs = [s for s in spoof_signals if s["side"] == "ask"]
    implied_direction = ("bearish" if bid_spoofs else
                         "bullish" if ask_spoofs else "mixed")
    return {
        "pattern_type": "spoofing",
        "signals": spoof_signals,
        "implied_direction": implied_direction,
        "market_ticker": s0.get("market_ticker", ""),
    }


# ---- Pattern 2: Human Panic ---------------------------------------------

def detect_panic(trades: List[Dict[str, Any]],
                  snapshots: List[Dict[str, Any]]) -> Optional[Dict]:
    """Detect panic buying or selling: high velocity + directional imbalance.

    trades:    recent trades for the past 30 minutes, chronological
    snapshots: recent snapshots, most recent last
    """
    if len(trades) < 5 or len(snapshots) < 4:
        return None

    now_utc = datetime.now(timezone.utc)
    window_30 = now_utc - timedelta(minutes=30)
    window_5 = now_utc - timedelta(minutes=5)

    def _parse_ts(ts: str) -> Optional[datetime]:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    trades_30 = [t for t in trades
                 if (_parse_ts(t["executed_at"]) or now_utc) >= window_30]
    trades_5 = [t for t in trades
                if (_parse_ts(t["executed_at"]) or now_utc) >= window_5]

    if not trades_30:
        return None

    # Velocity: trades per minute in last 5 min vs last 30 min
    velocity_30 = len(trades_30) / 30.0
    velocity_5 = len(trades_5) / 5.0 if trades_5 else 0.0
    multiplier = cfg.KALSHI_PANIC_VELOCITY_MULTIPLIER

    if velocity_5 < velocity_30 * multiplier:
        return None  # not elevated velocity

    # Directional imbalance in last 5 minutes
    buy_vol = sum(t["count"] for t in trades_5 if t["side"] == "yes")
    sell_vol = sum(t["count"] for t in trades_5 if t["side"] == "no")
    total_vol = buy_vol + sell_vol
    if total_vol == 0:
        return None
    imbalance = (buy_vol - sell_vol) / total_vol

    if abs(imbalance) < 0.65:
        return None  # not strongly directional

    direction = "panic_buying" if imbalance > 0 else "panic_selling"
    return {
        "pattern_type": "panic",
        "direction": direction,
        "velocity_5min": round(velocity_5, 2),
        "velocity_30min_baseline": round(velocity_30, 2),
        "velocity_multiplier": round(velocity_5 / max(velocity_30, 0.01), 1),
        "directional_imbalance": round(imbalance, 3),
        "buy_volume_5min": buy_vol,
        "sell_volume_5min": sell_vol,
        "market_ticker": snapshots[-1].get("market_ticker", ""),
    }


# ---- Pattern 3: Smart Money ---------------------------------------------

def detect_smart_money(snapshots: List[Dict[str, Any]]) -> Optional[Dict]:
    """Detect patient, large resting orders persisting for >= N cycles.

    A large order that rests without chasing price is a sign of
    informed, patient capital.
    """
    min_cycles = cfg.KALSHI_SMART_MONEY_MIN_CYCLES
    if len(snapshots) < min_cycles:
        return None

    window = snapshots[-min_cycles:]
    first_snap = window[0]
    last_snap = window[-1]
    total_depth = _total_bid_qty(last_snap) + _total_ask_qty(last_snap)
    if total_depth == 0:
        return None
    threshold = cfg.KALSHI_SPOOF_SIZE_THRESHOLD_PCT * total_depth

    smart_signals = []
    # Orders in the first snapshot that persist to the last snapshot
    for side, depth_key in (("bid", "bid_depth"), ("ask", "ask_depth")):
        for lv_first in (first_snap.get(depth_key) or []):
            pf, qf = lv_first.get("price"), lv_first.get("qty", 0)
            if qf < threshold:
                continue
            # Must be present in ALL cycles with similar qty (±20%)
            persistent = True
            absorbed_qty = 0
            for snap in window[1:]:
                match = next(
                    (lv for lv in (snap.get(depth_key) or [])
                     if abs(lv.get("price", 0) - pf) < 0.005),
                    None,
                )
                if not match:
                    persistent = False
                    break
                absorbed_qty = max(0, qf - match.get("qty", 0))
            if persistent:
                smart_signals.append({
                    "side": side,
                    "price": pf,
                    "original_qty": qf,
                    "absorbed_qty": absorbed_qty,
                    "pct_of_book": round(qf / total_depth, 3),
                    "accumulating": absorbed_qty > 0,
                })

    if not smart_signals:
        return None

    bid_signals = [s for s in smart_signals if s["side"] == "bid"]
    direction = "accumulation" if bid_signals else "distribution"
    return {
        "pattern_type": "smart_money",
        "direction": direction,
        "signals": smart_signals,
        "cycles_observed": min_cycles,
        "market_ticker": last_snap.get("market_ticker", ""),
    }


# ---- Pattern 4: Stale Price (Wagers Against Fresh Data) -----------------

def detect_stale_price(snapshots: List[Dict[str, Any]],
                        fair_value: float,
                        min_persist_cycles: int = 2
                        ) -> Optional[Dict]:
    """Detect when the market price has not updated after an AL3X
    forecast revision — wagers going against the latest weather data.

    Requires that the divergence between fair_value and the market
    mid-price exceeds the minimum edge threshold for at least
    min_persist_cycles consecutive snapshots.
    """
    if len(snapshots) < min_persist_cycles:
        return None
    window = snapshots[-min_persist_cycles:]
    min_edge = cfg.KALSHI_MIN_EDGE_CENTS / 100.0

    divergent_cycles = 0
    last_mid = None
    for snap in window:
        mid = _mid(snap)
        if mid is None:
            continue
        if abs(fair_value - mid) >= min_edge:
            divergent_cycles += 1
        last_mid = mid

    if divergent_cycles < min_persist_cycles or last_mid is None:
        return None

    direction = "yes" if fair_value > last_mid else "no"
    edge_cents = round((fair_value - last_mid) * 100, 1)
    return {
        "pattern_type": "stale_price",
        "fair_value": fair_value,
        "market_mid": last_mid,
        "edge_cents": edge_cents,
        "divergent_direction": direction,
        "persist_cycles": divergent_cycles,
        "market_ticker": snapshots[-1].get("market_ticker", ""),
    }


# ---- Pattern 5: Spread Expansion (Uncertainty Signal) -------------------

def detect_spread_expansion(snapshots: List[Dict[str, Any]],
                              lookback_minutes: int = 30
                              ) -> Optional[Dict]:
    """Detect when current spread is > 2× the rolling mean spread.
    High market uncertainty when our model is certain = edge.
    """
    if len(snapshots) < 4:
        return None
    spreads = []
    for snap in snapshots:
        b = snap.get("best_bid")
        a = snap.get("best_ask")
        if b is not None and a is not None:
            spreads.append((a - b) * 100)  # in cents

    if not spreads:
        return None

    mean_spread = sum(spreads[:-1]) / len(spreads[:-1])
    current_spread = spreads[-1]
    if mean_spread <= 0:
        return None

    ratio = current_spread / mean_spread
    if ratio < 2.0:
        return None

    return {
        "pattern_type": "spread_expansion",
        "current_spread_cents": round(current_spread, 1),
        "mean_spread_cents": round(mean_spread, 1),
        "expansion_ratio": round(ratio, 2),
        "market_ticker": snapshots[-1].get("market_ticker", ""),
    }


# ---- Pattern 6: Volume-Price Divergence (Exhaustion) --------------------

def detect_exhaustion(trades: List[Dict[str, Any]],
                       snapshots: List[Dict[str, Any]]) -> Optional[Dict]:
    """Detect when price is moving but volume per trade is declining.
    Price exhaustion against our forecast is a high-confidence entry.
    """
    if len(trades) < 10 or len(snapshots) < 4:
        return None

    # Split last 10 minutes into two 5-min windows
    now_utc = datetime.now(timezone.utc)

    def _parse_ts(ts: str) -> datetime:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except Exception:
            return now_utc

    early = [t for t in trades
             if now_utc - timedelta(minutes=10) <=
                _parse_ts(t["executed_at"]) <
                now_utc - timedelta(minutes=5)]
    late = [t for t in trades
            if _parse_ts(t["executed_at"]) >= now_utc - timedelta(minutes=5)]

    if not early or not late:
        return None

    avg_qty_early = sum(t["count"] for t in early) / len(early)
    avg_qty_late = sum(t["count"] for t in late) / len(late)
    if avg_qty_early <= 0:
        return None

    qty_decline_pct = (avg_qty_early - avg_qty_late) / avg_qty_early

    # Price movement in the same window
    mids = [_mid(s) for s in snapshots[-4:] if _mid(s) is not None]
    if len(mids) < 2:
        return None
    price_move = abs(mids[-1] - mids[0]) * 100  # cents

    if qty_decline_pct < 0.30 or price_move < 2.0:
        return None  # not a clear exhaustion

    direction = "up" if mids[-1] > mids[0] else "down"
    return {
        "pattern_type": "exhaustion",
        "price_move_direction": direction,
        "price_move_cents": round(price_move, 1),
        "avg_qty_early": round(avg_qty_early, 1),
        "avg_qty_late": round(avg_qty_late, 1),
        "volume_decline_pct": round(qty_decline_pct * 100, 1),
        "market_ticker": snapshots[-1].get("market_ticker", ""),
    }


# ---- Pattern 7: Cross-Market Monotonicity Violation --------------------
# Implemented in fair_value.py as detect_monotonicity_violation()
# Referenced here for completeness in the pattern taxonomy.


# ---- Pattern 8: Time Decay / Running Max Arbitrage ---------------------

def detect_running_max_arb(market_ticker: str,
                             threshold_f: float,
                             snapshot: Dict[str, Any],
                             running_max_f: Optional[float],
                             current_hour_eastern: int
                             ) -> Optional[Dict]:
    """After 1:30 PM, if ASOS running max >= threshold and YES is below
    KALSHI_RUNNING_MAX_CERTAINTY_THRESHOLD, flag as running-max arb.
    """
    if current_hour_eastern < 13 or running_max_f is None:
        return None
    if running_max_f < threshold_f:
        return None
    ask = snapshot.get("best_ask")
    if ask is None:
        return None
    threshold_price = cfg.KALSHI_RUNNING_MAX_CERTAINTY_THRESHOLD
    if ask >= threshold_price:
        return None  # already priced correctly

    return {
        "pattern_type": "running_max_arb",
        "threshold_f": threshold_f,
        "running_max_f": running_max_f,
        "yes_ask": ask,
        "edge_cents": round((threshold_price - ask) * 100, 1),
        "market_ticker": market_ticker,
    }


# ---- Master scan --------------------------------------------------------

def scan_all_patterns(
        market_ticker: str,
        threshold_f: Optional[float],
        snapshots: List[Dict[str, Any]],
        trades: List[Dict[str, Any]],
        fair_value: float,
        running_max_f: Optional[float],
        current_hour_eastern: int,
) -> List[Dict[str, Any]]:
    """Run all single-market pattern detectors and return any hits.

    Returns a list of pattern dicts (may be empty).
    """
    detected = []

    # Pattern 1: spoofing
    r = detect_spoofing(snapshots)
    if r:
        detected.append(r)

    # Pattern 2: panic
    r = detect_panic(trades, snapshots)
    if r:
        detected.append(r)

    # Pattern 3: smart money
    r = detect_smart_money(snapshots)
    if r:
        detected.append(r)

    # Pattern 4: stale price
    r = detect_stale_price(snapshots, fair_value)
    if r:
        detected.append(r)

    # Pattern 5: spread expansion
    r = detect_spread_expansion(snapshots)
    if r:
        detected.append(r)

    # Pattern 6: exhaustion
    r = detect_exhaustion(trades, snapshots)
    if r:
        detected.append(r)

    # Pattern 8: running max arb
    if threshold_f is not None and snapshots:
        r = detect_running_max_arb(
            market_ticker, threshold_f, snapshots[-1],
            running_max_f, current_hour_eastern,
        )
        if r:
            detected.append(r)

    return detected
