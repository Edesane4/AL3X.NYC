"""Kalshi market intelligence engine — main orchestrator.

Runs the 30-second polling loop, pattern detection, fair value
comparison, strategy evaluation, risk checking, and order execution
(paper or live). Called by the scheduler.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import config as cfg
from .fair_value import (compute_all_fair_values, estimate_sigma,
                          detect_monotonicity_violation)
from .kalshi_feed import KalshiFeed
from .kalshi_strategies import (strategy_model_divergence,
                                 strategy_running_max_arb,
                                 strategy_monotonicity_arb)
from .kelly_sizer import compute_unrealized_pnl
from .orderbook_patterns import scan_all_patterns
from .risk_controls import check_all, RiskStatus
from .storage import Storage

log = logging.getLogger("al3x.kalshi_engine")


class KalshiEngine:
    """Orchestrates one full scan cycle: fetch → compute → decide → act."""

    def __init__(self, storage: Storage, feed: KalshiFeed) -> None:
        self.storage = storage
        self.feed = feed
        self._consecutive_feed_errors: int = 0
        self._last_forecast: Optional[Dict[str, Any]] = None
        self._paper_mode: bool = cfg.KALSHI_PAPER_MODE

    def update_forecast(self, forecast: Dict[str, Any]) -> None:
        """Called by the scheduler after each AL3X forecast cycle."""
        self._last_forecast = forecast

    async def run_cycle(self, target_date_str: str) -> Dict[str, Any]:
        """Execute one full scan cycle. Returns a summary dict.

        target_date_str: 'YYYY-MM-DD' of the day whose high is being
        forecast and traded.
        """
        summary = {
            "target_date": target_date_str,
            "markets_scanned": 0,
            "patterns_detected": 0,
            "decisions_made": 0,
            "trades_placed": 0,
            "errors": [],
        }

        # ---- Step 1: Discover active markets ----------------------------
        markets = await self.feed.get_nyc_temp_markets(target_date_str)
        if not markets:
            self._consecutive_feed_errors += 1
            summary["errors"].append("no_markets_returned")
            return summary
        self._consecutive_feed_errors = 0
        summary["markets_scanned"] = len(markets)

        # ---- Step 2: Pull orderbooks and trades for all markets ---------
        snapshots: Dict[str, Dict[str, Any]] = {}
        for m in markets:
            ticker = m["market_ticker"]
            ob = await self.feed.get_orderbook(ticker)
            if ob:
                ob["market_ticker"] = ticker
                ob["threshold_f"] = m.get("threshold_f")
                ob["event_ticker"] = m.get("event_ticker", "")
                ob["time_to_close_sec"] = _time_to_close_sec(
                    m.get("close_time"))
                snapshots[ticker] = ob
                self.storage.save_kalshi_snapshot(ob)

            trades = await self.feed.get_recent_trades(ticker, limit=50)
            for t in trades:
                try:
                    self.storage.save_kalshi_trade(t)
                except Exception:  # noqa: BLE001
                    pass

        if not snapshots:
            self._consecutive_feed_errors += 1
            summary["errors"].append("no_snapshots_fetched")
            return summary

        # ---- Step 3: Extract AL3X forecast context ----------------------
        fc = self._last_forecast
        al3x_f = None
        sigma = 2.5
        running_max_f = None
        running_max_age_min = None
        al3x_uncertainty = None
        lead_hours = None

        if fc:
            al3x_f = fc.get("final_f")
            extras = fc.get("extras") or {}
            bma = extras.get("bma")
            qrf = extras.get("qrf")
            sigma = estimate_sigma(bma, qrf)
            running_max_f = fc.get("running_asos_max_f")
            lead_hours = extras.get("lead_hours")
            qrf_width = (qrf.get("interval_width") if qrf else None)
            al3x_uncertainty = (qrf_width / 2.0
                                 if qrf_width else sigma)
            # Approximate age of running max from forecast issued_at
            if running_max_f and fc.get("issued_at"):
                try:
                    issued = datetime.fromisoformat(fc["issued_at"])
                    now = datetime.now(issued.tzinfo or timezone.utc)
                    running_max_age_min = (
                        (now - issued).total_seconds() / 60.0)
                except Exception:  # noqa: BLE001
                    running_max_age_min = None

        if al3x_f is None:
            summary["errors"].append("no_al3x_forecast")
            return summary

        # ---- Step 4: Compute fair values --------------------------------
        fair_values = compute_all_fair_values(
            markets, al3x_f, sigma, running_max_f, running_max_age_min,
        )

        # ---- Step 5: Load state for risk and sizing ---------------------
        bankroll = self.storage.get_or_create_bankroll(self._paper_mode)
        open_positions = self.storage.open_kalshi_positions()
        feed_healthy = self._consecutive_feed_errors < 3

        # Yesterday's forecast error (for CB6)
        last_error = _get_yesterday_error(self.storage)

        now_eastern = datetime.now(cfg.EASTERN)
        current_hour = now_eastern.hour

        # ---- Step 6: Pattern detection + strategy evaluation ------------
        decisions: List[Dict[str, Any]] = []

        for ticker, snap in snapshots.items():
            fv_result = fair_values.get(ticker)
            if fv_result is None:
                continue
            fv = float(fv_result["fair_value"])
            threshold_f = snap.get("threshold_f")

            # Pull recent snapshots for pattern detection
            recent_snaps = self.storage.recent_snapshots(ticker, limit=10)
            recent_trades = self.storage.recent_trades_for_ticker(ticker)

            # Pattern scan
            patterns = scan_all_patterns(
                market_ticker=ticker,
                threshold_f=threshold_f,
                snapshots=recent_snaps,
                trades=recent_trades,
                fair_value=fv,
                running_max_f=running_max_f,
                current_hour_eastern=current_hour,
            )
            for p in patterns:
                self.storage.save_pattern_detection({
                    "market_ticker": ticker,
                    "pattern_type": p["pattern_type"],
                    "details": p,
                    "al3x_fair_value": fv,
                    "market_price": (snap.get("best_bid", 0)
                                      + snap.get("best_ask", 0)) / 2,
                    "edge_cents": p.get("edge_cents"),
                })
            summary["patterns_detected"] += len(patterns)

            # Risk check (per market)
            spread_cents = (
                (snap.get("best_ask", 1) - (snap.get("best_bid") or 0))
                * 100
            )
            time_to_close = snap.get("time_to_close_sec")
            lead_to_close = (time_to_close / 3600.0
                              if time_to_close else None)
            risk = check_all(
                bankroll=bankroll,
                open_positions=open_positions,
                feed_healthy=feed_healthy,
                last_forecast_error_f=last_error,
                current_spread_cents=spread_cents,
                lead_hours_to_settlement=lead_to_close,
                al3x_uncertainty_f=al3x_uncertainty,
            )
            if risk.halted:
                log.info("Risk halted on %s: %s",
                         ticker, risk.active_breakers)
                continue

            # Strategy 1: Model Divergence
            d1 = strategy_model_divergence(
                market_ticker=ticker,
                threshold_f=threshold_f or 0.0,
                snapshot=snap,
                fair_value_result=fv_result,
                recent_snapshots=recent_snaps,
                open_positions=open_positions,
                bankroll=bankroll,
                pattern_hits=patterns,
            )
            if d1:
                d1["size_multiplier"] = risk.size_multiplier
                d1["contracts"] = max(
                    1, int(d1["contracts"] * risk.size_multiplier))
                d1["cost_basis"] = round(
                    d1["contracts"] * d1["entry_price"], 4)
                decisions.append(d1)

            # Strategy 2: Running Max Arb
            d2 = strategy_running_max_arb(
                market_ticker=ticker,
                threshold_f=threshold_f or 0.0,
                snapshot=snap,
                running_max_f=running_max_f,
                running_max_age_minutes=running_max_age_min,
                open_positions=open_positions,
                bankroll=bankroll,
                current_hour_eastern=current_hour,
            )
            if d2:
                d2["size_multiplier"] = risk.size_multiplier
                d2["contracts"] = max(
                    1, int(d2["contracts"] * risk.size_multiplier))
                d2["cost_basis"] = round(
                    d2["contracts"] * d2["entry_price"], 4)
                decisions.append(d2)

        # Strategy 3: Cross-market arb (runs across all markets at once)
        arb_decisions = strategy_monotonicity_arb(
            fair_values=fair_values,
            snapshots_by_ticker=snapshots,
            open_positions=open_positions,
            bankroll=bankroll,
        )
        for arb in arb_decisions:
            # Global risk check for arb
            risk_arb = check_all(
                bankroll=bankroll,
                open_positions=open_positions,
                feed_healthy=feed_healthy,
                last_forecast_error_f=last_error,
                current_spread_cents=None,
                lead_hours_to_settlement=lead_hours,
                al3x_uncertainty_f=al3x_uncertainty,
            )
            if not risk_arb.halted:
                decisions.append(arb)

        summary["decisions_made"] = len(decisions)

        # ---- Step 7: Execute decisions (paper or live) ------------------
        for decision in decisions:
            placed = await self._execute(decision, bankroll)
            if placed:
                summary["trades_placed"] += 1
                # Refresh bankroll after each placement
                bankroll = self.storage.get_or_create_bankroll(
                    self._paper_mode)

        # ---- Step 8: Update unrealized P&L ------------------------------
        current_prices = {
            tk: (s.get("best_bid", 0) + s.get("best_ask", 0)) / 2
            for tk, s in snapshots.items()
        }
        open_pos = self.storage.open_kalshi_positions()
        unrealized = compute_unrealized_pnl(open_pos, current_prices)
        self.storage.update_bankroll(
            self._paper_mode, {"unrealized_pnl": unrealized})

        # Purge old snapshots weekly (cheap no-op on most cycles)
        try:
            self.storage.purge_old_kalshi_snapshots(days=7)
        except Exception:  # noqa: BLE001
            pass

        return summary

    async def _execute(self, decision: Dict[str, Any],
                        bankroll: Dict[str, Any]) -> bool:
        """Execute a single decision (paper log or live order).

        For arb decisions, executes both legs atomically in paper mode
        or sequentially in live mode.

        Returns True if the trade was logged/placed successfully.
        """
        try:
            # Arb decision has "legs" instead of direct fields
            if decision.get("strategy") == "arb":
                legs = decision.get("legs", [])
                if len(legs) != 2:
                    return False
                ok = True
                for leg in legs:
                    ok = ok and await self._execute_single(leg, bankroll)
                return ok
            return await self._execute_single(decision, bankroll)
        except Exception as e:  # noqa: BLE001
            log.warning("Execute failed: %s", e)
            return False

    async def _execute_single(self, d: Dict[str, Any],
                                bankroll: Dict[str, Any]) -> bool:
        ticker = d["market_ticker"]
        side = d["side"]
        contracts = d["contracts"]
        entry_price = d["entry_price"]
        cost_basis = d["cost_basis"]
        strategy = d["strategy"]

        if self._paper_mode or not cfg.KALSHI_LIVE_TRADING:
            # Paper trade: log the intended action
            pos_id = self.storage.save_kalshi_position({
                "market_ticker": ticker,
                "threshold_f": d.get("threshold_f"),
                "side": side,
                "entry_price": entry_price,
                "contracts": contracts,
                "cost_basis": cost_basis,
                "strategy": strategy,
                "entry_ev": d.get("ev"),
                "entry_edge_cents": d.get("edge_cents"),
                "al3x_fair_value": d.get("al3x_fair_value"),
                "paper_trade": 1,
                "notes": f"paper | {strategy}",
            })
            # Update bankroll deployed
            new_deployed = (float(bankroll.get("total_deployed", 0))
                            + cost_basis)
            self.storage.update_bankroll(True, {"total_deployed": new_deployed})
            log.info("[PAPER] %s %s %s x%d @ %.4f (EV=%.3f)",
                     strategy.upper(), side.upper(), ticker,
                     contracts, entry_price, d.get("ev", 0))
            return True
        else:
            # Live order
            import uuid
            client_id = f"al3x-{uuid.uuid4().hex[:8]}"
            result = await self.feed.place_order(
                market_ticker=ticker,
                side=side,
                action="buy",
                contracts=contracts,
                limit_price=entry_price,
                client_order_id=client_id,
            )
            if result:
                pos_id = self.storage.save_kalshi_position({
                    "market_ticker": ticker,
                    "threshold_f": d.get("threshold_f"),
                    "side": side,
                    "entry_price": entry_price,
                    "contracts": contracts,
                    "cost_basis": cost_basis,
                    "strategy": strategy,
                    "entry_ev": d.get("ev"),
                    "entry_edge_cents": d.get("edge_cents"),
                    "al3x_fair_value": d.get("al3x_fair_value"),
                    "paper_trade": 0,
                    "notes": f"live | order_id={client_id}",
                })
                new_deployed = (float(bankroll.get("total_deployed", 0))
                                + cost_basis)
                self.storage.update_bankroll(
                    False, {"total_deployed": new_deployed})
                log.info("[LIVE] %s %s %s x%d @ %.4f placed",
                         strategy.upper(), side.upper(), ticker,
                         contracts, entry_price)
                return True
            log.warning("[LIVE] Order failed for %s %s %s",
                        strategy, side, ticker)
            return False


# ---- Helpers ------------------------------------------------------------

def _time_to_close_sec(close_time_str: Optional[str]) -> Optional[int]:
    if not close_time_str:
        return None
    try:
        close = datetime.fromisoformat(
            close_time_str.replace("Z", "+00:00"))
        if close.tzinfo is None:
            close = close.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0, int((close - now).total_seconds()))
    except Exception:
        return None


def _get_yesterday_error(storage: Storage) -> Optional[float]:
    """Get the most recent verified forecast error from storage."""
    try:
        scores = storage.recent_scores(days=2)
        intraday = [s for s in scores if s.get("mode") == "intraday"]
        if intraday:
            return float(intraday[0]["abs_error_f"])
    except Exception:  # noqa: BLE001
        pass
    return None
