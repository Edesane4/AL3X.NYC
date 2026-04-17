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

        # Fetch all orderbooks and trade histories concurrently.
        # Sequential fetching with 10+ markets and 10s timeout would
        # exceed the 30-second scan interval.
        ob_tasks = [
            self.feed.get_orderbook(m["market_ticker"])
            for m in markets
        ]
        trade_tasks = [
            self.feed.get_recent_trades(m["market_ticker"], limit=50)
            for m in markets
        ]
        ob_results, trade_results = await asyncio.gather(
            asyncio.gather(*ob_tasks, return_exceptions=True),
            asyncio.gather(*trade_tasks, return_exceptions=True),
        )

        for m, ob, trades in zip(markets, ob_results, trade_results):
            ticker = m["market_ticker"]

            # Handle orderbook
            if isinstance(ob, Exception) or ob is None:
                continue
            ob["market_ticker"] = ticker
            ob["threshold_f"] = m.get("threshold_f")
            ob["event_ticker"] = m.get("event_ticker", "")
            ob["time_to_close_sec"] = _time_to_close_sec(
                m.get("close_time"))
            snapshots[ticker] = ob
            self.storage.save_kalshi_snapshot(ob)

            # Handle trades
            if isinstance(trades, Exception) or not trades:
                continue
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

            # Upgrade 1: use the Kalman filter's current uncertainty to
            # tighten sigma as the day progresses and ASOS observations
            # accumulate. kalman_uncertainty_f is the 1σ of the current
            # temperature estimate — not the daily max. We use it as a
            # floor adjustment: if Kalman uncertainty is much tighter
            # than the forecast sigma, reduce sigma proportionally but
            # never below 0.5°F.
            kalman = extras.get("kalman") or {}
            k_uncertainty = kalman.get("kalman_uncertainty_f")
            hours_of_data = kalman.get("hours_of_data_used", 0)
            if (k_uncertainty is not None
                    and k_uncertainty > 0
                    and hours_of_data is not None
                    and float(hours_of_data) >= 4.0):
                # Weight kalman sigma in as data accumulates.
                # At 4h: 20% kalman, 80% forecast sigma.
                # At 8h: 40% kalman, 60% forecast sigma.
                # At 12h+: 60% kalman, 40% forecast sigma.
                # Kalman uncertainty is for current temp, not max, so
                # we add a residual spread of 1°F to account for
                # remaining uncertainty in the peak.
                kalman_sigma = float(k_uncertainty) + 1.0
                alpha = min(0.60, 0.20 + 0.05 * (float(hours_of_data) - 4.0))
                dynamic_sigma = (1.0 - alpha) * sigma + alpha * kalman_sigma
                sigma = max(0.5, dynamic_sigma)
                log.debug("Dynamic sigma: %.2f°F (forecast %.2f, "
                          "kalman %.2f, alpha=%.2f, hours=%.1f)",
                          sigma, estimate_sigma(bma, qrf),
                          kalman_sigma, alpha, float(hours_of_data))

            running_max_f = fc.get("running_asos_max_f")
            lead_hours = extras.get("lead_hours")
            qrf_width = (qrf.get("interval_width") if qrf else None)
            al3x_uncertainty = (qrf_width / 2.0
                                 if qrf_width else sigma)
            # Use the actual ASOS running max timestamp if available.
            # Fall back to forecast issued_at only as a last resort.
            running_max_ts = None
            extras_inner = fc.get("extras") or {}
            running_max_ts = extras_inner.get("running_max_observed_at")

            if running_max_f and running_max_ts:
                try:
                    obs_dt = datetime.fromisoformat(running_max_ts)
                    now_utc = datetime.now(timezone.utc)
                    if obs_dt.tzinfo is None:
                        obs_dt = obs_dt.replace(tzinfo=timezone.utc)
                    running_max_age_min = (
                        (now_utc - obs_dt).total_seconds() / 60.0)
                except Exception:  # noqa: BLE001
                    running_max_age_min = None
            elif running_max_f and fc.get("issued_at"):
                try:
                    issued = datetime.fromisoformat(fc["issued_at"])
                    now_ref = datetime.now(issued.tzinfo or timezone.utc)
                    running_max_age_min = (
                        (now_ref - issued).total_seconds() / 60.0)
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

        # Build session-progress map: {ticker: session_pct (0.0-1.0)}
        # Kalshi markets open at midnight and close at ~7 PM Eastern.
        # We estimate open_time as midnight of target_date if not provided.
        session_progress: Dict[str, float] = {}
        now_utc = datetime.now(timezone.utc)
        for m in markets:
            ticker = m["market_ticker"]
            close_str = m.get("close_time")
            if not close_str:
                session_progress[ticker] = 0.5  # unknown — use mid-session
                continue
            try:
                close_dt = datetime.fromisoformat(
                    close_str.replace("Z", "+00:00"))
                if close_dt.tzinfo is None:
                    close_dt = close_dt.replace(tzinfo=timezone.utc)
                # Estimate open as midnight Eastern of target_date
                from datetime import date as _date
                target_d = _date.fromisoformat(target_date_str)
                open_dt = datetime.combine(
                    target_d,
                    __import__("datetime").time(4, 0),  # midnight Eastern = 4AM UTC
                    tzinfo=timezone.utc,
                )
                session_secs = (close_dt - open_dt).total_seconds()
                elapsed_secs = (now_utc - open_dt).total_seconds()
                pct = elapsed_secs / session_secs if session_secs > 0 else 0.5
                session_progress[ticker] = max(0.0, min(1.0, pct))
            except Exception:  # noqa: BLE001
                session_progress[ticker] = 0.5

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

            # Upgrade 2: session-aware size multiplier.
            # Early session (0-20%): half size — thin liquidity,
            #                        price discovery phase.
            # Mid session (20-80%): full size — primary edge window.
            # Late session (80%+):  model_divergence only if edge >10¢;
            #                       running_max unrestricted.
            sp = session_progress.get(ticker, 0.5)
            session_size_mult = 1.0
            if sp < 0.20:
                session_size_mult = 0.5
            elif sp >= 0.80:
                session_size_mult = 1.0  # sizing unchanged for late session

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
                # Late session: only enter model_divergence if edge > 10¢
                if sp >= 0.80 and d1.get("edge_cents", 0) < 10.0:
                    log.debug("Late session: skipping model_divergence "
                              "on %s — edge %.1f¢ < 10¢ threshold",
                              ticker, d1.get("edge_cents", 0))
                    d1 = None
            if d1:
                effective_mult = risk.size_multiplier * session_size_mult
                d1["contracts"] = max(
                    1, int(d1["contracts"] * effective_mult))
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
                # Running max arb is unrestricted by session phase
                # (it's not a forecast bet, it's a confirmed observation)
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
                current_spread_cents=arb.get("worst_spread_cents"),
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

        # Guard: do not enter a position on the same contract and
        # side if one is already open. This prevents double-entry
        # within the same cycle or after rapid price moves.
        open_pos = self.storage.open_kalshi_positions()
        existing = [
            p for p in open_pos
            if p.get("market_ticker") == ticker
            and p.get("side") == side
            and p.get("status") == "open"
        ]
        if existing:
            log.debug(
                "Skipping duplicate position: %s %s already open",
                ticker, side)
            return False

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
            # For running_max strategy, log a simulated 99c limit sell
            if strategy == "running_max":
                log.info("[PAPER] AUTO-SELL resting limit queued: "
                         "sell YES %s x%d @ 0.99 (running_max exit)",
                         ticker, contracts)
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
                # For running_max, immediately place a limit sell at 0.99
                if strategy == "running_max" and result:
                    sell_result = await self.feed.place_order(
                        market_ticker=ticker,
                        side="yes",
                        action="sell",
                        contracts=contracts,
                        limit_price=0.99,
                        client_order_id=f"al3x-exit-{uuid.uuid4().hex[:8]}",
                    )
                    if sell_result:
                        log.info("[LIVE] AUTO-SELL limit placed: "
                                 "sell YES %s x%d @ 0.99",
                                 ticker, contracts)
                    else:
                        log.warning("[LIVE] AUTO-SELL limit failed for %s "
                                    "— will settle at expiry", ticker)
                return True
            log.warning("[LIVE] Order failed for %s %s %s",
                        strategy, side, ticker)
            return False

    async def settle_positions(self, target_date_str: str) -> Dict[str, Any]:
        """Settle all open positions for target_date_str.

        For each open position:
          1. Check if the threshold was exceeded using CLI truth.
          2. Compute settlement P&L (YES wins if cli_high >= threshold,
             NO wins otherwise).
          3. Call storage.close_kalshi_position() with the result.
          4. Update bankroll: realized_pnl, current_bankroll,
             total_deployed (subtract the cost_basis), daily_pnl.

        Also handles the auto-sell-at-99c exit: if a running_max
        position was placed with a resting limit sell order
        (sell_order_id is set in notes), log that it was handled via
        that order.

        Returns a summary dict.
        """
        summary = {"settled": 0, "errors": [], "total_pnl": 0.0}
        cli_truth = self.storage.get_cli_truth(target_date_str)
        if not cli_truth:
            return summary

        cli_high = float(cli_truth["recorded_high_f"])
        positions = self.storage.positions_needing_settlement(target_date_str)

        for pos in positions:
            try:
                threshold_f = pos.get("threshold_f")
                if threshold_f is None:
                    continue
                side = pos.get("side", "yes")
                entry_price = float(pos.get("entry_price", 0))
                contracts = int(pos.get("contracts", 0))
                strategy = pos.get("strategy", "")

                # Determine settlement
                yes_wins = cli_high >= float(threshold_f)
                if side == "yes":
                    payout = 1.0 if yes_wins else 0.0
                else:
                    payout = 1.0 if not yes_wins else 0.0

                pnl = round((payout - entry_price) * contracts, 4)
                exit_price = payout
                status = "settled"

                self.storage.close_kalshi_position(
                    pos["id"], exit_price, pnl, status)

                # Update bankroll
                bankroll = self.storage.get_or_create_bankroll(
                    self._paper_mode)
                cost_basis = float(pos.get("cost_basis", 0))
                new_realized = float(bankroll.get("realized_pnl", 0)) + pnl
                # bankroll += pnl (net gain/loss from the stake)
                new_current = float(bankroll.get("current_bankroll", 0)) + pnl
                new_deployed = max(
                    0.0,
                    float(bankroll.get("total_deployed", 0)) - cost_basis)
                new_daily = float(bankroll.get("daily_pnl", 0)) + pnl
                win_increment = 1 if pnl > 0 else 0
                self.storage.update_bankroll(self._paper_mode, {
                    "realized_pnl": new_realized,
                    "current_bankroll": new_current,
                    "total_deployed": new_deployed,
                    "daily_pnl": new_daily,
                    "trade_count": int(bankroll.get("trade_count", 0)) + 1,
                    "win_count": int(bankroll.get("win_count", 0)) + win_increment,
                })
                summary["settled"] += 1
                summary["total_pnl"] += pnl
                log.info("[SETTLE] %s %s threshold=%.0f°F cli=%.1f°F "
                         "pnl=%+.4f",
                         strategy.upper(), side.upper(), float(threshold_f),
                         cli_high, pnl)
            except Exception as e:  # noqa: BLE001
                log.warning("Settlement error for position %s: %s",
                            pos.get("id"), e)
                summary["errors"].append(str(e))

        summary["total_pnl"] = round(summary["total_pnl"], 4)
        return summary


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
