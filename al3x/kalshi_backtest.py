"""Historical backtest for the Kalshi market intelligence engine.

Pulls historical Kalshi market data for NYC temperature contracts,
cross-references against stored AL3X forecasts and CLI truth,
and simulates every trade decision to compute theoretical P&L.

Run via the API endpoint /api/kalshi/backtest or triggered manually.
Results are stored in the kalshi_backtest_results table for audit.

The backtest answers: "If AL3X had been running against these markets
for the past N days, what would the edge and P&L have been?"
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from . import config as cfg
from .fair_value import compute_fair_value, compute_edge, estimate_sigma
from .kelly_sizer import kelly_fraction, size_position
from .kalshi_feed import KalshiFeed, _safe_price, _parse_threshold
from .storage import Storage

log = logging.getLogger("al3x.backtest")

_BACKTEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS kalshi_backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    target_date TEXT NOT NULL,
    market_ticker TEXT NOT NULL,
    threshold_f REAL,
    strategy TEXT NOT NULL,
    simulated_side TEXT,
    simulated_entry_price REAL,
    simulated_contracts INTEGER,
    simulated_cost_basis REAL,
    al3x_fair_value REAL,
    al3x_forecast_f REAL,
    al3x_sigma REAL,
    market_open_price REAL,
    market_close_price REAL,
    settlement_result TEXT,           -- 'yes_wins', 'no_wins', 'unknown'
    cli_high_f REAL,
    simulated_pnl REAL,
    edge_cents REAL,
    ev REAL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_bt_date
    ON kalshi_backtest_results(target_date);
"""


def _ensure_schema(storage: Storage) -> None:
    with storage._conn() as c:  # noqa: SLF001
        c.executescript(_BACKTEST_SCHEMA)


class KalshiBacktest:
    """Runs the historical backtest over a date range.

    Requires:
      1. AL3X forecasts in the DB for those dates (from forecaster)
      2. CLI truth in the DB for those dates (from learning system)
      3. Kalshi historical market data (fetched live via the API)
    """

    def __init__(self, storage: Storage, feed: KalshiFeed) -> None:
        self.storage = storage
        self.feed = feed
        _ensure_schema(storage)

    async def run(self, days_back: int = 30,
                   starting_bankroll: float = 500.0
                   ) -> Dict[str, Any]:
        """Run the backtest over the past `days_back` days.

        Returns a summary with P&L, win rate, EV, and per-strategy
        breakdowns. All results are saved to kalshi_backtest_results.
        """
        run_at = datetime.now(timezone.utc).isoformat()
        today = date.today()
        results: List[Dict[str, Any]] = []
        errors: List[str] = []

        log.info("Backtest starting: %d days back from %s", days_back, today)

        # Load all forecast rows once and index by target_date.
        # This avoids O(N²) DB queries (one full join per day).
        all_fc_rows = self.storage.forecasts_with_truth(
            days=days_back + 5, prefer_mode="night_before")
        fc_by_date: Dict[str, Any] = {
            r["target_date"]: r for r in all_fc_rows
            if r.get("target_date")
        }

        for days_ago in range(1, days_back + 1):
            target = today - timedelta(days=days_ago)
            target_str = target.isoformat()

            # Skip if no CLI truth (no ground truth to evaluate against)
            cli_truth = self.storage.get_cli_truth(target_str)
            if not cli_truth:
                log.debug("Backtest: no CLI truth for %s — skipping",
                          target_str)
                continue

            cli_high = float(cli_truth["recorded_high_f"])

            fc = fc_by_date.get(target_str)
            if not fc:
                log.debug("Backtest: no AL3X forecast for %s", target_str)
                continue

            al3x_f = float(fc.get("final_f") or 0)
            extras = fc.get("extras") or {}
            bma = extras.get("bma")
            qrf = extras.get("qrf")
            sigma = estimate_sigma(bma, qrf)

            # Fetch historical Kalshi markets for this date
            markets = await self.feed.get_nyc_temp_markets(target_str)
            if not markets:
                log.debug("Backtest: no Kalshi markets for %s", target_str)
                errors.append(f"no_markets:{target_str}")
                continue

            day_results = await self._backtest_day(
                target_str=target_str,
                markets=markets,
                al3x_f=al3x_f,
                sigma=sigma,
                cli_high=cli_high,
                starting_bankroll=starting_bankroll,
                run_at=run_at,
            )
            results.extend(day_results)

        # Save all results to DB
        for r in results:
            self._save_result(r)

        return self._summarize(results, starting_bankroll, errors)

    async def _backtest_day(self,
                              target_str: str,
                              markets: List[Dict[str, Any]],
                              al3x_f: float,
                              sigma: float,
                              cli_high: float,
                              starting_bankroll: float,
                              run_at: str
                              ) -> List[Dict[str, Any]]:
        """Simulate all trade decisions for one historical day."""
        day_results = []
        simulated_bankroll = starting_bankroll
        simulated_deployed = 0.0

        for m in markets:
            ticker = m["market_ticker"]
            threshold = m.get("threshold_f")
            if threshold is None:
                continue

            # Use the market's last_price as a proxy for the open price
            # (Kalshi historical API gives us last-known price)
            market_price = m.get("last_price") or m.get("yes_bid")
            if market_price is None:
                continue
            market_price = float(market_price)

            # Compute fair value
            fv_result = compute_fair_value(
                threshold_f=float(threshold),
                al3x_forecast_f=al3x_f,
                sigma=sigma,
            )
            fv = float(fv_result["fair_value"])

            # Determine if YES settles
            settlement = ("yes_wins" if cli_high >= float(threshold)
                           else "no_wins")

            # Evaluate each strategy
            for strategy, entry_price, side in _strategy_candidates(
                    fv, market_price, threshold, cli_high):

                edge_dict = compute_edge(fv, market_price, side)
                if not edge_dict["actionable"]:
                    continue

                # Size the position
                sizing = size_position(
                    fair_value=fv,
                    entry_price=entry_price,
                    side=side,
                    current_bankroll=simulated_bankroll,
                    current_total_exposure=simulated_deployed,
                    current_threshold_exposure=0.0,
                    strategy=strategy,
                )
                if sizing["contracts"] == 0:
                    continue

                contracts = sizing["contracts"]
                cost = sizing["cost_basis"]

                # Compute settlement P&L
                pnl = _compute_settlement_pnl(
                    side, entry_price, contracts, settlement)

                simulated_deployed += cost
                # pnl from _compute_settlement_pnl is already net
                # (payout - entry) × contracts. Add directly.
                simulated_bankroll += pnl

                day_results.append({
                    "run_at": run_at,
                    "target_date": target_str,
                    "market_ticker": ticker,
                    "threshold_f": float(threshold),
                    "strategy": strategy,
                    "simulated_side": side,
                    "simulated_entry_price": entry_price,
                    "simulated_contracts": contracts,
                    "simulated_cost_basis": round(cost, 4),
                    "al3x_fair_value": fv,
                    "al3x_forecast_f": al3x_f,
                    "al3x_sigma": sigma,
                    "market_open_price": market_price,
                    "market_close_price": market_price,  # best available
                    "settlement_result": settlement,
                    "cli_high_f": cli_high,
                    "simulated_pnl": round(pnl, 4),
                    "edge_cents": edge_dict["edge_cents"],
                    "ev": edge_dict["ev"],
                    "notes": "",
                })

        return day_results

    def _save_result(self, r: Dict[str, Any]) -> None:
        try:
            with self.storage._conn() as c:  # noqa: SLF001
                c.execute(
                    """INSERT INTO kalshi_backtest_results
                       (run_at, target_date, market_ticker, threshold_f,
                        strategy, simulated_side, simulated_entry_price,
                        simulated_contracts, simulated_cost_basis,
                        al3x_fair_value, al3x_forecast_f, al3x_sigma,
                        market_open_price, market_close_price,
                        settlement_result, cli_high_f, simulated_pnl,
                        edge_cents, ev, notes)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        r["run_at"], r["target_date"], r["market_ticker"],
                        r.get("threshold_f"), r["strategy"],
                        r["simulated_side"], r["simulated_entry_price"],
                        r["simulated_contracts"], r["simulated_cost_basis"],
                        r["al3x_fair_value"], r["al3x_forecast_f"],
                        r["al3x_sigma"], r["market_open_price"],
                        r["market_close_price"], r["settlement_result"],
                        r["cli_high_f"], r["simulated_pnl"],
                        r["edge_cents"], r["ev"], r.get("notes", ""),
                    ),
                )
        except Exception as e:  # noqa: BLE001
            log.debug("Backtest result save failed: %s", e)

    def _summarize(self, results: List[Dict[str, Any]],
                    starting_bankroll: float,
                    errors: List[str]) -> Dict[str, Any]:
        if not results:
            return {"ok": False, "reason": "no_results",
                    "errors": errors}

        total_pnl = sum(r["simulated_pnl"] for r in results)
        total_cost = sum(r["simulated_cost_basis"] for r in results)
        winning = [r for r in results if r["simulated_pnl"] > 0]
        win_rate = len(winning) / len(results) if results else 0.0
        mean_ev = (sum(r["ev"] for r in results) / len(results)
                   if results else 0.0)
        mean_edge = (sum(r["edge_cents"] for r in results) / len(results)
                      if results else 0.0)
        roi = total_pnl / total_cost if total_cost > 0 else 0.0

        # Per-strategy breakdown
        by_strategy: Dict[str, Dict[str, Any]] = {}
        for r in results:
            s = r["strategy"]
            if s not in by_strategy:
                by_strategy[s] = {"trades": 0, "wins": 0,
                                   "pnl": 0.0, "cost": 0.0}
            by_strategy[s]["trades"] += 1
            by_strategy[s]["pnl"] += r["simulated_pnl"]
            by_strategy[s]["cost"] += r["simulated_cost_basis"]
            if r["simulated_pnl"] > 0:
                by_strategy[s]["wins"] += 1
        for s in by_strategy:
            n = by_strategy[s]["trades"]
            by_strategy[s]["win_rate"] = round(
                by_strategy[s]["wins"] / n, 3) if n else 0.0
            by_strategy[s]["roi"] = round(
                by_strategy[s]["pnl"] / by_strategy[s]["cost"], 3
            ) if by_strategy[s]["cost"] > 0 else 0.0

        summary = {
            "ok": True,
            "total_trades": len(results),
            "winning_trades": len(winning),
            "win_rate": round(win_rate, 3),
            "total_pnl": round(total_pnl, 2),
            "total_cost_basis": round(total_cost, 2),
            "roi": round(roi, 3),
            "mean_ev": round(mean_ev, 4),
            "mean_edge_cents": round(mean_edge, 2),
            "starting_bankroll": starting_bankroll,
            "ending_bankroll": round(starting_bankroll + total_pnl, 2),
            "by_strategy": by_strategy,
            "errors": errors,
            "positive_ev": mean_ev >= cfg.KALSHI_MIN_EV,
            "recommendation": (
                "GO LIVE — positive EV confirmed across backtest"
                if (win_rate >= 0.55 and mean_ev >= cfg.KALSHI_MIN_EV
                    and total_pnl > 0)
                else "DO NOT GO LIVE — insufficient edge in backtest"
            ),
        }
        log.info("Backtest complete: %d trades, %.1f%% win rate, "
                 "total P&L $%.2f, ROI %.1f%%",
                 len(results), win_rate * 100, total_pnl, roi * 100)
        return summary


# ---- Helpers ------------------------------------------------------------

def _strategy_candidates(
        fv: float, market_price: float,
        threshold: float, cli_high: float,
) -> List[Tuple[str, float, str]]:
    """Return list of (strategy, entry_price, side) candidates."""
    candidates = []
    min_edge = cfg.KALSHI_MIN_EDGE_CENTS / 100.0

    # Model Divergence: YES side
    if fv - market_price >= min_edge:
        candidates.append(("model_divergence", market_price, "yes"))

    # Model Divergence: NO side
    if market_price - fv >= min_edge:
        no_price = 1.0 - market_price
        candidates.append(("model_divergence", no_price, "no"))

    # Running Max Arb: only if cli_high confirms the threshold was met
    # and the YES price was below certainty threshold
    if (cli_high >= threshold
            and market_price < cfg.KALSHI_RUNNING_MAX_CERTAINTY_THRESHOLD):
        candidates.append(("running_max", market_price, "yes"))

    return candidates


def _compute_settlement_pnl(side: str, entry_price: float,
                              contracts: int, settlement: str) -> float:
    """Compute P&L at settlement for a binary Kalshi contract.

    YES contract: pays $1 per contract if high > threshold (yes_wins),
                  pays $0 if no_wins (lose entry_price per contract).
    NO  contract: pays $1 per contract if no_wins,
                  pays $0 if yes_wins.

    P&L = (payout - entry_price) * contracts
    """
    if side == "yes":
        payout = 1.0 if settlement == "yes_wins" else 0.0
    else:
        payout = 1.0 if settlement == "no_wins" else 0.0
    return round((payout - entry_price) * contracts, 4)
