---
description: Full autonomous tuning system. 10 parallel optimization layers + forensic loss analysis + causal counterfactual on every promotion + A/B shadow tournament for live distribution-shift testing + Pareto-frontier multi-objective optimization. Supersedes /goals optimize-evergreen.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# /goals autotune

## Mission

Build and run a full autonomous tuning system that reviews every paper trade, every source value, every CLI truth, every regime tag, and every microstructure snapshot — across 10 parallel optimization layers — and continuously promotes a Pareto-dominant production config.

This goal **supersedes** `/goals optimize-evergreen`. If optimize-evergreen is already running, kill it and migrate its study database to autotune's schema (instructions below). Requires `/goals shadow-flywheel` to be active.

## The 10 optimization layers

Each layer runs as an independent Optuna study, all writing to the same `al3x.db`. They share the paper_trades dataset but tune disjoint parameter subsets to prevent interference.

### Layer 1 — Strategy-level
Per-strategy parameter sets for each of the 6 wagering strategies:
- `ensemble`: edge_min, sigma_min, member_weights
- `sigma`: sigma_threshold, max_size_multiplier
- `locks`: observed_confidence, physics_safety_margin, trajectory_confluence_req
- `divergence`: min_spread_F, source_pair_weights
- `regime`: regime_confidence_min
- `calibrated`: min_samples_per_bin, regularization

Optuna study: `autotune_strategy`. Each strategy's params are isolated — strategy A's optimization doesn't move strategy B's knobs.

### Layer 2 — Per-regime
For each regime in `{heat_dome, marine_layer, cold_front, calm_clear, snow_event, thunderstorm, post_frontal, normal}`:
- Kelly fraction multiplier (regime can dial up/down vs. base)
- Edge minimum multiplier
- Sigma threshold multiplier
- Strategy enable mask (which of the 6 strategies fire in this regime)

Optuna study: `autotune_regime`. Optimizes 8 regimes × ~6 params = 48-dim per-regime space.

### Layer 3 — Source weights & EMOS
- Softmax weights over all sources (production + promoted shadows)
- EMOS coefficients (a, b, c, d) per source for mean and variance correction
- Source independence collapse policy (5 options)

Optuna study: `autotune_sources`. The largest dimensional space; most compute lives here.

### Layer 4 — Microstructure
Order-book features at decision time, tuned for entry filtering:
- Taker imbalance threshold (skip trade if YES taker buys > X% in last 1h)
- Spread tightness minimum (skip if spread > X cents)
- Depth at top-of-book minimum (skip if < N contracts)
- Candlestick momentum gate (skip if YES rose > X¢ in last 3h on volume)

Optuna study: `autotune_micro`.

### Layer 5 — Calibration
Multi-dimensional isotonic regression mapping raw P(YES) → calibrated P(YES) sliced by:
- (strategy × regime × lead_time_bucket × sigma_band)

Tunes: min samples per bin, regularization strength, blend weight between local and global calibrators.

Optuna study: `autotune_calib`.

### Layer 6 — Risk
- Kelly fraction base (0.10–0.40)
- Per-contract position cap (% of bankroll)
- Total gross exposure cap (% of bankroll)
- Per-day loss limit (% of bankroll → halt trading)
- Drawdown circuit breaker (% peak-to-trough → halt)

Optuna study: `autotune_risk`.

### Layer 7 — Time-of-day
Separate parameter sets for distinct trading windows:
- Pre-noon (ensemble-driven): different edge_min, larger sigma_min
- Noon decision (the main entry): tightest constraints
- Afternoon lock (2pm–6pm): observed/physics lock thresholds
- Evening settlement (6pm–close): final-hour params, microstructure-heavy

Optuna study: `autotune_tod`.

### Layer 8 — Pairwise source bias
Learn joint error distributions: when ECMWF and GFS *both* say warm, what's the actual bias direction and magnitude? Same for every source pair (28 pairs from 8 sources).

Stores a `pairwise_bias` table:
```sql
CREATE TABLE IF NOT EXISTS pairwise_bias (
  source_a TEXT, source_b TEXT,
  agreement_band TEXT,           -- 'both_warm','both_cold','disagree_large','close'
  observed_bias_F REAL,          -- mean signed error of (mean(a,b) - truth)
  n_samples INTEGER,
  updated_at_utc TEXT,
  PRIMARY KEY (source_a, source_b, agreement_band)
);
```

Recomputed nightly. Forecaster reads this and applies a correction to weighted source mean.

### Layer 9 — Meta-learner
For every promoted config (going back to start of history), record:
- Param diff vs prior
- Subsequent 7-day, 30-day, 60-day Sharpe lift (or loss)

Train a lightweight gradient-boost model (lightgbm) on (param_diff_vector) → (90-day Sharpe lift). Use feature importances to **bias the next Optuna search** toward parameter directions that historically produced wins.

Optuna study: `autotune_meta`. Updates trial-suggestion priors weekly.

### Layer 10 — Strategy gate optimizer
For each of the 6 strategies:
- Compute rolling 200-trade Sharpe + 95% bootstrap CI
- If upper bound < 0 → auto-disable (write to `configs/strategy_gates.json`)
- If gated strategy reaches > 50 paper trades again (manual re-enable or shadow promotion) and rolling Sharpe lower bound > 0 → auto-re-enable

This runs hourly. Independent of Optuna — pure rule-based.

## The meta-loop (the 100× layer)

### Forensic loss analysis
For every paper trade that closes with a loss > $1 (configurable), run `tools/loss_forensics.py`:

```
For loss row:
  1. Reconstruct decision-time state from source_snapshot.
  2. For each of the 10 layers, sweep ±20% around each param dimension.
  3. Find the smallest param delta that would have prevented this trade.
  4. Write to losses_forensic table:
     (trade_id, preventing_param, preventing_delta, est_pnl_save, category)
  5. Categories: source_overconfident, regime_mistag, microstructure_missed,
     calibration_drift, lock_threshold_too_loose, edge_min_too_low, etc.
```

Loss forensics output **biases Optuna's next round** — params that frequently appear in forensic reports get sampled more aggressively. This is closed-loop self-improvement.

### Causal counterfactual on every promotion
When a config gets promoted, schedule a 7-day checkpoint:

```
At promotion + 7 days:
  1. Pull all paper trades opened in the last 7 days.
  2. Replay them under prior_config and current_config.
  3. Compute Sharpe, PnL, drawdown under each.
  4. If current_config underperforms prior by > 10% Sharpe AND statistical
     significance > 90% → auto-revert to prior_config.
  5. Write reversion to configs/reversions.jsonl with full diff.
  6. Telegram: "Reverted config_hash X — failed 7-day counterfactual."
```

Prevents lucky-trial promotions from poisoning production.

### A/B shadow tournament
Top 5 candidate configs from Optuna run **in parallel** on live paper trades:

- Each tick, every candidate computes its own would-be paper trade.
- Trades are logged to `paper_trades_ab` with `config_hash` and a tournament_id.
- After 14 days, the candidate that Pareto-dominates the most on (Sharpe, max DD, win rate, CVaR) on LIVE data — not historical replay — becomes the promotion target.

This catches distribution shift: a config that won on historical data may lose on current market regime. Live A/B is the only protection.

### Anomaly detection
Hourly check:
- Is the optimizer's best-objective trajectory suddenly flat or declining?
- Are paper trades suddenly losing across all configs?
- Has source MAE jumped > 30% for any single source vs 30-day baseline?

If yes → Telegram alert, write to `logs/anomaly.jsonl`, pause promotions for 24h.

## Multi-objective Pareto

Objective vector (compute for every candidate):
```
[
  median_walk_forward_sharpe,
  -max_drawdown_pct,        # higher is better
  win_rate_pct,
  -cvar_5pct,                # tail risk; higher is better
  median_paper_dollars_per_day
]
```

A candidate **dominates** production iff it's ≥ on all 5 axes AND strictly > on ≥ 2.

Promotion criteria:
- Pareto-dominates production
- Survives bootstrap robustness (5th percentile Sharpe > 0.3)
- Survives anti-overfit gates below
- Passes the A/B shadow tournament if one is running

## Anti-overfit hard rules

1. **Walk-forward only.** No in-sample evaluation. Period.
2. **Sacred hold-out.** Last 14 days excluded from all training. Every promotion must beat production on the hold-out too.
3. **Minimum trades per fold:** 100. Smaller folds skipped.
4. **Parameter shrinkage prior.** L2 toward production config. Large jumps cost objective points.
5. **Stop if data stale.** If no new paper_trades in 48h, halt and alert.
6. **Diversity bonus.** Discourage clusters of near-identical candidates in promotion queue.
7. **No more than 1 promotion per 24 hours.** Even if 3 candidates qualify, promote 1, observe, then iterate.

## Schema additions to al3x.db

```sql
CREATE TABLE IF NOT EXISTS losses_forensic (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trade_id INTEGER NOT NULL,
  preventing_param TEXT NOT NULL,
  preventing_delta REAL NOT NULL,
  est_pnl_save REAL NOT NULL,
  category TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  FOREIGN KEY (trade_id) REFERENCES paper_trades(id)
);

CREATE TABLE IF NOT EXISTS paper_trades_ab (
  -- same schema as paper_trades plus:
  tournament_id TEXT NOT NULL,
  candidate_config_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS promotion_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  promoted_at_utc TEXT NOT NULL,
  prior_hash TEXT NOT NULL,
  new_hash TEXT NOT NULL,
  expected_sharpe_lift REAL,
  observed_sharpe_lift_7d REAL,
  observed_sharpe_lift_30d REAL,
  reverted INTEGER DEFAULT 0,
  revert_reason TEXT
);

CREATE TABLE IF NOT EXISTS pairwise_bias (
  source_a TEXT, source_b TEXT,
  agreement_band TEXT,
  observed_bias_F REAL,
  n_samples INTEGER,
  updated_at_utc TEXT,
  PRIMARY KEY (source_a, source_b, agreement_band)
);

CREATE TABLE IF NOT EXISTS strategy_gates (
  strategy TEXT PRIMARY KEY,
  enabled INTEGER NOT NULL,
  reason TEXT,
  rolling_sharpe REAL,
  rolling_sharpe_ci_lo REAL,
  rolling_sharpe_ci_hi REAL,
  updated_at_utc TEXT
);
```

## Outputs

- `configs/production.json` — what the live system reads
- `configs/candidate_*.json` — every Optuna candidate that passed bootstrap
- `configs/strategy_gates.json` — which of the 6 strategies are enabled
- `configs/reversions.jsonl` — every auto-revert with full diff
- `configs/promotion_queue.json` — pending Pareto-dominant candidates
- `{OBSIDIAN_VAULT_PATH}/AL3X/Autotune/{YYYY-MM-DD}.md` — daily report:
  - All 10 layer states + best objective
  - Forensic loss categories (top 5 today)
  - A/B tournament standings
  - Pareto-dominant candidate diff vs production
  - Hold-out performance: production vs top candidate
  - Anomaly flags if any
  - The single most important line: **"AL3X.NYC is +X% sharper than 30 days ago"** computed against Pareto-weighted objective vector.

## Acceptance criteria

- All 10 Optuna studies exist in al3x.db with at least one completed trial each
- `losses_forensic`, `paper_trades_ab`, `promotion_history`, `pairwise_bias`, `strategy_gates` tables created
- At least one forensic loss postmortem completed end-to-end
- A/B tournament wired up with at least 2 candidates running
- Pareto dominance test verified with unit tests on synthetic data
- Auto-revert mechanism verified with a forced-bad-promotion test then verified rollback
- Anomaly detector verified with a forced spike in source MAE
- Strategy gate auto-disable verified with a forced losing strategy
- Daily Obsidian report produced
- Telegram fires "Autotune online — 10 layers active. Last 30d Sharpe lift: X%."

## Migration from /goals optimize-evergreen

If the old optimizer is running:

```bash
# 1. Stop the old daemon
bash tools/compound_kill.sh   # if running under compound
# OR if running standalone:
pkill -f "optimize_evergreen"

# 2. Backup the old study
sqlite3 al3x.db ".dump --table=optuna_studies --table=optuna_trials" > backup_evergreen_$(date +%Y%m%d).sql

# 3. Migrate trial data to the new schema
python -m al3x.learning.migrate_evergreen_to_autotune

# 4. Restart under autotune
nohup python -m daemon.autotune > logs/autotune.log 2>&1 &
echo $! > .pids/autotune.pid
```

Then loop forever.
