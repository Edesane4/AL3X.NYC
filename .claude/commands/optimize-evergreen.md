---
description: Continuous Bayesian hyperparameter optimization over AL3X.NYC's entire knob set, walk-forward validated against the paper_trades dataset. Writes a locked config nightly.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# /goals optimize-evergreen

## Mission

Run an **infinite Bayesian hyperparameter optimization loop** over every tunable parameter in AL3X.NYC. Eval each candidate config on the `paper_trades` dataset using walk-forward validation. Promote winners. Lock losers out. Compound nightly.

This goal **requires** `/goals shadow-flywheel` to be running. It eats that data.

## The search space (every knob)

Build a single `al3x/learning/search_space.py` that exposes:

| Parameter group | Knobs | Range |
|---|---|---|
| Source weights | `w_hrrr`, `w_gfs`, `w_gfs_mos`, `w_ecmwf`, `w_nbm`, `w_nws_point`, `w_gefs`, `w_open_meteo` | softmax over [0, 5] each |
| EMOS coefficients per source | `a`, `b`, `c`, `d` (mean and variance correction) | per-source priors centered on identity |
| Lead-time buffer curve | buffer_at_hour for h in {0,2,4,6,8,12,18,24,36,48} | 0.0 — 1.0 monotonic |
| Sigma thresholds | `sigma_min_settlement`, `sigma_min_elite`, `sigma_min_physics` | 1.0 — 5.0 |
| Kelly fraction | `kelly_frac` | 0.10 — 0.40 |
| Edge minimum | `min_edge_cents` | 2 — 8 |
| Regime gates | one boolean per regime × metric × side | enabled / disabled |
| Source independence rules | NWS/MOS/NBM/NWS_POINT collapse policy | 1 of 5 policies |
| Time-of-day weighting | obs anchor weight curve | piecewise linear |
| Isotonic calibration on/off per metric | bool × 4 | T/F |

That's a 60+ dimensional space. Optuna's TPE sampler handles it.

## Loop

```
while True:
    1. Load all paper_trades from last 180 days, grouped by config_hash.
    2. Run an Optuna study (storage=sqlite:///al3x.db, study='evergreen').
    3. For each trial:
         a. Sample params from search space
         b. Recompute internal_p_yes for every paper trade using sampled params
            applied to its source_snapshot JSON (no re-fetching — pure replay)
         c. Recompute edge, position, pnl_dollars
         d. Walk-forward folds: train [t0, t0+30d], test [t0+30d, t0+45d]; slide
            by 7 days; require >= 6 folds
         e. Objective = median fold Sharpe ratio − 0.5 * worst fold Sharpe
            (penalize fragility) − 0.1 * max drawdown in dollars
         f. Report trial to study
    4. Every 50 trials:
         - Snapshot best-so-far params
         - Run robustness test: Monte Carlo bootstrap 500 resamples of the
           paper_trades set, recompute objective. Require 5th percentile Sharpe > 0.5.
         - If pass: write to `configs/candidate_{YYYYMMDD_HHMM}.json`
    5. Once per day at 02:00 ET:
         - Diff candidate vs production config
         - If candidate beats production by >= 20% on median fold Sharpe AND
           passes robustness test, promote to `configs/production.json`
         - Compute new config_hash, write to telegram + Obsidian
         - The shadow-flywheel will pick up the new config on its next cycle
    6. Continue.
```

## Replay engine — critical

In `al3x/learning/replay.py`, build a function:

```python
def replay_paper_trade(row: dict, params: dict) -> dict:
    """
    Given a paper_trades row (with source_snapshot JSON) and a candidate
    parameter dict, recompute what AL3X would have done with those params.
    Returns updated {edge_cents, kelly_fraction, size_contracts, pnl_dollars,
    correct}.
    """
```

This must be a **pure function** of the snapshot — no network calls, no DB reads beyond the row. It is called millions of times during optimization; performance matters. Vectorize with numpy where possible.

## Anti-overfitting requirements

- **Walk-forward only.** Never train and test on the same window.
- **Hold-out set.** Reserve the most recent 14 days as a sacred hold-out. Every promoted config must beat production on that hold-out too.
- **Minimum trades per fold:** 100. Skip folds with less and reduce slide rate.
- **Parameter shrinkage prior.** Apply L2 regularization toward current production config; large jumps cost objective points. Prevents lucky-trial drift.
- **Stop if data stale.** If no new paper_trades in 48h, halt optimization and alert. The flywheel is down.

## Outputs

- `configs/production.json` — what the live system reads
- `configs/candidate_*.json` — every snapshot
- `configs/promotion_log.jsonl` — every promotion with diff, Sharpe lift, hold-out result
- `{OBSIDIAN_VAULT_PATH}/AL3X/Optimizer/{YYYY-MM-DD}.md` — daily report including:
  - Trials completed today
  - Best objective so far
  - Top 5 parameter contributions (Optuna importances)
  - Promoted config diff vs yesterday
  - Hold-out performance: current vs new

## Acceptance criteria

- `study evergreen` exists in al3x.db with > 200 completed trials
- `replay_paper_trade` unit tested with at least 5 known-outcome cases
- One full promotion cycle has run on real paper_trades data (even if no promotion fires)
- Obsidian report produced
- Telegram fires "Optimizer online — best Sharpe {x}"
- Shrinkage prior verified active (delta vs production reported on every promoted candidate)

Then loop forever.
