---
description: The compound machine. Runs shadow-flywheel + autotune (10-layer optimizer) + alpha-hunt as one coordinated daemon. Each output becomes the next input. Start once, walk away, AL3X.NYC sharpens itself forever.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, WebSearch, WebFetch
---

# /goals compound

## Mission

Unify `/goals shadow-flywheel`, `/goals autotune`, and `/goals alpha-hunt` into a single supervised daemon where each loop's output is the next loop's input. This is the highest-leverage thing you can run on AL3X.NYC.

(Note: `/goals autotune` supersedes the older `/goals optimize-evergreen`. If you're on the old version, the autotune slash command includes a migration block.)

## The compound architecture

```
   ┌─────────────────────────────────────────────────────────────┐
   │                  daemon/compound.py                          │
   │                                                              │
   │   ┌──────────────────┐                                       │
   │   │ shadow-flywheel  │  ─────► paper_trades (DB)             │
   │   │ 15-min ticks     │                                       │
   │   └──────────────────┘             │                         │
   │           ▲                        ▼                         │
   │           │              ┌───────────────────┐               │
   │           │              │ autotune          │               │
   │           │              │ 10 parallel       │               │
   │           │              │ Optuna studies +  │               │
   │           │              │ forensics + A/B   │               │
   │           │              └───────────────────┘               │
   │           │                        │                         │
   │           │              configs/production.json             │
   │           │              + strategy_gates.json               │
   │           │              + pairwise_bias                     │
   │           │                        │                         │
   │           └────────────────────────┘                         │
   │                                                              │
   │   ┌──────────────────┐                                       │
   │   │ alpha-hunt       │  ─────► shadow_sources / promo queue  │
   │   │ daily            │                                       │
   │   └──────────────────┘             │                         │
   │           ▲                        ▼                         │
   │           │              feeds autotune Layer 3              │
   │           │              search space dynamically            │
   └──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    Telegram + Obsidian
```

## Build order

1. Build `daemon/compound.py` as a single async process using `asyncio` + `apscheduler`:
   - 15-min job: `flywheel_tick()` — pulls Kalshi NYC markets, runs forecaster, writes paper_trades
   - Hourly job: `grading_tick()` — pulls NWS CLI, resolves open paper trades
   - Hourly job: `strategy_gate_tick()` — auto-disable/re-enable strategies based on rolling Sharpe CI
   - 2-min job: `autotune_tick()` — drives all 10 Optuna studies, one trial each round-robin
   - 5-min job: `ab_tournament_tick()` — runs top 5 candidate configs on live paper trades
   - 30-min job: `forensics_tick()` — postmortem every newly closed losing trade
   - Daily 02:00 ET: `promotion_tick()` — Pareto-dominance check, promote if criteria met
   - Daily 02:30 ET: `counterfactual_tick()` — 7-day check on prior promotions, auto-revert if needed
   - Daily 03:00 ET: `pairwise_bias_tick()` — recompute joint source-pair biases
   - Daily 04:00 ET: `alpha_hunt_tick()` — research + shadow source implementation
   - Weekly Sunday 03:00 ET: `shadow_eval_tick()` — evaluate shadow sources
   - Weekly Sunday 03:30 ET: `meta_learner_tick()` — retrain promotion-success model

2. Wire the cross-talk:
   - `paper_trades` insertions trigger autotune study update markers across all 10 layers
   - `configs/production.json` promotions trigger a flywheel config reload at next tick boundary
   - `configs/strategy_gates.json` changes apply on next 15-min flywheel tick
   - `shadow_sources.py` registrations picked up at next flywheel tick
   - `configs/promotion_queue.json` from alpha-hunt → auto-injected into autotune Layer 3 within 24h
   - `losses_forensic` insertions bias next autotune trial-sampling round
   - `configs/reversions.jsonl` writes trigger immediate flywheel config reload back to prior

3. Centralized state in `al3x.db`:
   ```sql
   CREATE TABLE IF NOT EXISTS compound_state (
     key TEXT PRIMARY KEY,
     value TEXT NOT NULL,
     updated_at_utc TEXT NOT NULL
   );
   -- holds 'production_config_hash', 'last_promotion_at', 'last_revert_at',
   --       'last_alpha_hunt_at', 'open_paper_trade_count',
   --       'ab_tournament_id', 'gated_strategies', 'anomaly_flag'
   ```

4. **Supervisor and recovery.** Wrap every job in retry-with-backoff. If a job fails 3× consecutively, fire Telegram and continue running other jobs. Never let one failure crash the daemon. Maintain `logs/compound_health.jsonl` with one line per job-tick.

5. **Health dashboard.** Build `tools/compound_status.py`:
   ```bash
   python tools/compound_status.py
   ```
   Prints to terminal:
   - Daemon uptime
   - Last successful flywheel tick + open paper trades
   - Last grading run + trades resolved today
   - Autotune: 10 layer states with trials count & best objective each
   - A/B tournament standings (Pareto position per candidate)
   - Strategy gate state (which of the 6 strategies are enabled/disabled, why)
   - Alpha-hunt: shadow sources running, last research run
   - Last promotion + last revert
   - Anomaly flags
   - Telegram heartbeat status
   - PnL today / 7-day / 30-day on paper bankroll
   - Any retry-with-backoff failures in last 24h

## The kill switch

Build `tools/compound_kill.sh`:
```bash
#!/usr/bin/env bash
PID=$(cat .pids/compound.pid 2>/dev/null)
if [ -n "$PID" ]; then
  kill -TERM "$PID"
  sleep 3
  kill -KILL "$PID" 2>/dev/null
fi
rm -f .pids/compound.pid
echo "compound daemon stopped"
```

And a graceful SIGTERM handler in `daemon/compound.py` that flushes paper_trades writes, closes all 10 Optuna studies, writes a "shutting down" Telegram, and exits cleanly.

## Daily Obsidian roll-up

At 23:00 ET, write `{OBSIDIAN_VAULT_PATH}/AL3X/Compound/{YYYY-MM-DD}.md` combining:
- **Flywheel:** trades opened/closed/PnL/calibration table
- **Autotune:** trials per layer, best objective per layer, Pareto-dominant candidate diff, any promotions, any reversions, top 5 forensic loss categories today
- **A/B tournament:** standings, current leader, time-to-decision
- **Strategy gates:** which strategies are gated and why
- **Alpha-hunt:** papers read, shadows running, weekly eval if Sunday
- **System health:** uptime, errors, retries, anomaly flags
- **The single most important line:** **"AL3X.NYC is +{x}% sharper than 30 days ago"** computed against the Pareto-weighted objective vector.

## Run command

```bash
# from repo root
mkdir -p .pids logs configs
nohup python -m daemon.compound > logs/compound.log 2>&1 &
echo $! > .pids/compound.pid
sleep 5
python tools/compound_status.py
```

You should see the status dashboard show all loops green within 60 seconds. Telegram will fire "AL3X.NYC compound machine online. 10-layer autotune active. Sharpening forever."

## Acceptance criteria

- One full 15-min flywheel tick completes
- All 10 autotune Optuna studies have ≥ 1 completed trial
- One forensic loss postmortem completed
- A/B tournament wired up with ≥ 2 candidates
- One alpha-hunt cycle has written a research note
- Health dashboard prints all green
- SIGTERM handler verified (start, kill, restart, no corruption)
- Telegram online message fires
- Obsidian daily roll-up file created for today

When all nine are green: write "COMPOUND MACHINE ONLINE" to `logs/compound_status.txt` and continue running forever.
