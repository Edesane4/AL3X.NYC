# AL3X.NYC `/goals` — The Compound Machine v2

Four Claude Code slash commands that turn AL3X.NYC into a fully self-tuning system. Each runs autonomously and compounds results over time.

## What changed in v2

`/goals autotune` supersedes the older `/goals optimize-evergreen`. v2 adds:
- 10 parallel optimization layers (was 1 monolithic search)
- Per-strategy, per-regime, per-time-of-day tuning
- Forensic loss postmortems on every losing trade
- Causal counterfactual on every promotion with auto-revert
- A/B shadow tournament on LIVE paper trades (distribution-shift protection)
- Pareto-frontier multi-objective optimization (Sharpe + drawdown + win rate + CVaR)
- Strategy auto-disable/re-enable with statistical confidence
- Pairwise source-bias learning
- Meta-learner that biases search toward historically winning parameter directions

## Installation

From your AL3X.NYC repo root on the Mac:

```bash
# 1. Make sure these directories exist
mkdir -p .claude/commands al3x/learning daemon tools logs .pids configs

# 2. Drop all four .md files from this package into .claude/commands/

# 3. Commit
git add .claude/commands/
git commit -m "Add /goals compound machine v2 (autotune supersedes optimize-evergreen)"
git push
```

## The four commands

| Command | Purpose | Cadence |
|---|---|---|
| `/goals shadow-flywheel` | Paper-trade every Kalshi NYC bucket every 15 min, grade vs NWS CLI | 15-min ticks |
| `/goals autotune` | 10-layer Optuna + forensics + A/B + Pareto. Reviews every trade, tunes every parameter. | Continuous |
| `/goals alpha-hunt` | Auto-research literature, implement shadow sources, promote winners | Daily + Sunday evals |
| `/goals compound` | All three in one supervised daemon. Cross-wires their outputs. | Single process, 24/7 |

## Recommended startup

**Option A — All in (recommended):**
```
/goals compound
```
Single command. Walk away.

**Option B — Phased rollout:**
1. Day 1: `/goals shadow-flywheel` — get paper trades flowing
2. Day 3: `/goals autotune` — start the 10-layer optimizer
3. Day 7: `/goals alpha-hunt` — start finding new techniques
4. Day 14: kill all three, then `/goals compound` to unify under supervisor

## Required environment vars in `.env`

```
KALSHI_API_KEY=...           # read-only is fine; no live trades placed
OBSIDIAN_VAULT_PATH=/Users/you/Obsidian/AL3X
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
ANTHROPIC_API_KEY=...          # for alpha-hunt research summaries
```

## Migrating from v1 (if you already ran optimize-evergreen)

The autotune.md slash command includes a migration block. Run `/goals autotune` and it will detect the old study and migrate trial data into the new 10-layer schema automatically.

## Stopping

```bash
bash tools/compound_kill.sh
```

## Monitoring

```bash
python tools/compound_status.py    # full dashboard
tail -f logs/compound.log          # raw stream
```

Or just open today's note in Obsidian: `AL3X/Compound/{today}.md`. Scroll to the bottom for the single number that matters:

> **"AL3X.NYC is +X% sharper than 30 days ago"**

Trending up = the compound machine is working. Flat or negative for 60+ days = something is broken; audit.
