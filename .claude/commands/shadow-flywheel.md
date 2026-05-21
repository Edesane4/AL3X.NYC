---
description: Run the AL3X.NYC shadow-trading data flywheel — paper-trade every active Kalshi NYC weather bucket every 15 min, grade against NWS CLI, build the dataset that powers all learning loops.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# /goals shadow-flywheel

## Mission

You are running AL3X.NYC's **data flywheel**. This is the prerequisite for every other learning loop in the system — isotonic calibration, sigma auto-tuner, regime classifier, EMOS coefficients, Beta fill models. None of them can learn without a large, growing, well-graded dataset of paper trades. Your job is to build and run that dataset generator until told to stop.

## What this does

Every 15 minutes (loop forever):

1. **Pull live Kalshi NYC weather markets.** Call `KalshiClient` (already in `al3x/markets/` or build it under `al3x/markets/kalshi.py` if missing). Pull every active bucket contract for KNYC high temp, KNYC low temp, and any KNYC precip / wind contracts. Persist the YES bid, YES ask, NO bid, NO ask, mid, last-trade price, volume, OI, and a UTC timestamp.

2. **Compute AL3X internal P(YES) for every bucket.** Use the existing `al3x.forecaster.produce()` pipeline against the freshest source data. Do **not** re-call expensive sources if cached within 5 minutes — read from `forecast_runs` and source tables in `al3x.db`. Apply the current EMOS / source-weighting config.

3. **Decide a paper position per bucket.** Apply the existing edge engine. If |edge| > 3¢ and Kelly size > 0, paper-enter at the ask (YES) or bid (NO) crossed +1¢ to model realistic slippage. Log entry price, internal P(YES), edge in cents, Kelly fraction, computed size, source ensemble snapshot (JSON), and active regime.

4. **Persist to `paper_trades` table.** Schema:
   ```sql
   CREATE TABLE IF NOT EXISTS paper_trades (
     id INTEGER PRIMARY KEY AUTOINCREMENT,
     opened_at_utc TEXT NOT NULL,
     target_date TEXT NOT NULL,
     metric TEXT NOT NULL,             -- 'high','low','precip','wind'
     ticker TEXT NOT NULL,
     bucket_lo REAL, bucket_hi REAL,
     side TEXT NOT NULL,               -- 'YES','NO'
     entry_price REAL NOT NULL,        -- cents 1-99
     internal_p_yes REAL NOT NULL,
     edge_cents REAL NOT NULL,
     kelly_fraction REAL NOT NULL,
     size_contracts REAL NOT NULL,
     hours_to_settle REAL NOT NULL,
     regime TEXT,                      -- 'cold_front','marine','heat_dome','calm','snow', etc.
     source_snapshot TEXT NOT NULL,    -- JSON of every source's value
     config_hash TEXT NOT NULL,        -- hash of weights/EMOS/thresholds in effect
     closed_at_utc TEXT,
     close_reason TEXT,                -- 'CLI_RESOLVED','EXPIRED','MANUAL'
     settled_value REAL,               -- actual CLI value
     pnl_cents REAL,                   -- per-contract pnl in cents
     pnl_dollars REAL,                 -- size * pnl_cents / 100 - fees
     correct INTEGER                   -- 1 if YES side won the bucket, 0 otherwise
   );
   CREATE INDEX IF NOT EXISTS idx_pt_target ON paper_trades(target_date);
   CREATE INDEX IF NOT EXISTS idx_pt_open   ON paper_trades(closed_at_utc) WHERE closed_at_utc IS NULL;
   CREATE INDEX IF NOT EXISTS idx_pt_config ON paper_trades(config_hash);
   ```

5. **Grading loop runs every hour, hard-fires at NWS CLI release time.** Pull NWS CLI for KNYC for each `target_date` where open paper trades exist. When CLI lands, resolve every open bucket on that date: compute correct, pnl_cents (YES side wins = 100 - entry, NO side wins = entry; loser = -entry/-(100-entry)), pnl_dollars (subtract 0.07/contract Kalshi taker fee). Write `closed_at_utc`, `close_reason='CLI_RESOLVED'`, etc.

6. **Daily summary at 23:00 ET.** Append a daily summary to the Obsidian vault at `{OBSIDIAN_VAULT_PATH}/AL3X/ShadowFlywheel/{YYYY-MM-DD}.md`:
   - Total paper trades opened / closed / open
   - PnL in cents & dollars, broken out by metric, regime, sigma band, edge band, lead-time bucket
   - Calibration: avg internal_p_yes vs realized win rate (10 bins)
   - Per-source MAE on today's resolved day from `source_mae_at_noon.py` style aggregation
   - Top 3 surprise misses and top 3 cleanest hits with one-line forensics

7. **Telegram heartbeat every 4h.** Short message: "Flywheel up. {open} open / {today_closed} closed today / {today_pnl_dollars:+.2f} PnL".

## Hard rules

- **No real money trades. Ever.** This is shadow only. If you find yourself reading API keys for live trading endpoints, stop. Use only price-read endpoints from Kalshi.
- **Idempotent ticks.** Don't double-open a bucket if a paper trade for that ticker is already open. Match on `(ticker, opened_at_utc within last 15min)`.
- **Never drop rows.** If grading fails, leave the row open and retry next cycle. CLI publishes can lag — give it 36h before declaring `close_reason='EXPIRED'`.
- **Config hash on every row.** Every paper trade must record the SHA256 of the active config (`source_weights`, `emos_coeffs`, `kelly_fraction`, `sigma_thresholds`, `regime_gates`). This is what makes `/goals optimize-evergreen` able to walk-forward validate across config changes.
- **Snapshot every source value.** The `source_snapshot` JSON column must contain every source's value at decision time. Without this, you cannot retroactively re-evaluate alternative weightings.

## How to run

Build everything needed. Then start it via:

```bash
nohup python -m daemon.shadow_flywheel > logs/flywheel.log 2>&1 &
echo $! > .pids/flywheel.pid
```

Or, if running inside Claude Code, just keep iterating the 15-min cycle in this session and report status every 4 cycles.

## Acceptance criteria before declaring built

- `paper_trades` table exists and inserts work end-to-end on a dry run
- One full 15-min cycle has produced ≥5 rows on a real Kalshi NYC bucket set
- The grading loop has successfully resolved at least one historical day from al3x.db's existing CLI data
- Obsidian daily summary file is produced for today
- Telegram heartbeat fires
- `.pids/flywheel.pid` exists and the process responds to `kill -0`

When all six are green, write "FLYWHEEL ONLINE" to Telegram and to `logs/flywheel_status.txt` and continue the loop.
