---
description: Autonomous research agent. Crawl arXiv, AMS, NWS, BAMS, prediction-market microstructure papers; implement candidate techniques as shadow sources; auto-promote winners after 30-day parallel run.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, WebSearch, WebFetch
---

# /goals alpha-hunt

## Mission

AL3X.NYC's existing 7 weather sources and probability stack are a starting point, not a ceiling. Your job is to find new techniques in the academic and operational literature that could improve forecast accuracy or market edge, test them in parallel with the live system, and promote winners — forever.

## Search domains (rotate daily)

1. **Ensemble post-processing** — EMOS, NGR, BMA, ECC, Schaake shuffle, copula-based recalibration, IDR, isotonic distributional regression
2. **Neural/ML forecasting** — Pangu-Weather, GraphCast, FuXi, FourCastNet, MetNet-3 — public weights and API availability
3. **Mesoscale features** — NYC urban heat island climatology, JFK/LGA/EWR/NYC station differentials, sea-breeze indices, NYC microclimate papers from journals like *Weather and Forecasting*
4. **NWS technical** — NBM v5+ release notes, GFSv17 changes, HRRR-Ensemble specs, NDFD updates
5. **Prediction-market microstructure** — Kalshi market-maker behavior papers, order-flow toxicity (VPIN), Avellaneda-Stoikov, prediction-market mispricing literature
6. **Calibration techniques** — Platt scaling, isotonic regression, beta calibration, temperature scaling, conformal prediction
7. **Climatology data sources** — NOAA datasets we don't pull yet (NCEI hourly QCLCD, MADIS, AWOS-3, lightning data, satellite IR/VIS)

## Loop (daily)

```
1. Pick today's domain from the rotation.
2. Web search 5–10 queries within that domain for papers/articles published
   in the last 12 months. Prefer arXiv, AMS journals, BAMS, AGU, NOAA tech
   notes, Bank of England / Fed working papers (for microstructure).
3. For each candidate, write a 1-paragraph summary to
   {OBSIDIAN_VAULT_PATH}/AL3X/Research/Inbox/{YYYY-MM-DD}_{slug}.md
4. Score each candidate on:
     - Implementability (free API? open weights? pure math?) 1–5
     - Expected MAE improvement at noon intraday 1–5
     - Implementation cost in hours 1–5 (lower = better)
     - Data dependency risk 1–5
   Compute total: implementability + expected_lift + (6 - cost) + (6 - risk).
5. Auto-select top candidate scoring >= 14. If none, promote the highest
   scorer to the human review queue in Obsidian and continue.
6. Implement the selected candidate as a SHADOW SOURCE — meaning it writes
   to source_snapshot under a new key (e.g. 'shadow_graphcast') but
   does NOT enter the production weight set.
7. Backfill on the last 60 days of stored forecast data where possible.
   If the source can only run forward, start fresh.
8. The shadow source now runs every cycle alongside production.
```

## Shadow source registry

Build `al3x/learning/shadow_sources.py` with this contract:

```python
class ShadowSource:
    name: str               # 'shadow_graphcast', 'shadow_emos_v2', etc.
    introduced_at: datetime
    description: str        # one-liner describing the technique
    paper_url: str | None
    
    def predict(self, target_date, metric) -> float | None:
        """Return forecast value. Cache aggressively. Never block production."""
```

Every shadow source must:
- Never raise into the production scan loop. Failures are logged and the source returns None.
- Cost zero or near-zero $ to operate. If it requires a paid API > $10/mo, route to human review queue.
- Have a paper or doc URL on file.

## Evaluation loop (runs every Sunday)

```
For each shadow source with >= 30 days of data and >= 30 graded paper trades
using its values in source_snapshot:

1. Compute MAE, ME, RMSE, CRPS at the noon intraday window vs CLI truth.
2. Compute what production paper PnL would have been if this source had been
   included with a Bayesian-optimal weight (run a 1-D optimization just for it).
3. Compute statistical significance vs status-quo PnL: bootstrap 5000 resamples,
   require 95% CI of PnL lift > 0.
4. If pass: write to configs/promotion_queue.json with proposed weight.
5. The optimize-evergreen daemon will pick it up on the next param search
   cycle and find its true optimal weight in the joint space.
```

## Sources to chase first (prepriotized — go in this order)

1. **GraphCast / Pangu-Weather public inference** via huggingface or Google's public endpoint. Free, ML weather model, has shown sub-IFS MAE in many regions. NYC point extraction is straightforward.
2. **NBM v4.2 probabilistic guidance** — already in NWS, just need the right product code. Gives natively probabilistic forecasts.
3. **NCEI Local Climatological Data hourly** — full NYC station history for richer climatology priors.
4. **Conformal prediction wrapper** around the existing ensemble — gives statistically valid bucket probabilities at any miscoverage level without retraining.
5. **VPIN order-flow toxicity** on Kalshi tape — detects when informed traders are in the book and shadow-trades should be skipped.

Implement these in order before the autonomous discovery loop starts producing its own.

## Outputs

- `{OBSIDIAN_VAULT_PATH}/AL3X/Research/Inbox/` — every paper read
- `{OBSIDIAN_VAULT_PATH}/AL3X/Research/Shadow/` — every implemented shadow source with weekly performance
- `{OBSIDIAN_VAULT_PATH}/AL3X/Research/Promoted/` — every source that made it to production with full receipts
- `configs/promotion_queue.json` — pending promotions for optimize-evergreen
- Telegram: weekly digest "Alpha-hunt: {n} papers read / {n} shadows running / {n} promoted / best lift +{x}¢ edge"

## Acceptance criteria

- 10 papers in inbox
- 1 shadow source implemented end-to-end and producing values
- Weekly evaluation report runs successfully
- Telegram digest fires

Then loop forever.
