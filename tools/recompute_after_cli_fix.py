"""Recompute CLI truth, scores, and retunes after FIX 1.

Before FIX 1 (CLI parser section anchoring), every verified CLI truth
was the climatological normal high, not the observed daily high. Every
score, bias retune, weight retune, and attribution row built on top of
those scores was wrong.

After deploying FIX 1, run this script MANUALLY (operator review the
results before accepting):

    python3 tools/recompute_after_cli_fix.py --dry-run
    python3 tools/recompute_after_cli_fix.py --apply

Backfill strategy: NWS only serves the latest CLI at the standard
endpoint. Historical CLI products go through the version-walk endpoint:

    https://forecast.weather.gov/product.php?site=OKX&issuedby=NYC
        &product=CLI&format=CI&version=N&glossary=0

where N walks backwards through the version history. Versions 1..~90
cover roughly the last 2-3 months of daily products.

This script:
  1. Walks the version history until it has seen `days` distinct CLI
     target_dates or until the template stops yielding a new one.
  2. Reparses each fetched CLI and UPSERTs the truth row.
  3. Optionally deletes all rows from bias_state and source_weights so
     the next retune starts from defaults, and kicks off a fresh
     retune.
  4. Reports per-day diffs (old_high → new_high) for operator review.

Safe to re-run. Use --dry-run to see the diff before actually
modifying the DB.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Resolve repo root when invoked as `python3 tools/...`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from al3x.data_sources import DataSources, _parse_cli  # noqa: E402
from al3x.learning import Learning  # noqa: E402
from al3x.storage import Storage  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s")
log = logging.getLogger("recompute_cli")


async def _fetch_version(sources: DataSources, version: int) -> Optional[str]:
    """Fetch one version of the NYC CLI product. Returns HTML text or None."""
    url = ("https://forecast.weather.gov/product.php"
           f"?site=OKX&issuedby=NYC&product=CLI&format=CI"
           f"&version={version}&glossary=0")
    try:
        r = await sources._client.get(url, timeout=20.0)  # noqa: SLF001
        if r.status_code != 200 or len(r.text) < 200:
            return None
        return r.text
    except Exception as e:  # noqa: BLE001
        log.warning("fetch v%d error: %s", version, e)
        return None


async def backfill_cli(storage: Storage, days: int, apply: bool
                       ) -> List[Tuple[str, Optional[float], Optional[float]]]:
    """Walk version history and upsert corrected CLI truth.

    Returns a list of (target_date, old_high_f, new_high_f) for review.
    """
    sources = DataSources()
    try:
        seen: Dict[str, Dict[str, Any]] = {}
        max_version = 120  # walk roughly 4 months of daily products
        for version in range(1, max_version + 1):
            if len(seen) >= days:
                break
            text = await _fetch_version(sources, version)
            if not text:
                continue
            parsed = _parse_cli(text)
            if not parsed or not parsed.get("target_date"):
                continue
            td = parsed["target_date"]
            if td in seen:
                continue
            seen[td] = parsed
            log.info("v%-3d covers %s → high %.1f°F", version, td,
                     parsed["recorded_high_f"])

        diffs: List[Tuple[str, Optional[float], Optional[float]]] = []
        today = date.today().isoformat()
        for td, parsed in seen.items():
            if td >= today:
                continue  # never overwrite future/today (not yet verified)
            existing = storage.get_cli_truth(td)
            old = (float(existing["recorded_high_f"])
                   if existing and existing.get("recorded_high_f") is not None
                   else None)
            new = float(parsed["recorded_high_f"])
            if old is None or abs(old - new) > 1e-6:
                diffs.append((td, old, new))
                if apply:
                    storage.save_cli_truth(td, new, parsed["posted_at"],
                                           parsed["raw_text"])
        return diffs
    finally:
        await sources.close()


def rescore_days(storage: Storage, learning: Learning, days: int) -> int:
    today = date.today()
    scored = 0
    for days_ago in range(1, days + 1):
        target = (today - timedelta(days=days_ago)).isoformat()
        truth = storage.get_cli_truth(target)
        if not truth or truth.get("recorded_high_f") is None:
            continue
        res = learning.score_day(target, float(truth["recorded_high_f"]))
        scored += int(res.get("scored") or 0)
    return scored


def wipe_tuning_state(storage: Storage) -> None:
    """Reset bias_state and source_weights so retunes re-derive from defaults
    against the now-corrected scores."""
    with storage._conn() as c:  # noqa: SLF001
        c.execute("DELETE FROM bias_state")
        c.execute("DELETE FROM source_weights")
        c.execute("DELETE FROM correction_attribution")
        c.execute("DELETE FROM scores")  # so rescore is authoritative


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("AL3X_DB_PATH", "./al3x.db"))
    ap.add_argument("--days", type=int, default=60,
                    help="number of calendar days to backfill (default: 60)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print diffs only; do not modify DB")
    ap.add_argument("--apply", action="store_true",
                    help="write corrected truths, wipe tuning state, rescore")
    args = ap.parse_args()

    if not args.dry_run and not args.apply:
        log.error("must pass --dry-run or --apply")
        return 2

    storage = Storage(args.db)
    learning = Learning(storage)

    log.info("Step 1: backfilling CLI truth from version-walk "
             "(target %d days) — %s mode",
             args.days, "APPLY" if args.apply else "DRY-RUN")
    diffs = await backfill_cli(storage, days=args.days, apply=args.apply)
    if not diffs:
        log.info("No corrections needed — existing truths already match.")
    else:
        log.info("Truth diffs (%d rows):", len(diffs))
        log.info(f"  {'date':<12} {'old':>7}  {'new':>7}  {'Δ':>6}")
        for td, old, new in sorted(diffs):
            old_s = f"{old:.1f}" if old is not None else "   —"
            delta = (f"{new - old:+.1f}" if old is not None else "   NEW")
            log.info(f"  {td:<12} {old_s:>7}  {new:>7.1f}  {delta:>6}")

    if not args.apply:
        log.info("Dry-run complete. Re-run with --apply to persist.")
        return 0

    log.info("Step 2: wiping bias_state, source_weights, "
             "correction_attribution, scores")
    wipe_tuning_state(storage)

    log.info("Step 3: rescoring each verified day")
    scored = rescore_days(storage, learning, days=args.days)
    log.info("Scored %d forecast rows", scored)

    log.info("Step 4: retune weights (alpha=0.50)")
    w_res = learning.retune_weights(blend_alpha=0.50)
    log.info("weights retune: %s", w_res)

    log.info("Step 5: retune biases")
    b_res = learning.retune_biases()
    log.info("biases retune: %s", b_res)

    log.info("Done. Next intraday cycle will re-train QRF on the "
             "corrected score history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
