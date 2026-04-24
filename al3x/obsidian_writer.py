"""Session 2 G14 — write daily check-ins and forecast-event notes to
the user's Obsidian vault.

The vault path is configured via the ``OBSIDIAN_VAULT_PATH`` env var.
If unset or pointing to a non-existent directory, all write functions
become no-ops and log at DEBUG.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

log = logging.getLogger("al3x.obsidian")


def _vault_path() -> Optional[Path]:
    path_str = os.getenv("OBSIDIAN_VAULT_PATH")
    if not path_str:
        return None
    p = Path(path_str).expanduser()
    if not p.exists():
        log.warning("OBSIDIAN_VAULT_PATH does not exist: %s", p)
        return None
    return p


def write_daily_checkin(content: str, target_date: date) -> bool:
    """Write the daily check-in note (overwrites if it exists).

    Returns True if the note was written, False if Obsidian integration
    is disabled (no env var, or the vault path is missing).
    """
    vault = _vault_path()
    if vault is None:
        log.debug("Obsidian vault not configured; skipping daily check-in")
        return False
    checkin_dir = vault / "05-Daily Check-ins"
    checkin_dir.mkdir(exist_ok=True)
    path = checkin_dir / f"{target_date.isoformat()}.md"
    path.write_text(content)
    log.info("Wrote daily check-in to %s", path)
    return True


def append_event(event_type: str, content: str, target_date: date) -> bool:
    """Append a forecast-event note (one file per day per event type).

    Each append writes ``content`` followed by a horizontal rule so
    multiple events of the same type for the same date stack cleanly.
    Returns True if appended, False if Obsidian integration is disabled.
    """
    vault = _vault_path()
    if vault is None:
        return False
    events_dir = vault / "06-Forecast Events"
    events_dir.mkdir(exist_ok=True)
    path = events_dir / f"{target_date.isoformat()}-{event_type}.md"
    with path.open("a") as f:
        f.write(content + "\n---\n")
    log.info("Appended %s event to %s", event_type, path)
    return True
