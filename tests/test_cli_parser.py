"""Unit tests for NWS CLI parser (FIX 1 — section-anchored MAXIMUM regex)."""

from __future__ import annotations

from pathlib import Path

from al3x.data_sources import _parse_cli

FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_parse_observed_max_despite_normal_higher_in_text() -> None:
    """Real NWS CLI: observed 78°F, normal 62°F. Parser must return 78."""
    result = _parse_cli(_read("cli_normal.txt"))
    assert result is not None, "parser returned None on valid CLI"
    assert result["recorded_high_f"] == 78.0, (
        f"expected observed max 78.0, got {result['recorded_high_f']} — "
        "parser is still matching the climatological normal"
    )
    assert result["target_date"] == "2026-04-14"


def test_parse_observed_max_when_record_block_appears_first() -> None:
    """Record-maximum block placed above TEMPERATURE (F) section.

    Defensive exclusion (RECORD / NORMAL / LAST YEAR) must keep the
    fallback regex from picking the 102 record, and the section-anchored
    path must land on the observed 94.
    """
    result = _parse_cli(_read("cli_record_first.txt"))
    assert result is not None
    assert result["recorded_high_f"] == 94.0, (
        f"expected observed max 94.0, got {result['recorded_high_f']}"
    )
    assert result["target_date"] == "2026-07-18"


def test_parse_returns_none_on_malformed_cli() -> None:
    """No TEMPERATURE section at all — must return None, not raise."""
    result = _parse_cli(_read("cli_malformed.txt"))
    assert result is None
