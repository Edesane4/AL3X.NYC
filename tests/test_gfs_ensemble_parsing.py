"""FIX 3 — GFS/GEFS ensemble member parsing against Open-Meteo.

Evidence: 0/200 successful live calls; every call returned "only 1
members parsed". Open-Meteo returns the control run as
``temperature_2m`` and perturbed members as ``temperature_2m_memberNN``
— the old ``startswith`` grouping collapsed both and then failed to
collect them correctly.
"""

from __future__ import annotations

from al3x.data_sources import _parse_gfs_ensemble_members


def _fake_payload(n_members: int,
                   include_control: bool = True,
                   n_hours: int = 24,
                   base_temp: float = 70.0) -> dict:
    """Build a minimal Open-Meteo-style hourly payload.

    Every member gets a unique flat-line temperature so the max is
    distinguishable across members.
    """
    hourly: dict = {}
    if include_control:
        hourly["temperature_2m"] = [base_temp] * n_hours
    for i in range(1, n_members + 1):
        hourly[f"temperature_2m_member{i:02d}"] = [base_temp + i * 0.1] * n_hours
    # Add some unrelated keys that must be ignored by the parser
    hourly["relativehumidity_2m"] = [50.0] * n_hours
    hourly["time"] = ["2026-04-22T00:00"] * n_hours
    return hourly


def test_thirty_members_plus_control_parsed() -> None:
    """Control + 30 perturbed members with distinct values → 31 unique."""
    hourly = _fake_payload(n_members=30)
    idx_today = list(range(24))  # all hours are target-date
    members = _parse_gfs_ensemble_members(hourly, idx_today)
    assert len(members) >= 30, (
        f"expected at least 30 unique members, got {len(members)}"
    )
    assert len(members) == 31  # control + 30


def test_only_control_returns_single_member() -> None:
    """Only ``temperature_2m`` is served (no perturbed members) → 1 member
    parsed, which triggers the upstream caller to return an error string."""
    hourly = _fake_payload(n_members=0, include_control=True)
    idx_today = list(range(24))
    members = _parse_gfs_ensemble_members(hourly, idx_today)
    assert len(members) == 1, (
        f"expected 1 member (control only), got {len(members)}"
    )


def test_control_duplicated_as_member00_is_deduped() -> None:
    """Some feeds serve the control both as ``temperature_2m`` and as
    ``temperature_2m_member00`` with identical values — dedupe by
    rounded max so the control isn't double-counted."""
    hourly = _fake_payload(n_members=5)
    hourly["temperature_2m_member00"] = hourly["temperature_2m"]
    idx_today = list(range(24))
    members = _parse_gfs_ensemble_members(hourly, idx_today)
    # Control + 5 perturbed = 6 unique (member00 == control, dedupe drops one)
    assert len(members) == 6


def test_unrelated_keys_ignored() -> None:
    """``relativehumidity_2m`` is not a member series and must not be
    treated as one."""
    hourly = _fake_payload(n_members=3)
    idx_today = list(range(24))
    members = _parse_gfs_ensemble_members(hourly, idx_today)
    # control + 3 members = 4; would be 5 if RH key leaked in
    assert len(members) == 4
