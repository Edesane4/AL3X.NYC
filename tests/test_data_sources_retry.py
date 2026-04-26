"""Session 4 Part 3 — retry-on-5xx for Open-Meteo path.

Verifies that DataSources._get_with_retry_5xx:
  * succeeds on first try when upstream returns 200
  * retries once on 502/503/504 then returns the 200
  * retries once on connect/timeout error then returns success
  * gives up after max_attempts and returns the final 5xx response
  * does NOT retry on 4xx (client error)
  * raises the final exception when max_attempts of connect errors hit
  * passes through unexpected exception types without retry

Uses asyncio.run() inside sync test functions because pytest-asyncio
is not a project dependency.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import httpx
import pytest

from al3x.data_sources import DataSources


def _make_sources_with_responses(responses):
    """Build a DataSources whose _client.get returns the given sequence.

    `responses` is a list where each item is either:
      - an httpx.Response (will be returned)
      - an Exception (will be raised)

    Calls beyond len(responses) raise StopIteration, which signals
    a test bug (we expected fewer attempts than were made).
    """
    ds = DataSources.__new__(DataSources)  # bypass __init__ network setup
    ds._client = MagicMock()
    iter_responses = iter(responses)

    async def _fake_get(url, params=None):
        item = next(iter_responses)
        if isinstance(item, Exception):
            raise item
        return item

    ds._client.get = _fake_get
    return ds


def _resp(status_code: int, text: str = "ok") -> httpx.Response:
    """Build a synthetic httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("GET", "https://example.test/"),
        text=text,
    )


def test_first_try_200_no_retry():
    async def _go():
        ds = _make_sources_with_responses([_resp(200, "fine")])
        r = await ds._get_with_retry_5xx(
            "https://example.test/", {"k": "v"}, "test_source",
            backoff_seconds=0.0,
        )
        assert r.status_code == 200
    asyncio.run(_go())


def test_retry_on_502_then_succeeds(caplog):
    async def _go():
        ds = _make_sources_with_responses([_resp(502), _resp(200, "ok")])
        r = await ds._get_with_retry_5xx(
            "https://example.test/", {}, "ecmwf",
            backoff_seconds=0.0,
        )
        assert r.status_code == 200
    caplog.set_level(logging.INFO, logger="al3x.data")
    asyncio.run(_go())
    assert any("ecmwf" in rec.message and "502" in rec.message
               for rec in caplog.records), (
        "expected an INFO log line citing the 502 retry; "
        f"got: {[rec.message for rec in caplog.records]}"
    )


def test_retry_on_connect_error_then_succeeds(caplog):
    async def _go():
        err = httpx.ConnectError("connection refused")
        ds = _make_sources_with_responses([err, _resp(200)])
        r = await ds._get_with_retry_5xx(
            "https://example.test/", {}, "hrrr",
            backoff_seconds=0.0,
        )
        assert r.status_code == 200
    caplog.set_level(logging.INFO, logger="al3x.data")
    asyncio.run(_go())


def test_no_retry_on_404():
    """4xx is caller's bug, not server's — must not retry."""
    async def _go():
        ds = _make_sources_with_responses([_resp(404, "not found")])
        r = await ds._get_with_retry_5xx(
            "https://example.test/", {}, "test_source",
            backoff_seconds=0.0,
        )
        assert r.status_code == 404
        # If a second call had happened, the iter would have raised
        # StopIteration before we got here.
    asyncio.run(_go())


def test_exhausts_retries_returns_final_5xx():
    async def _go():
        ds = _make_sources_with_responses([_resp(503), _resp(503)])
        r = await ds._get_with_retry_5xx(
            "https://example.test/", {}, "test_source",
            max_attempts=2,
            backoff_seconds=0.0,
        )
        # Final 5xx is returned, not raised — caller's raise_for_status
        # will turn it into the actual error.
        assert r.status_code == 503
    asyncio.run(_go())


def test_exhausts_retries_raises_final_connect_error():
    async def _go():
        err1 = httpx.ConnectError("first")
        err2 = httpx.ConnectError("second")
        ds = _make_sources_with_responses([err1, err2])
        with pytest.raises(httpx.ConnectError):
            await ds._get_with_retry_5xx(
                "https://example.test/", {}, "test_source",
                max_attempts=2,
                backoff_seconds=0.0,
            )
    asyncio.run(_go())


def test_no_retry_on_unexpected_exception_type():
    """Only connect/timeout/protocol errors retry. ValueError must escape."""
    async def _go():
        ds = _make_sources_with_responses([ValueError("unexpected")])
        with pytest.raises(ValueError):
            await ds._get_with_retry_5xx(
                "https://example.test/", {}, "test_source",
                backoff_seconds=0.0,
            )
    asyncio.run(_go())
