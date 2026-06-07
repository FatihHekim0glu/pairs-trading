"""Unit tests for :mod:`pairs.data_providers.polygon`.

Every test mocks ``httpx`` -- no live network is hit. The retry/backoff sleeps
are patched to no-ops to keep the suite fast.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest

from pairs.data_providers.exceptions import (
    ProviderAuthError,
    ProviderDataError,
    ProviderError,
    ProviderRateLimitError,
)
from pairs.data_providers.polygon import PolygonProvider, _TokenBucket


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    status_code: int = 200,
    json_payload: Any = None,
    raise_value_error_on_json: bool = False,
    text: str = "",
) -> MagicMock:
    """Return a mock object emulating :class:`httpx.Response`."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if raise_value_error_on_json:
        resp.json.side_effect = ValueError("not json")
    else:
        resp.json.return_value = json_payload if json_payload is not None else {}
    return resp


def _make_session(*responses: MagicMock) -> MagicMock:
    """Build a mock ``httpx.Client`` that returns ``responses`` in order."""
    session = MagicMock(spec=httpx.Client)
    session.request.side_effect = list(responses)
    return session


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable backoff sleeps so retry-heavy tests run instantly."""
    monkeypatch.setattr(
        "pairs.data_providers.polygon.time.sleep", lambda *_a, **_k: None
    )


# ---------------------------------------------------------------------------
# Construction / auth
# ---------------------------------------------------------------------------


def test_polygon_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing without a key (or env var) raises ProviderAuthError."""
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(ProviderAuthError):
        PolygonProvider(api_key=None)


def test_polygon_uses_env_var_when_no_explicit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no explicit key is passed, ``POLYGON_API_KEY`` is consulted."""
    monkeypatch.setenv("POLYGON_API_KEY", "env-key-123")
    session = _make_session(_make_response(200, {"results": []}))
    provider = PolygonProvider(session=session)
    provider.get_grouped_daily(date(2024, 1, 2))
    called_with = session.request.call_args
    assert called_with.kwargs["params"]["apiKey"] == "env-key-123"


# ---------------------------------------------------------------------------
# EOD / aggregates
# ---------------------------------------------------------------------------


def test_get_eod_returns_titlecase_ohlcv_columns() -> None:
    """The eod frame must expose Open/High/Low/Close/Volume columns."""
    bar = {
        "t": int(pd.Timestamp("2024-01-02", tz="UTC").timestamp() * 1000),
        "o": 100.0,
        "h": 105.0,
        "l": 99.5,
        "c": 104.0,
        "v": 1_000_000,
    }
    session = _make_session(_make_response(200, {"results": [bar]}))
    provider = PolygonProvider(api_key="k", session=session)
    df = provider.get_eod("aapl", date(2024, 1, 2), date(2024, 1, 2))
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.name == "Date"
    assert df["Close"].iloc[0] == pytest.approx(104.0)


def test_get_eod_empty_results_returns_empty_frame() -> None:
    """A 200 with empty ``results`` collapses to an empty TitleCase frame."""
    session = _make_session(_make_response(200, {"results": []}))
    provider = PolygonProvider(api_key="k", session=session)
    df = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 2))
    assert df.empty
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_get_eod_rejects_inverted_range() -> None:
    """``start > end`` is a ValueError before any HTTP traffic."""
    provider = PolygonProvider(api_key="k", session=_make_session())
    with pytest.raises(ValueError):
        provider.get_eod("AAPL", date(2024, 1, 5), date(2024, 1, 1))


# ---------------------------------------------------------------------------
# Retry / status codes
# ---------------------------------------------------------------------------


def test_polygon_retries_on_429_then_succeeds() -> None:
    """A 429 followed by a 200 must yield the parsed payload (no exception)."""
    session = _make_session(
        _make_response(429),
        _make_response(200, {"results": []}),
    )
    provider = PolygonProvider(api_key="k", session=session)
    df = provider.get_grouped_daily(date(2024, 1, 2))
    assert df.empty
    assert session.request.call_count == 2


def test_polygon_raises_rate_limit_after_exhausting_retries() -> None:
    """Three consecutive 429s exhaust the retry budget and raise."""
    session = _make_session(
        _make_response(429),
        _make_response(429),
        _make_response(429),
    )
    provider = PolygonProvider(api_key="k", session=session)
    with pytest.raises(ProviderRateLimitError):
        provider.get_grouped_daily(date(2024, 1, 2))


def test_polygon_raises_auth_error_on_401() -> None:
    """401 is non-retryable -- surfaces as ProviderAuthError immediately."""
    session = _make_session(_make_response(401, text="unauthorized"))
    provider = PolygonProvider(api_key="bad", session=session)
    with pytest.raises(ProviderAuthError):
        provider.get_grouped_daily(date(2024, 1, 2))


def test_polygon_raises_auth_error_on_403() -> None:
    """403 is treated the same as 401 -- ProviderAuthError, no retry."""
    session = _make_session(_make_response(403))
    provider = PolygonProvider(api_key="bad", session=session)
    with pytest.raises(ProviderAuthError):
        provider.get_grouped_daily(date(2024, 1, 2))


def test_polygon_retries_on_500_then_succeeds() -> None:
    """A 5xx followed by a 200 must yield the parsed payload."""
    session = _make_session(
        _make_response(503),
        _make_response(200, {"results": []}),
    )
    provider = PolygonProvider(api_key="k", session=session)
    df = provider.get_grouped_daily(date(2024, 1, 2))
    assert df.empty
    assert session.request.call_count == 2


def test_polygon_raises_provider_error_on_4xx_other() -> None:
    """A non-auth 4xx (e.g. 400) raises a generic ProviderError without retry."""
    session = _make_session(_make_response(400, text="bad request"))
    provider = PolygonProvider(api_key="k", session=session)
    with pytest.raises(ProviderError):
        provider.get_grouped_daily(date(2024, 1, 2))


def test_polygon_retries_on_network_error_then_succeeds() -> None:
    """A transient httpx error must be retried, not propagated."""
    session = MagicMock(spec=httpx.Client)
    session.request.side_effect = [
        httpx.ConnectError("boom"),
        _make_response(200, {"results": []}),
    ]
    provider = PolygonProvider(api_key="k", session=session)
    df = provider.get_grouped_daily(date(2024, 1, 2))
    assert df.empty
    assert session.request.call_count == 2


# ---------------------------------------------------------------------------
# Malformed payloads
# ---------------------------------------------------------------------------


def test_polygon_raises_data_error_on_non_json() -> None:
    """A 200 with a non-JSON body becomes ProviderDataError, not bubbled ValueError."""
    session = _make_session(_make_response(200, raise_value_error_on_json=True))
    provider = PolygonProvider(api_key="k", session=session)
    with pytest.raises(ProviderDataError):
        provider.get_grouped_daily(date(2024, 1, 2))


def test_get_ticker_meta_raises_on_malformed_results() -> None:
    """``results`` must be a dict; anything else raises ProviderDataError."""
    session = _make_session(_make_response(200, {"results": "not-a-dict"}))
    provider = PolygonProvider(api_key="k", session=session)
    with pytest.raises(ProviderDataError):
        provider.get_ticker_meta("AAPL")


# ---------------------------------------------------------------------------
# Grouped daily
# ---------------------------------------------------------------------------


def test_get_grouped_daily_indexes_by_ticker() -> None:
    """grouped-daily output must index by symbol with TitleCase OHLCV columns."""
    payload = {
        "results": [
            {"T": "AAPL", "o": 100, "h": 110, "l": 99, "c": 105, "v": 1_000_000},
            {"T": "MSFT", "o": 200, "h": 210, "l": 199, "c": 205, "v": 2_000_000},
        ]
    }
    session = _make_session(_make_response(200, payload))
    provider = PolygonProvider(api_key="k", session=session)
    df = provider.get_grouped_daily(date(2024, 1, 2))
    assert df.index.name == "ticker"
    assert {"AAPL", "MSFT"}.issubset(set(df.index))
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.loc["AAPL", "Close"] == pytest.approx(105)


def test_get_grouped_daily_skips_rows_without_symbol() -> None:
    """Malformed rows missing ``T`` must be silently dropped (not crash)."""
    payload = {
        "results": [
            {"T": "AAPL", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1},
            {"o": 1, "h": 1, "l": 1, "c": 1, "v": 1},  # no T
        ]
    }
    session = _make_session(_make_response(200, payload))
    provider = PolygonProvider(api_key="k", session=session)
    df = provider.get_grouped_daily(date(2024, 1, 2))
    assert list(df.index) == ["AAPL"]


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


def test_token_bucket_does_not_block_under_rpm() -> None:
    """Acquiring fewer tokens than the rpm ceiling must not sleep."""
    bucket = _TokenBucket(rpm=5)
    # Under the cap, acquisitions return immediately.
    for _ in range(5):
        bucket.acquire()
    assert len(bucket._window) == 5  # noqa: SLF001 -- white-box check


def test_token_bucket_evicts_old_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sliding window must drop entries older than the configured period."""
    bucket = _TokenBucket(rpm=2)
    fake_now = [1000.0]
    monkeypatch.setattr(
        "pairs.data_providers.polygon.time.monotonic", lambda: fake_now[0]
    )
    bucket.acquire()
    bucket.acquire()
    # Advance well past the 60-second window; next acquire must succeed without sleeping.
    fake_now[0] += 120.0
    bucket.acquire()
    assert len(bucket._window) == 1  # noqa: SLF001 -- white-box check


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_close_does_not_close_externally_owned_session() -> None:
    """The provider must not close a session it did not create."""
    external = MagicMock(spec=httpx.Client)
    external.request.return_value = _make_response(200, {"results": []})
    provider = PolygonProvider(api_key="k", session=external)
    provider.close()
    external.close.assert_not_called()
