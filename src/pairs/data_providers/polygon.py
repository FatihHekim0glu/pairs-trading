"""Polygon.io REST client with token-bucket rate limiting and survivorship-aware endpoints.

Public surface is :class:`PolygonProvider`; the factory in
:mod:`pairs.data_providers.factory` is the normal entry point.

Why Polygon
-----------
The free yfinance feed silently omits delisted tickers. Pairs trading is the
strategy class most sensitive to that omission: about one in ten pairs in a
typical five-year window contains a name that was acquired or delisted, and
those pairs disproportionately look like winners in hindsight. Polygon's
grouped-daily endpoint returns every ticker that *actually* traded on a given
historical calendar date, which is what we need to build an honest point-in-time
universe.

Rate-limit / retry
------------------
A simple sliding-window token bucket caps outbound traffic at 100 requests per
minute (Polygon Starter tier). Each transient failure (429 / 5xx / network)
triggers exponential backoff with three attempts (1s, 2s, 4s + jitter). The
bucket is process-local; we assume one process per Streamlit session.

Adjusted vs raw
---------------
We always request ``adjusted=true`` (split + dividend). The pairs strategy
expects adjusted close.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta
from typing import Any

import httpx
import pandas as pd

from pairs.data_providers.exceptions import (
    ProviderAuthError,
    ProviderDataError,
    ProviderError,
    ProviderRateLimitError,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io"
_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 1.0
_STARTER_RPM = 100
_RATE_WINDOW_SECONDS = 60.0
_HTTP_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Sliding-window rate limiter -- at most ``rpm`` requests per 60 seconds."""

    def __init__(self, rpm: int = _STARTER_RPM) -> None:
        self._rpm = rpm
        self._window: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            cutoff = now - _RATE_WINDOW_SECONDS
            while self._window and self._window[0] < cutoff:
                self._window.popleft()
            if len(self._window) >= self._rpm:
                wait = _RATE_WINDOW_SECONDS - (now - self._window[0]) + 0.05
                if wait > 0:
                    time.sleep(wait)
                now = time.monotonic()
                cutoff = now - _RATE_WINDOW_SECONDS
                while self._window and self._window[0] < cutoff:
                    self._window.popleft()
            self._window.append(now)


# ---------------------------------------------------------------------------
# Main provider
# ---------------------------------------------------------------------------


class PolygonProvider:
    """Polygon.io REST adapter with point-in-time accuracy.

    Parameters
    ----------
    api_key
        Polygon API key. Falls back to the ``POLYGON_API_KEY`` env var.
    session
        Optional pre-configured ``httpx.Client``. The provider takes ownership
        only when it constructs the client itself.
    rpm
        Token-bucket ceiling in requests/minute. Defaults to the Polygon
        Starter tier limit (100).
    """

    def __init__(
        self,
        api_key: str | None = None,
        session: httpx.Client | None = None,
        rpm: int = _STARTER_RPM,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("POLYGON_API_KEY", "")
        if not self._api_key:
            raise ProviderAuthError(
                "POLYGON_API_KEY is required to instantiate PolygonProvider"
            )
        self._owns_session = session is None
        self._session = session or httpx.Client(timeout=_HTTP_TIMEOUT)
        self._bucket = _TokenBucket(rpm=rpm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ticker_meta(self, ticker: str) -> dict[str, Any]:
        """Return the ``/v3/reference/tickers/{ticker}`` payload as a dict."""
        ticker = ticker.strip().upper()
        payload = self._request("GET", f"/v3/reference/tickers/{ticker}")
        results = payload.get("results")
        if not isinstance(results, dict):
            raise ProviderDataError(f"No reference data returned for {ticker}")
        return results

    def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Return daily OHLCV for ``ticker`` in ``[start, end]`` (both inclusive).

        Output is TitleCase (Open/High/Low/Close/Volume) with a tz-naive
        ``DatetimeIndex`` named ``Date``. Close is split- and dividend-adjusted
        (Polygon ``adjusted=true``).
        """
        ticker = ticker.strip().upper()
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        return self._fetch_aggs(ticker, start, end)

    def get_grouped_daily(self, date_: date) -> pd.DataFrame:
        """Grouped-daily snapshot of every actively-traded US stock on ``date_``.

        Index is the ticker symbol; columns are TitleCase OHLCV. Used by the
        S&P 500 universe builder to know which symbols actually traded on a
        given historical date -- which is precisely the survivorship-bias
        question this module exists to answer.
        """
        path = f"/v2/aggs/grouped/locale/us/market/stocks/{date_.isoformat()}"
        payload = self._request("GET", path, params={"adjusted": "true"})
        results = payload.get("results") or []
        if not results:
            return pd.DataFrame(columns=list(_OHLCV_COLUMNS))
        rows: dict[str, dict[str, float]] = {}
        for bar in results:
            symbol = bar.get("T")
            if not symbol:
                continue
            rows[symbol] = {
                "Open": float(bar.get("o", 0.0)),
                "High": float(bar.get("h", 0.0)),
                "Low": float(bar.get("l", 0.0)),
                "Close": float(bar.get("c", 0.0)),
                "Volume": float(bar.get("v", 0.0)),
            }
        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "ticker"
        return df

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{_BASE_URL}{path}"
        merged_params = dict(params or {})
        merged_params["apiKey"] = self._api_key
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            self._bucket.acquire()
            try:
                response = self._session.request(method, url, params=merged_params)
            except httpx.HTTPError as exc:
                last_exc = exc
                self._sleep_backoff(attempt)
                continue
            status_code = response.status_code
            if status_code in (401, 403):
                raise ProviderAuthError(
                    f"Polygon auth rejected ({status_code}); check POLYGON_API_KEY"
                )
            if status_code == 429:
                last_exc = ProviderRateLimitError(
                    f"Polygon rate limit hit (attempt {attempt + 1}/{_MAX_ATTEMPTS})"
                )
                self._sleep_backoff(attempt)
                continue
            if status_code >= 500:
                last_exc = ProviderError(
                    f"Polygon {status_code} on {path} (attempt {attempt + 1}/{_MAX_ATTEMPTS})"
                )
                self._sleep_backoff(attempt)
                continue
            if status_code >= 400:
                raise ProviderError(
                    f"Polygon {status_code} on {path}: {response.text[:200]}"
                )
            try:
                return response.json()
            except ValueError as exc:
                raise ProviderDataError(
                    f"Polygon returned non-JSON for {path}"
                ) from exc
        if isinstance(last_exc, ProviderRateLimitError):
            raise last_exc
        raise ProviderError(
            f"Polygon request failed after {_MAX_ATTEMPTS} attempts on {path}"
        ) from last_exc

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        delay = _BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0.0, 0.25)
        time.sleep(delay)

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def _fetch_aggs(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        path = (
            f"/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}"
        )
        payload = self._request(
            "GET",
            path,
            params={"adjusted": "true", "sort": "asc", "limit": 50_000},
        )
        results = payload.get("results") or []
        if not results:
            return pd.DataFrame(columns=list(_OHLCV_COLUMNS))
        records: list[dict[str, Any]] = []
        for bar in results:
            ts_ms = bar.get("t")
            if ts_ms is None:
                continue
            # Polygon's daily aggregates encode the trading day as the UTC
            # midnight timestamp of that calendar date. Do NOT shift to
            # America/New_York -- that would move the bar back one day.
            records.append(
                {
                    "Date": pd.Timestamp(ts_ms, unit="ms", tz="UTC")
                    .tz_localize(None)
                    .normalize(),
                    "Open": float(bar.get("o", 0.0)),
                    "High": float(bar.get("h", 0.0)),
                    "Low": float(bar.get("l", 0.0)),
                    "Close": float(bar.get("c", 0.0)),
                    "Volume": float(bar.get("v", 0.0)),
                }
            )
        if not records:
            return pd.DataFrame(columns=list(_OHLCV_COLUMNS))
        df = pd.DataFrame.from_records(records).set_index("Date").sort_index()
        df.index = pd.DatetimeIndex(df.index, name="Date")
        return df


# Silence "imported but unused" for datetime/timedelta -- kept for future use in
# rate-limit reset windows.
_ = (datetime, timedelta)
