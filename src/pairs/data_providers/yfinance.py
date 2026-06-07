"""yfinance-backed provider that mirrors :class:`PolygonProvider`'s surface.

Lets the rest of the codebase code against a single provider interface
regardless of whether a Polygon key is configured. Methods that have no
yfinance equivalent (``get_grouped_daily``, ``get_ticker_meta``) raise
:class:`ProviderError` so the caller degrades gracefully -- the Streamlit
demo, for example, falls back to the curated/custom universe when the PIT
universe option requires endpoints this provider cannot serve.

The actual fetch is delegated to :func:`pairs.data.loader.load_prices` so we
inherit the on-disk cache and validation already present in the repo.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

from pairs.data_providers.exceptions import ProviderDataError, ProviderError

logger = logging.getLogger(__name__)

_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class YFinanceProvider:
    """Drop-in replacement when ``POLYGON_API_KEY`` is not configured.

    Delegates ``get_eod`` to :func:`pairs.data.loader.load_prices` so the
    existing parquet cache and ticker-format validation continue to apply.
    The grouped-daily and ticker-meta endpoints have no yfinance equivalent
    and raise :class:`ProviderError`.
    """

    def __init__(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Public API (mirrors PolygonProvider)
    # ------------------------------------------------------------------

    def get_ticker_meta(self, ticker: str) -> dict[str, Any]:
        del ticker
        raise ProviderError(
            "get_ticker_meta requires POLYGON_API_KEY; yfinance has no equivalent"
        )

    def get_eod(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch via :func:`pairs.data.loader.load_prices` and TitleCase-normalise."""
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        # Lazy import: loader pulls in yfinance/pyarrow, which we do not want
        # at import time of this module (the factory may pick the Polygon
        # branch and skip yfinance entirely).
        from pairs.data.loader import load_prices

        ticker = ticker.strip().upper()
        # loader uses a half-open [start, end) range; widen by one day so the
        # inclusive end_date semantic that PolygonProvider exposes still holds.
        end_exclusive = end + timedelta(days=1)
        try:
            frame = load_prices(
                [ticker],
                str(start),
                str(end_exclusive),
            )
        except Exception as exc:  # noqa: BLE001 -- want any loader failure surfaced uniformly
            raise ProviderDataError(
                f"yfinance load failed for {ticker}: {exc}"
            ) from exc
        if frame is None or frame.empty:
            return pd.DataFrame(columns=list(_OHLCV_COLUMNS))
        # Flatten MultiIndex columns: load_prices returns (ticker, field).
        if isinstance(frame.columns, pd.MultiIndex):
            level_zero = frame.columns.get_level_values(0)
            if ticker in level_zero:
                frame = frame.loc[:, ticker]
            else:
                frame.columns = frame.columns.get_level_values(-1)
        # Title-case OHLCV columns so the surface matches PolygonProvider.
        rename_map = {
            col: col.title()
            for col in frame.columns
            if str(col).lower() in {"open", "high", "low", "close", "volume", "adj close"}
        }
        if rename_map:
            frame = frame.rename(columns=rename_map)
        keep = [c for c in _OHLCV_COLUMNS if c in frame.columns]
        if not keep:
            return pd.DataFrame(columns=list(_OHLCV_COLUMNS))
        out = frame.loc[:, keep].copy()
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out.index = pd.DatetimeIndex(out.index, name="Date")
        return out.sort_index()

    def get_grouped_daily(self, date_: date) -> pd.DataFrame:
        del date_
        raise ProviderError(
            "get_grouped_daily requires POLYGON_API_KEY; yfinance has no equivalent"
        )

    def close(self) -> None:  # parity with PolygonProvider
        return None
