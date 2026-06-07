"""Vendor-agnostic price providers with survivorship-bias-aware universes.

Pairs trading is more sensitive to survivorship bias than most quant strategies
because the trade thesis is *relative*: a pair containing a name that later got
acquired or delisted at a wide spread looks like a clean mean-reversion winner
in hindsight. Roughly one in ten pairs in any five-year window includes such a
name, biasing reported Sharpes upward by a non-trivial margin.

This subpackage provides:

* :class:`~pairs.data_providers.polygon.PolygonProvider` -- token-bucket-rate-limited
  REST client for Polygon.io with retry on 429/5xx and an authenticated grouped-daily
  endpoint that lists every actively-traded US stock on a given calendar date.
* :class:`~pairs.data_providers.yfinance.YFinanceProvider` -- thin wrapper that
  delegates to the existing :mod:`pairs.data.loader` pipeline so local dev keeps
  working without a Polygon key.
* :func:`~pairs.data_providers.factory.make_provider` -- environment-driven
  factory: returns the Polygon provider when ``POLYGON_API_KEY`` is set,
  otherwise the yfinance fallback.
* :class:`~pairs.data_providers.sp500_universe.SP500UniverseBuilder` -- approximates
  point-in-time S&P 500 membership by intersecting a snapshot of current
  constituents with Polygon's grouped-daily list at each as-of date, dropping
  symbols that were not yet trading.

The factory abstraction is what lets the Streamlit demo show *honest* backtests:
when a Polygon key is configured, the "S&P 500 PIT" universe option becomes
available and the pair selector draws only from names that were actually trading
on the as-of date.
"""

from __future__ import annotations

from pairs.data_providers.exceptions import (
    ProviderAuthError,
    ProviderDataError,
    ProviderError,
    ProviderRateLimitError,
)
from pairs.data_providers.factory import make_provider
from pairs.data_providers.polygon import PolygonProvider
from pairs.data_providers.sp500_universe import CURRENT_SP500, SP500UniverseBuilder
from pairs.data_providers.yfinance import YFinanceProvider

__all__ = [
    "CURRENT_SP500",
    "PolygonProvider",
    "ProviderAuthError",
    "ProviderDataError",
    "ProviderError",
    "ProviderRateLimitError",
    "SP500UniverseBuilder",
    "YFinanceProvider",
    "make_provider",
]
