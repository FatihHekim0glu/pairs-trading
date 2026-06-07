"""Provider factory with environment-driven fallback.

:func:`make_provider` is the single entry point used by the Streamlit demo and
any library consumer that wants the "best available" data source without
hardcoding which one. It picks :class:`~pairs.data_providers.polygon.PolygonProvider`
when ``POLYGON_API_KEY`` is set in the environment (or passed explicitly),
otherwise it returns :class:`~pairs.data_providers.yfinance.YFinanceProvider`.

The Streamlit sidebar uses this to decide whether the "S&P 500 PIT" universe
option should be offered -- that option requires the grouped-daily endpoint
which only the Polygon provider can serve.
"""

from __future__ import annotations

import logging
import os
from typing import Union

from pairs.data_providers.polygon import PolygonProvider
from pairs.data_providers.yfinance import YFinanceProvider

logger = logging.getLogger(__name__)

Provider = Union[PolygonProvider, YFinanceProvider]


def make_provider(api_key: str | None = None) -> Provider:
    """Return the most capable provider available given the current environment.

    Parameters
    ----------
    api_key
        Optional explicit Polygon key. When ``None``, falls back to the
        ``POLYGON_API_KEY`` environment variable.

    Returns
    -------
    PolygonProvider | YFinanceProvider
        Polygon when a key is configured (live or via env), yfinance otherwise.
    """
    resolved = (api_key if api_key is not None else os.environ.get("POLYGON_API_KEY", "")).strip()
    if resolved:
        logger.debug("POLYGON_API_KEY detected; using Polygon provider")
        return PolygonProvider(api_key=resolved)
    logger.info("POLYGON_API_KEY not set -- using yfinance fallback provider")
    return YFinanceProvider()
