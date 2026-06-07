"""Unit tests for :mod:`pairs.data_providers.sp500_universe`."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from pairs.data_providers.sp500_universe import CURRENT_SP500, SP500UniverseBuilder


pytestmark = pytest.mark.unit


def _grouped_with(tickers: list[str]) -> pd.DataFrame:
    """Return a fake grouped-daily DataFrame indexed by ``tickers``."""
    df = pd.DataFrame(
        {
            "Open": [1.0] * len(tickers),
            "High": [1.0] * len(tickers),
            "Low": [1.0] * len(tickers),
            "Close": [1.0] * len(tickers),
            "Volume": [1.0] * len(tickers),
        },
        index=pd.Index(tickers, name="ticker"),
    )
    return df


def test_current_sp500_is_a_nonempty_tuple_of_uppercase_tickers() -> None:
    """Sanity-check the static constituent table."""
    assert isinstance(CURRENT_SP500, tuple)
    assert len(CURRENT_SP500) > 400  # S&P 500 with class shares is ~503
    assert all(isinstance(t, str) and t.isupper() for t in CURRENT_SP500)


def test_membership_as_of_intersects_with_grouped_daily() -> None:
    """A ticker must appear iff it is in CURRENT_SP500 AND traded on as_of."""
    provider = MagicMock()
    # Active that day: 2 S&P names + 1 non-S&P name; non-S&P must be filtered out.
    provider.get_grouped_daily.return_value = _grouped_with(["AAPL", "MSFT", "ZZZZ"])
    builder = SP500UniverseBuilder(provider=provider)
    members = builder.get_membership_as_of(date(2024, 1, 2))
    assert "AAPL" in members
    assert "MSFT" in members
    assert "ZZZZ" not in members


def test_membership_as_of_drops_sp500_names_not_trading_that_day() -> None:
    """If only AAPL traded, MSFT must NOT appear even though it is in CURRENT_SP500."""
    provider = MagicMock()
    provider.get_grouped_daily.return_value = _grouped_with(["AAPL"])
    builder = SP500UniverseBuilder(provider=provider)
    members = builder.get_membership_as_of(date(2024, 1, 2))
    assert members == ["AAPL"]


def test_membership_as_of_empty_grouped_daily_returns_empty_list() -> None:
    """Holidays / weekends produce an empty grouped-daily; result must be []."""
    provider = MagicMock()
    provider.get_grouped_daily.return_value = pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"]
    )
    builder = SP500UniverseBuilder(provider=provider)
    assert builder.get_membership_as_of(date(2024, 1, 6)) == []


def test_membership_window_uses_month_end_anchors_by_default() -> None:
    """``ME`` is the default freq; one call per month-end is expected."""
    provider = MagicMock()
    provider.get_grouped_daily.return_value = _grouped_with(["AAPL", "MSFT"])
    builder = SP500UniverseBuilder(provider=provider)
    window = builder.get_membership_window(date(2024, 1, 1), date(2024, 3, 31))
    # Jan-end, Feb-end, Mar-end => 3 calls.
    assert provider.get_grouped_daily.call_count == 3
    assert all(set(v) == {"AAPL", "MSFT"} for v in window.values())


def test_membership_window_rejects_inverted_range() -> None:
    """``start > end`` is a ValueError."""
    provider = MagicMock()
    builder = SP500UniverseBuilder(provider=provider)
    with pytest.raises(ValueError):
        builder.get_membership_window(date(2024, 3, 31), date(2024, 1, 1))


def test_membership_window_falls_back_to_endpoints_for_narrow_range() -> None:
    """When no month-end falls inside ``[start, end]``, the builder must still emit
    *something* -- we anchor at start and end so the dict is never empty."""
    provider = MagicMock()
    provider.get_grouped_daily.return_value = _grouped_with(["AAPL"])
    builder = SP500UniverseBuilder(provider=provider)
    window = builder.get_membership_window(date(2024, 6, 3), date(2024, 6, 4))
    assert len(window) >= 1
    assert all(v == ["AAPL"] for v in window.values())
