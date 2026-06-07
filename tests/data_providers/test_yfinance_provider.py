"""Unit tests for :mod:`pairs.data_providers.yfinance`."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from pairs.data_providers.exceptions import ProviderDataError, ProviderError
from pairs.data_providers.yfinance import YFinanceProvider


pytestmark = pytest.mark.unit


def test_yfinance_grouped_daily_unsupported() -> None:
    """Grouped-daily has no yfinance equivalent and must raise ProviderError."""
    provider = YFinanceProvider()
    with pytest.raises(ProviderError):
        provider.get_grouped_daily(date(2024, 1, 2))


def test_yfinance_ticker_meta_unsupported() -> None:
    """Ticker metadata also has no yfinance equivalent."""
    provider = YFinanceProvider()
    with pytest.raises(ProviderError):
        provider.get_ticker_meta("AAPL")


def test_yfinance_get_eod_rejects_inverted_range() -> None:
    """``start > end`` is a ValueError before any loader work."""
    provider = YFinanceProvider()
    with pytest.raises(ValueError):
        provider.get_eod("AAPL", date(2024, 1, 5), date(2024, 1, 1))


def test_yfinance_get_eod_delegates_to_load_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful loader call must yield TitleCase OHLCV columns."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], name="Date"
    )
    frame = pd.DataFrame(
        {
            ("AAPL", "Open"): [100.0, 101.0],
            ("AAPL", "High"): [102.0, 103.0],
            ("AAPL", "Low"): [99.0, 100.0],
            ("AAPL", "Close"): [101.0, 102.0],
            ("AAPL", "Volume"): [1_000_000, 1_100_000],
        },
        index=idx,
    )

    def fake_load_prices(*_a: object, **_k: object) -> pd.DataFrame:
        return frame

    import pairs.data.loader as loader_mod

    monkeypatch.setattr(loader_mod, "load_prices", fake_load_prices, raising=True)
    provider = YFinanceProvider()
    out = provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 3))
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out["Close"].iloc[-1] == pytest.approx(102.0)


def test_yfinance_get_eod_wraps_loader_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any loader exception must surface as ProviderDataError."""

    def boom(*_a: object, **_k: object) -> pd.DataFrame:
        raise RuntimeError("loader exploded")

    import pairs.data.loader as loader_mod

    monkeypatch.setattr(loader_mod, "load_prices", boom, raising=True)
    provider = YFinanceProvider()
    with pytest.raises(ProviderDataError):
        provider.get_eod("AAPL", date(2024, 1, 2), date(2024, 1, 3))
