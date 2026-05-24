"""Tests for ``pairs.data.loader``.

The yfinance adapter is monkeypatched throughout; no network IO occurs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairs._exceptions import InputError, InsufficientDataError
from pairs.data import loader as loader_mod


def _make_frame(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="B", tz="UTC", name="date", inclusive="left")
    cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    data = np.tile(np.linspace(100, 110, len(idx))[:, None], (1, len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture
def stub_download(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[str, ...], str, str]]:
    """Replace ``_batch_download`` with a recorder that returns a synthetic frame."""
    calls: list[tuple[tuple[str, ...], str, str]] = []

    def _fake(tickers: list[str], start: str, end: str) -> pd.DataFrame:
        calls.append((tuple(tickers), start, end))
        return _make_frame(tickers, start, end)

    monkeypatch.setattr(loader_mod, "_batch_download", _fake)
    return calls


def test_loader_returns_multiindex_columns(
    tmp_cache: Path,
    stub_download: list[tuple[tuple[str, ...], str, str]],
) -> None:
    df = loader_mod.load_prices(["AAA", "BBB"], "2020-01-02", "2020-02-01")
    assert isinstance(df.columns, pd.MultiIndex)
    assert {"AAA", "BBB"} <= set(df.columns.get_level_values(0))
    assert "Adj Close" in df.columns.get_level_values(1)
    assert len(stub_download) == 2


def test_loader_force_refresh_invalidates_cache(
    tmp_cache: Path,
    stub_download: list[tuple[tuple[str, ...], str, str]],
) -> None:
    loader_mod.load_prices(["AAA"], "2020-01-02", "2020-02-01")
    assert (tmp_cache / "prices" / "AAA.parquet").exists()
    first_n = len(stub_download)
    loader_mod.load_prices(["AAA"], "2020-01-02", "2020-02-01", force_refresh=True)
    # A fresh download must have been triggered.
    assert len(stub_download) == first_n + 1


def test_loader_empty_for_delisted_ticker(
    tmp_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _empty(tickers: list[str], start: str, end: str) -> pd.DataFrame:
        return pd.DataFrame(
            columns=pd.MultiIndex.from_product([tickers, ["Adj Close"]]),
        )

    monkeypatch.setattr(loader_mod, "_batch_download", _empty)
    with pytest.raises(InsufficientDataError):
        loader_mod.load_prices(["ZZZZ"], "2020-01-02", "2020-02-01")


def test_loader_raises_on_invalid_ticker_format(tmp_cache: Path) -> None:
    with pytest.raises(InputError):
        loader_mod.load_prices(["lower!case"], "2020-01-02", "2020-02-01")
    with pytest.raises(InputError):
        loader_mod.load_prices([], "2020-01-02", "2020-02-01")
    with pytest.raises(InputError):
        loader_mod.load_prices(["AAA"], "2020-02-01", "2020-01-02")


def test_loader_partial_refresh_only_fetches_tail(
    tmp_cache: Path,
    stub_download: list[tuple[tuple[str, ...], str, str]],
) -> None:
    loader_mod.load_prices(["AAA"], "2020-01-02", "2020-02-01")
    initial_calls = len(stub_download)
    # Asking for an overlapping but longer window should only fetch the missing
    # tail (not refetch the cached prefix).
    loader_mod.load_prices(["AAA"], "2020-01-02", "2020-03-01")
    assert len(stub_download) == initial_calls + 1
    _, tail_start, _ = stub_download[-1]
    # The newly-fetched start should be strictly after 2020-02-01 (the cached max).
    assert pd.Timestamp(tail_start) > pd.Timestamp("2020-01-30")
