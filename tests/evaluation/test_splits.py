"""Tests for the ``IsOosSplit`` primitive."""

from __future__ import annotations

import pandas as pd
import pytest

from pairs._exceptions import InputError
from pairs.evaluation import IsOosSplit


def test_from_dates_correct_partitions() -> None:
    prices = pd.DataFrame(
        {"x": range(200)},
        index=pd.date_range("2020-01-01", periods=200, freq="B"),
    )
    split = IsOosSplit.from_dates(prices, train_end="2020-06-01", oos_start="2020-06-15")
    assert split.train_index.max() <= pd.Timestamp("2020-06-01")
    assert split.test_index.min() >= pd.Timestamp("2020-06-15")
    assert split.train_index.intersection(split.test_index).empty


def test_embargo_index_disjoint_from_train_test() -> None:
    prices = pd.DataFrame(
        {"x": range(120)},
        index=pd.date_range("2021-01-01", periods=120, freq="D"),
    )
    split = IsOosSplit.from_dates(
        prices,
        train_end="2021-02-15",
        oos_start="2021-03-01",
        embargo_days=5,
    )
    assert split.embargo_index.intersection(split.test_index).empty
    # Embargoed indices are calendar-derived: they may sit between train and test
    # but must not overlap the test partition itself.
    assert (split.embargo_index > split.test_index.max()).all()


def test_from_dates_rejects_invalid_order() -> None:
    prices = pd.DataFrame(
        {"x": range(20)},
        index=pd.date_range("2020-01-01", periods=20, freq="D"),
    )
    with pytest.raises(InputError):
        IsOosSplit.from_dates(prices, train_end="2020-01-15", oos_start="2020-01-10")
