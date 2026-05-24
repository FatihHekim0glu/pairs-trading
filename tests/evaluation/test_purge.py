"""Tests for purge and embargo helpers."""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from pairs.evaluation._purge import embargo_indices, purge_indices


@given(
    label=st.integers(min_value=0, max_value=30),
    train_n=st.integers(min_value=5, max_value=120),
    test_n=st.integers(min_value=1, max_value=30),
    gap=st.integers(min_value=-10, max_value=60),
)
@settings(max_examples=40, deadline=None)
def test_purge_removes_label_overlap(label: int, train_n: int, test_n: int, gap: int) -> None:
    train_idx = pd.date_range("2020-01-01", periods=train_n, freq="B")
    test_start = train_idx[-1] + pd.Timedelta(days=max(gap, 1))
    test_idx = pd.date_range(test_start, periods=test_n, freq="B")
    kept = purge_indices(train_idx, test_idx, label)
    # Any kept training timestamp's label window must not overlap the test span.
    horizon = pd.Timedelta(days=label)
    for ts in kept:
        end = ts + horizon
        assert (end < test_idx.min()) or (ts > test_idx.max())


def test_embargo_exact_size() -> None:
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    test = idx[40:50]
    embargo = embargo_indices(idx, test, embargo_days=5)
    # Embargo covers (test_end, test_end+5]
    assert embargo.min() == test.max() + pd.Timedelta(days=1)
    assert embargo.max() == test.max() + pd.Timedelta(days=5)
    assert len(embargo) == 5
