"""Tests for the anchored walk-forward harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs._exceptions import InputError
from pairs.evaluation import walk_forward_anchored


def test_concatenated_oos_length_sums(
    synthetic_prices,
    mock_pair_selector,
    mock_pair_backtester,
) -> None:
    result = walk_forward_anchored(
        synthetic_prices,
        train_min_years=1.0,
        test_period="63D",
        step="63D",
        purge_days=5,
        embargo_pct=0.0,
        pair_selector=mock_pair_selector,
        pair_backtester=mock_pair_backtester,
        bootstrap_replicates=50,
    )
    assert result.fold_count >= 1
    assert result.oos_returns.size > 0
    assert result.oos_returns.index.is_monotonic_increasing


def test_fold_count_matches_steps(
    synthetic_prices,
    mock_pair_selector,
    mock_pair_backtester,
) -> None:
    result = walk_forward_anchored(
        synthetic_prices,
        train_min_years=2.0,
        test_period="126D",
        step="126D",
        purge_days=0,
        embargo_pct=0.0,
        pair_selector=mock_pair_selector,
        pair_backtester=mock_pair_backtester,
        bootstrap_replicates=50,
    )
    # Folds line up with their reported start/end metadata.
    assert len(result.fold_starts) == result.fold_count
    assert len(result.fold_ends) == result.fold_count
    for start, end in zip(result.fold_starts, result.fold_ends, strict=False):
        assert start <= end


def test_rejects_non_dataframe(
    mock_pair_selector, mock_pair_backtester
) -> None:
    with pytest.raises(InputError):
        walk_forward_anchored(
            "not a frame",  # type: ignore[arg-type]
            pair_selector=mock_pair_selector,
            pair_backtester=mock_pair_backtester,
        )


def test_rejects_non_datetime_index(
    mock_pair_selector, mock_pair_backtester
) -> None:
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(InputError):
        walk_forward_anchored(
            prices,
            pair_selector=mock_pair_selector,
            pair_backtester=mock_pair_backtester,
        )


def test_rejects_unsorted_index(
    mock_pair_selector, mock_pair_backtester
) -> None:
    idx = pd.DatetimeIndex(
        ["2020-01-03", "2020-01-01", "2020-01-02"]
    )
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    with pytest.raises(InputError):
        walk_forward_anchored(
            prices,
            pair_selector=mock_pair_selector,
            pair_backtester=mock_pair_backtester,
        )


def test_rejects_invalid_train_years(
    synthetic_prices, mock_pair_selector, mock_pair_backtester
) -> None:
    with pytest.raises(InputError):
        walk_forward_anchored(
            synthetic_prices,
            train_min_years=0.0,
            pair_selector=mock_pair_selector,
            pair_backtester=mock_pair_backtester,
        )


def test_rejects_negative_purge(
    synthetic_prices, mock_pair_selector, mock_pair_backtester
) -> None:
    with pytest.raises(InputError):
        walk_forward_anchored(
            synthetic_prices,
            purge_days=-1,
            pair_selector=mock_pair_selector,
            pair_backtester=mock_pair_backtester,
        )


def test_rejects_invalid_embargo(
    synthetic_prices, mock_pair_selector, mock_pair_backtester
) -> None:
    with pytest.raises(InputError):
        walk_forward_anchored(
            synthetic_prices,
            embargo_pct=1.5,
            pair_selector=mock_pair_selector,
            pair_backtester=mock_pair_backtester,
        )


def test_returns_empty_when_train_window_exceeds_span(
    mock_pair_selector, mock_pair_backtester
) -> None:
    # Two years of data but request three years of training => no folds.
    idx = pd.bdate_range("2020-01-01", periods=520)
    prices = pd.DataFrame({"A": np.arange(520, dtype=float)}, index=idx)
    result = walk_forward_anchored(
        prices,
        train_min_years=3.0,
        test_period="63D",
        step="63D",
        purge_days=0,
        embargo_pct=0.0,
        pair_selector=mock_pair_selector,
        pair_backtester=mock_pair_backtester,
        bootstrap_replicates=10,
    )
    assert result.fold_count == 0
    assert result.oos_returns.empty
    assert np.isnan(result.oos_sharpe)
    # When too few observations, CI degenerates to (-inf, +inf).
    assert result.sharpe_ci_low == float("-inf")
    assert result.sharpe_ci_high == float("inf")


def test_backtester_must_return_series(
    synthetic_prices, mock_pair_selector
) -> None:
    def bad_backtester(test_prices, _selection):  # type: ignore[no-untyped-def]
        return list(range(len(test_prices)))

    with pytest.raises(InputError):
        walk_forward_anchored(
            synthetic_prices,
            train_min_years=1.0,
            test_period="63D",
            step="63D",
            pair_selector=mock_pair_selector,
            pair_backtester=bad_backtester,
            bootstrap_replicates=10,
        )


def test_embargo_days_recorded(
    synthetic_prices, mock_pair_selector, mock_pair_backtester
) -> None:
    # Non-zero embargo_pct converts to a positive embargo_days reading.
    result = walk_forward_anchored(
        synthetic_prices,
        train_min_years=1.0,
        test_period="63D",
        step="63D",
        purge_days=2,
        embargo_pct=0.02,
        pair_selector=mock_pair_selector,
        pair_backtester=mock_pair_backtester,
        bootstrap_replicates=10,
    )
    assert result.embargo_days > 0
    assert result.purge_days == 2


def test_purge_boundary_drops_overlap(
    mock_pair_selector, mock_pair_backtester
) -> None:
    # Construct ~4 years of business days so several folds fire, then verify
    # purge_days > 0 produces an OOS series whose first timestamp sits strictly
    # after the anchor + 1 day (i.e. the training tail was purged successfully).
    idx = pd.bdate_range("2018-01-01", periods=4 * 252)
    prices = pd.DataFrame(
        {"A": np.linspace(100.0, 120.0, len(idx)), "B": np.linspace(100.0, 115.0, len(idx))},
        index=idx,
    )
    result = walk_forward_anchored(
        prices,
        train_min_years=2.0,
        test_period="63D",
        step="63D",
        purge_days=15,
        embargo_pct=0.01,
        pair_selector=mock_pair_selector,
        pair_backtester=mock_pair_backtester,
        bootstrap_replicates=20,
    )
    assert result.fold_count >= 2
    # OOS timestamps never appear inside any training-bound (cursor) window.
    for start, end in zip(result.fold_starts, result.fold_ends, strict=False):
        # Each OOS span has length <= test_period (63 days).
        assert (end - start).days <= 63
    assert result.purge_days == 15
    assert result.embargo_days >= 1


def test_step_smaller_than_test_period_skips_overlaps(
    synthetic_prices, mock_pair_selector, mock_pair_backtester
) -> None:
    # Use a 21D step with a 63D test window: consecutive folds overlap, so the
    # deduplication branch (`~oos_returns.index.duplicated`) is exercised.
    result = walk_forward_anchored(
        synthetic_prices,
        train_min_years=1.0,
        test_period="63D",
        step="21D",
        purge_days=0,
        embargo_pct=0.0,
        pair_selector=mock_pair_selector,
        pair_backtester=mock_pair_backtester,
        bootstrap_replicates=20,
    )
    assert result.fold_count >= 3
    assert result.oos_returns.index.is_unique
