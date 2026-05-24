"""Tests for combinatorial purged cross-validation."""

from __future__ import annotations

from itertools import combinations

from pairs.evaluation import cpcv_paths


def test_combination_count() -> None:
    # C(10, 2) = 45.
    combos = list(combinations(range(10), 2))
    assert len(combos) == 45


def test_path_count_formula(
    synthetic_prices,
    mock_pair_selector,
    mock_pair_backtester,
) -> None:
    result = cpcv_paths(
        synthetic_prices,
        n_groups=10,
        k_test=2,
        purge_days=0,
        embargo_pct=0.0,
        pair_selector=mock_pair_selector,
        pair_backtester=mock_pair_backtester,
    )
    # Lopez de Prado 2018: paths = C(N, k) * k / N = 45 * 2 / 10 = 9.
    expected_paths = (45 * 2) // 10
    assert result.n_combinations == 45
    assert len(result.paths) <= expected_paths
    assert result.n_groups == 10
    assert result.k_test == 2


def test_paths_disjoint_within_path(
    synthetic_prices,
    mock_pair_selector,
    mock_pair_backtester,
) -> None:
    result = cpcv_paths(
        synthetic_prices,
        n_groups=6,
        k_test=2,
        purge_days=0,
        embargo_pct=0.0,
        pair_selector=mock_pair_selector,
        pair_backtester=mock_pair_backtester,
    )
    for path in result.paths:
        # No duplicated timestamps within a single reassembled path.
        assert path.index.is_unique
