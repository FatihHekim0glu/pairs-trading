"""Tests for the sensitivity grid and break-even helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs._exceptions import InputError
from pairs.backtest import break_even_cost, sensitivity_grid
from pairs.spread import build_spread, fit_ou, zscore
from pairs.strategy import generate_signals


def test_grid_shape(synthetic_ou_prices: tuple[pd.Series, pd.Series, float]) -> None:
    prices_a, prices_b, hedge = synthetic_ou_prices
    spread = build_spread(prices_a, prices_b, beta=hedge, alpha=0.0)
    ou = fit_ou(spread)
    z = zscore(spread, use_ou=True, ou_result=ou)
    signal = generate_signals(z.fillna(0.0), z_entry=1.5, z_exit=0.5, z_stop=4.0)
    grid = sensitivity_grid(
        prices_a,
        prices_b,
        signal,
        hedge,
        cost_grid=[0.0, 5.0, 10.0],
        borrow_grid=[0.0, 25.0],
    )
    assert isinstance(grid, pd.DataFrame)
    assert len(grid) == 6
    assert {"cost_bps", "borrow_bps", "sharpe", "total_return", "n_trades"} <= set(grid.columns)


def test_grid_dict_form_equivalent(
    synthetic_ou_prices: tuple[pd.Series, pd.Series, float],
) -> None:
    prices_a, prices_b, hedge = synthetic_ou_prices
    spread = build_spread(prices_a, prices_b, beta=hedge, alpha=0.0)
    ou = fit_ou(spread)
    z = zscore(spread, use_ou=True, ou_result=ou)
    signal = generate_signals(z.fillna(0.0), z_entry=1.5, z_exit=0.5, z_stop=4.0)
    list_grid = sensitivity_grid(
        prices_a, prices_b, signal, hedge, cost_grid=[0.0, 5.0], borrow_grid=[0.0]
    )
    dict_grid = sensitivity_grid(
        prices_a, prices_b, signal, hedge, cost_grid={"bps": [0.0, 5.0]}, borrow_grid=[0.0]
    )
    pd.testing.assert_frame_equal(
        list_grid.reset_index(drop=True),
        dict_grid.reset_index(drop=True),
    )


def test_break_even_monotone_in_gross_sharpe() -> None:
    # Synthetic grid: at borrow=0, sharpe decreases as cost increases.
    grid = pd.DataFrame(
        {
            "cost_bps": [0.0, 5.0, 10.0, 20.0, 0.0, 5.0, 10.0, 20.0],
            "borrow_bps": [0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, 25.0],
            "sharpe": [2.0, 1.0, 0.0, -1.0, 1.5, 0.5, -0.5, -1.5],
            "total_return": [0.1] * 8,
            "n_trades": [5] * 8,
        }
    )
    be0 = break_even_cost(grid, borrow_bps=0.0)
    be25 = break_even_cost(grid, borrow_bps=25.0)
    # At borrow=0 the cross happens exactly at cost=10.
    assert be0 == pytest.approx(10.0)
    # At borrow=25 the cross is between cost=5 (sharpe=0.5) and cost=10 (sharpe=-0.5) -> 7.5.
    assert be25 == pytest.approx(7.5)
    # Higher borrow rate should produce a *lower* break-even cost (less remaining edge).
    assert be25 < be0


def test_break_even_handles_all_positive() -> None:
    grid = pd.DataFrame(
        {
            "cost_bps": [0.0, 5.0, 10.0],
            "borrow_bps": [0.0, 0.0, 0.0],
            "sharpe": [3.0, 2.0, 1.0],
            "total_return": [0.0, 0.0, 0.0],
            "n_trades": [1, 1, 1],
        }
    )
    assert np.isinf(break_even_cost(grid, borrow_bps=0.0))


def test_break_even_handles_all_negative() -> None:
    grid = pd.DataFrame(
        {
            "cost_bps": [0.0, 5.0],
            "borrow_bps": [0.0, 0.0],
            "sharpe": [-1.0, -2.0],
            "total_return": [0.0, 0.0],
            "n_trades": [1, 1],
        }
    )
    assert break_even_cost(grid, borrow_bps=0.0) == pytest.approx(0.0)


def test_break_even_rejects_unknown_borrow() -> None:
    grid = pd.DataFrame(
        {
            "cost_bps": [0.0, 5.0],
            "borrow_bps": [0.0, 0.0],
            "sharpe": [1.0, -1.0],
            "total_return": [0.0, 0.0],
            "n_trades": [1, 1],
        }
    )
    with pytest.raises(InputError):
        break_even_cost(grid, borrow_bps=999.0)


def test_sensitivity_rejects_empty_grids(
    synthetic_ou_prices: tuple[pd.Series, pd.Series, float],
) -> None:
    prices_a, prices_b, hedge = synthetic_ou_prices
    signal = pd.Series(np.zeros(len(prices_a), dtype=np.int8), index=prices_a.index, dtype="int8")
    with pytest.raises(InputError):
        sensitivity_grid(prices_a, prices_b, signal, hedge, cost_grid=[], borrow_grid=[0.0])
