"""Smoke tests and property checks on the backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pairs._exceptions import InputError
from pairs.backtest import backtest_pair
from pairs.backtest.accounting import FixedCommission
from pairs.backtest.borrow import ConstantBorrow
from pairs.backtest.costs import CompositeCost
from pairs.backtest.slippage import ConstantBpsSlippage
from pairs.spread import build_spread, fit_ou, zscore
from pairs.strategy import generate_signals


def _make_cost(bps: float) -> CompositeCost:
    return CompositeCost(
        slippage=ConstantBpsSlippage(bps=bps),
        commission=FixedCommission(per_trade=0.0),
        borrow=ConstantBorrow(rate_bps_annual=0.0),
        name=f"bps{bps}",
    )


def test_zero_cost_synthetic_positive_sharpe(
    synthetic_ou_prices: tuple[pd.Series, pd.Series, float],
    zero_cost_model: CompositeCost,
) -> None:
    prices_a, prices_b, hedge = synthetic_ou_prices
    spread = build_spread(prices_a, prices_b, beta=hedge, alpha=0.0)
    ou = fit_ou(spread)
    z = zscore(spread, use_ou=True, ou_result=ou)
    signal = generate_signals(z.fillna(0.0), z_entry=1.5, z_exit=0.5, z_stop=4.0)
    result = backtest_pair(
        prices_a,
        prices_b,
        signal,
        hedge_ratio=hedge,
        cost_model=zero_cost_model,
        capital=1.0,
    )
    # The OU spread is mean-reverting by construction, so with no costs an
    # entry-at-2sigma / exit-at-0.5sigma rule should produce non-negative gross PnL.
    assert result.metrics["total_pnl_gross"] >= 0.0
    # With zero costs the gross and net should match exactly.
    np.testing.assert_allclose(
        result.returns.to_numpy(),
        result.gross_returns.to_numpy(),
        atol=1e-12,
    )


def test_flat_z_zero_trades_zero_pnl(flat_prices: tuple[pd.Series, pd.Series]) -> None:
    a, b = flat_prices
    signal = pd.Series(np.zeros(len(a), dtype=np.int8), index=a.index, dtype="int8")
    cost = _make_cost(0.0)
    result = backtest_pair(a, b, signal, hedge_ratio=1.0, cost_model=cost)
    assert result.metrics["n_trades"] == 0
    assert float(result.equity.iloc[-1]) == pytest.approx(1.0)
    assert float(result.metrics["total_pnl_net"]) == pytest.approx(0.0)


def test_cost_monotonicity_basic(
    synthetic_ou_prices: tuple[pd.Series, pd.Series, float],
) -> None:
    prices_a, prices_b, hedge = synthetic_ou_prices
    spread = build_spread(prices_a, prices_b, beta=hedge, alpha=0.0)
    ou = fit_ou(spread)
    z = zscore(spread, use_ou=True, ou_result=ou)
    signal = generate_signals(z.fillna(0.0), z_entry=1.5, z_exit=0.5, z_stop=4.0)
    low = backtest_pair(
        prices_a,
        prices_b,
        signal,
        hedge_ratio=hedge,
        cost_model=_make_cost(1.0),
    )
    high = backtest_pair(
        prices_a,
        prices_b,
        signal,
        hedge_ratio=hedge,
        cost_model=_make_cost(2.0),
    )
    # Higher costs cannot leave us with a *better* net P&L.
    assert high.metrics["total_pnl_net"] <= low.metrics["total_pnl_net"] + 1e-9


@given(
    bps=st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=1_000),
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_cost_monotonicity_property(bps: float, seed: int) -> None:
    """Doubling slippage cannot improve total net P&L."""
    rng = np.random.default_rng(seed)
    n = 80
    index = pd.RangeIndex(n)
    common = 100.0 + np.cumsum(rng.standard_normal(n) * 0.1)
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] + 0.15 * (0.0 - spread[i - 1]) + 0.4 * rng.standard_normal()
    prices_a = pd.Series(common + spread, index=index).clip(lower=1.0)
    prices_b = pd.Series(common, index=index).clip(lower=1.0)
    signal_vals = rng.choice([-1, 0, 1], size=n).astype(np.int8)
    signal = pd.Series(signal_vals, index=index, dtype="int8")
    low = backtest_pair(prices_a, prices_b, signal, hedge_ratio=1.0, cost_model=_make_cost(bps))
    high = backtest_pair(
        prices_a, prices_b, signal, hedge_ratio=1.0, cost_model=_make_cost(2.0 * bps)
    )
    assert high.metrics["total_pnl_net"] <= low.metrics["total_pnl_net"] + 1e-9


def test_engine_input_validation() -> None:
    n = 5
    a = pd.Series(np.ones(n), index=pd.RangeIndex(n))
    b = pd.Series(np.ones(n), index=pd.RangeIndex(n))
    signal = pd.Series(np.zeros(n, dtype=np.int8), index=pd.RangeIndex(n))
    cost = _make_cost(0.0)
    # Mismatched indices.
    with pytest.raises(InputError):
        backtest_pair(
            a,
            pd.Series(np.ones(n + 1), index=pd.RangeIndex(n + 1)),
            signal,
            hedge_ratio=1.0,
            cost_model=cost,
        )
    # Too short.
    with pytest.raises(InputError):
        backtest_pair(
            pd.Series([1.0]),
            pd.Series([1.0]),
            pd.Series([0], dtype="int8"),
            hedge_ratio=1.0,
            cost_model=cost,
        )
    # Negative capital.
    with pytest.raises(InputError):
        backtest_pair(a, b, signal, hedge_ratio=1.0, cost_model=cost, capital=-1.0)


def test_engine_with_datetime_index(rng: np.random.Generator) -> None:
    n = 40
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    prices_a = pd.Series(100.0 + np.cumsum(rng.standard_normal(n) * 0.5), index=index)
    prices_b = pd.Series(100.0 + np.cumsum(rng.standard_normal(n) * 0.5), index=index)
    signal = pd.Series(rng.choice([-1, 0, 1], size=n).astype(np.int8), index=index, dtype="int8")
    result = backtest_pair(
        prices_a,
        prices_b,
        signal,
        hedge_ratio=1.0,
        cost_model=_make_cost(0.0),
    )
    assert len(result.equity) == n
    assert isinstance(result.equity.index, pd.DatetimeIndex)
