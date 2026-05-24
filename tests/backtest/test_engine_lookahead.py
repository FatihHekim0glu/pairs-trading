"""Lookahead-bias guards: the chokepoint enforced via Hypothesis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pairs.backtest import backtest_pair
from pairs.backtest.accounting import FixedCommission
from pairs.backtest.borrow import ConstantBorrow
from pairs.backtest.costs import CompositeCost
from pairs.backtest.slippage import ConstantBpsSlippage


def _zero_cost() -> CompositeCost:
    return CompositeCost(
        slippage=ConstantBpsSlippage(bps=0.0),
        commission=FixedCommission(per_trade=0.0),
        borrow=ConstantBorrow(rate_bps_annual=0.0),
        name="zero",
    )


@given(
    prices=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=5, max_value=80),
        elements=st.floats(min_value=10.0, max_value=200.0, allow_nan=False),
    ),
    signal_seed=st.integers(min_value=0, max_value=10_000),
)
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_position_equals_signal_shift_1(prices: np.ndarray, signal_seed: int) -> None:
    """THE CHOKEPOINT: position[t] must equal signal[t-1]."""
    n = prices.size
    rng = np.random.default_rng(signal_seed)
    signal_values = rng.choice([-1, 0, 1], size=n).astype(np.int8)
    index = pd.RangeIndex(n)
    prices_a = pd.Series(prices, index=index)
    prices_b = pd.Series(prices[::-1].copy(), index=index)
    signal = pd.Series(signal_values, index=index, dtype="int8")

    result = backtest_pair(prices_a, prices_b, signal, hedge_ratio=1.0, cost_model=_zero_cost())
    pos = result.positions["position"].to_numpy().astype(int)
    expected = np.concatenate(([0], signal_values[:-1].astype(int)))
    np.testing.assert_array_equal(pos, expected)


@given(
    prices=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=10, max_value=40),
        elements=st.floats(min_value=10.0, max_value=100.0, allow_nan=False),
    ),
    shift_k=st.integers(min_value=1, max_value=3),
)
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_shift_invariance(prices: np.ndarray, shift_k: int) -> None:
    """Shifting the z-score by ``k`` bars shifts every P&L bar by the same ``k``."""
    n = prices.size
    assume(n > shift_k + 2)
    rng = np.random.default_rng(int(prices.sum() * 1000) % (2**32))
    signal_values = rng.choice([-1, 0, 1], size=n).astype(np.int8)
    index = pd.RangeIndex(n)
    prices_a = pd.Series(prices, index=index)
    prices_b = pd.Series(prices * 0.5 + 50.0, index=index)
    signal = pd.Series(signal_values, index=index, dtype="int8")
    shifted_signal = signal.shift(shift_k).fillna(0).astype("int8")

    base = backtest_pair(prices_a, prices_b, signal, hedge_ratio=1.0, cost_model=_zero_cost())
    delayed = backtest_pair(
        prices_a, prices_b, shifted_signal, hedge_ratio=1.0, cost_model=_zero_cost()
    )

    # Position series should match exactly when shifted by k.
    pos_base = base.positions["position"].to_numpy().astype(int)
    pos_delayed = delayed.positions["position"].to_numpy().astype(int)
    np.testing.assert_array_equal(pos_base[: n - shift_k], pos_delayed[shift_k:])


def test_pit_stability(rng: np.random.Generator) -> None:
    """``backtest(prices[:t])`` and ``backtest(prices[:t+1])`` agree at time ``t``."""
    n = 60
    index = pd.RangeIndex(n)
    prices_a = pd.Series(100.0 + np.cumsum(rng.standard_normal(n) * 0.5), index=index)
    prices_b = pd.Series(100.0 + np.cumsum(rng.standard_normal(n) * 0.5), index=index)
    signal_vals = rng.choice([-1, 0, 1], size=n).astype(np.int8)
    signal = pd.Series(signal_vals, index=index, dtype="int8")

    short = backtest_pair(
        prices_a.iloc[:30],
        prices_b.iloc[:30],
        signal.iloc[:30],
        hedge_ratio=1.0,
        cost_model=_zero_cost(),
    )
    longer = backtest_pair(
        prices_a.iloc[:31],
        prices_b.iloc[:31],
        signal.iloc[:31],
        hedge_ratio=1.0,
        cost_model=_zero_cost(),
    )
    # Bars 0..29 in the longer run must equal the entire short run.
    np.testing.assert_allclose(
        short.returns.to_numpy(),
        longer.returns.iloc[:30].to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        short.positions["position"].to_numpy(),
        longer.positions["position"].iloc[:30].to_numpy(),
    )
