"""The cost-decomposition columns must add up to the total cost drag."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs.backtest import backtest_pair
from pairs.backtest.accounting import PerShareCommission
from pairs.backtest.borrow import ConstantBorrow
from pairs.backtest.costs import CompositeCost
from pairs.backtest.slippage import HalfSpreadSlippage


def test_decomposition_sums_to_total_cost_drag(rng: np.random.Generator) -> None:
    n = 60
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    a = pd.Series(100.0 + np.cumsum(rng.standard_normal(n) * 0.2), index=index).clip(lower=1.0)
    b = pd.Series(50.0 + np.cumsum(rng.standard_normal(n) * 0.1), index=index).clip(lower=1.0)
    signal_vals = rng.choice([-1, 0, 1], size=n).astype(np.int8)
    signal = pd.Series(signal_vals, index=index, dtype="int8")

    cost = CompositeCost(
        slippage=HalfSpreadSlippage(spread_bps=8.0),
        commission=PerShareCommission(per_share=0.005, min_per_trade=1.0),
        borrow=ConstantBorrow(rate_bps_annual=50.0),
        name="mixed",
    )
    result = backtest_pair(a, b, signal, hedge_ratio=1.0, cost_model=cost, capital=100_000.0)

    decomp_total = result.cost_decomposition[
        ["commission", "slippage", "borrow", "dividend"]
    ].sum().sum()
    # Rebuild gross - net per bar; they must equal the decomposition row-wise.
    drag = (result.gross_returns - result.returns) * 100_000.0
    np.testing.assert_allclose(
        drag.to_numpy(),
        result.cost_decomposition.sum(axis=1).to_numpy(),
        atol=1e-9,
    )
    np.testing.assert_allclose(decomp_total, drag.sum(), atol=1e-6)
