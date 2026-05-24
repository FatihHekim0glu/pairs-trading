"""Shared fixtures for backtest-layer tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.backtest.accounting import FixedCommission
from pairs.backtest.borrow import ConstantBorrow
from pairs.backtest.costs import CompositeCost
from pairs.backtest.slippage import ConstantBpsSlippage


@pytest.fixture
def zero_cost_model() -> CompositeCost:
    """Return a cost model that charges nothing."""
    return CompositeCost(
        slippage=ConstantBpsSlippage(bps=0.0),
        commission=FixedCommission(per_trade=0.0),
        borrow=ConstantBorrow(rate_bps_annual=0.0),
        name="zero",
    )


@pytest.fixture
def bps_cost_model() -> CompositeCost:
    """Return a 5 bps slippage, zero commission, zero borrow model."""
    return CompositeCost(
        slippage=ConstantBpsSlippage(bps=5.0),
        commission=FixedCommission(per_trade=0.0),
        borrow=ConstantBorrow(rate_bps_annual=0.0),
        name="bps5",
    )


def _synthetic_ou_path(
    n: int,
    *,
    rng: np.random.Generator,
    theta: float = 0.1,
    mu: float = 0.0,
    sigma: float = 0.5,
) -> np.ndarray:
    """Simulate a discrete OU path of length ``n``."""
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sigma * rng.standard_normal()
    return x


@pytest.fixture
def synthetic_ou_prices(rng: np.random.Generator) -> tuple[pd.Series, pd.Series, float]:
    """Two cointegrated price series sharing a stationary spread.

    Returns ``(prices_a, prices_b, hedge_ratio)`` where the spread
    ``a - hedge_ratio * b`` follows a mean-reverting OU process.
    """
    n = 500
    index = pd.RangeIndex(n)
    # Common stochastic trend (random walk) and stationary spread.
    common = np.cumsum(rng.standard_normal(n) * 0.1) + 50.0
    spread = _synthetic_ou_path(n, rng=rng, theta=0.15, mu=0.0, sigma=0.4)
    hedge = 1.0
    prices_b = pd.Series(common, index=index, name="b")
    prices_a = pd.Series(common * hedge + spread, index=index, name="a")
    # Guard against negative prices (extremely unlikely with these parameters but keep safe).
    prices_a = prices_a.clip(lower=1.0)
    prices_b = prices_b.clip(lower=1.0)
    return prices_a, prices_b, hedge


@pytest.fixture
def flat_prices() -> tuple[pd.Series, pd.Series]:
    """Two constant-price series; any strategy must produce zero P&L."""
    n = 100
    index = pd.RangeIndex(n)
    return (
        pd.Series(np.full(n, 100.0), index=index, name="a"),
        pd.Series(np.full(n, 100.0), index=index, name="b"),
    )


@pytest.fixture
def cointegrated_prices(rng: np.random.Generator) -> tuple[pd.Series, pd.Series, float]:
    """Two cointegrated series with a slightly stronger mean-reversion."""
    n = 400
    index = pd.RangeIndex(n)
    common = np.cumsum(rng.standard_normal(n) * 0.05) + 100.0
    spread = _synthetic_ou_path(n, rng=rng, theta=0.2, mu=0.0, sigma=0.3)
    hedge = 1.0
    a = common + spread
    b = common
    return (
        pd.Series(a, index=index, name="a"),
        pd.Series(b, index=index, name="b"),
        hedge,
    )
