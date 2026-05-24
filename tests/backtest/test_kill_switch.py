"""Blacklist / kill-switch tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

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


def test_blacklist_force_exit() -> None:
    n = 10
    index = pd.RangeIndex(n)
    a = pd.Series(100.0 + np.arange(n) * 0.1, index=index)
    b = pd.Series(100.0 - np.arange(n) * 0.1, index=index)
    signal = pd.Series(np.full(n, 1, dtype=np.int8), index=index, dtype="int8")
    blacklist = pd.Series(False, index=index)
    blacklist.iloc[5:] = True
    result = backtest_pair(
        a,
        b,
        signal,
        hedge_ratio=1.0,
        cost_model=_zero_cost(),
        blacklist=blacklist,
    )
    pos = result.positions["position"].to_numpy()
    # Position must be forced to zero from bar 6 onward (lagged blacklist).
    assert (pos[6:] == 0).all()
    # The trade ledger should record at least one entry with reason 'blacklist'.
    if not result.trades.empty:
        assert (result.trades["exit_reason"] == "blacklist").any()


def test_blacklist_no_reentry_same_bar() -> None:
    n = 8
    index = pd.RangeIndex(n)
    a = pd.Series(np.full(n, 100.0), index=index)
    b = pd.Series(np.full(n, 100.0), index=index)
    # Signal oscillates so the engine would re-enter on every bar absent the kill-switch.
    signal = pd.Series([1, -1, 1, -1, 1, -1, 1, -1], index=index, dtype="int8")
    blacklist = pd.Series([False, False, True, True, True, True, True, True], index=index)
    result = backtest_pair(
        a,
        b,
        signal,
        hedge_ratio=1.0,
        cost_model=_zero_cost(),
        blacklist=blacklist,
    )
    pos = result.positions["position"].to_numpy()
    # blacklist becomes effective from bar 3 onward via the shift(1) lag.
    assert (pos[3:] == 0).all()
