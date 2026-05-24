"""Tests for cost-model classes."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from pairs._exceptions import InputError
from pairs.backtest.accounting import FixedCommission, PerShareCommission, two_leg_sizing
from pairs.backtest.borrow import ConstantBorrow, ProfileBorrow
from pairs.backtest.slippage import (
    AlmgrenChrissSlippage,
    ConstantBpsSlippage,
    HalfSpreadSlippage,
)


def test_constant_bps_slippage_value() -> None:
    s = ConstantBpsSlippage(bps=10.0)
    # 100 shares * $50 * 10bps = 100 * 50 * 1e-3 = 50.0
    assert s.slippage(price=50.0, qty=100.0, side=1, adv=None) == pytest.approx(5.0)


def test_constant_bps_rejects_negative() -> None:
    with pytest.raises(InputError):
        ConstantBpsSlippage(bps=-1.0)


def test_half_spread_slippage_scalar() -> None:
    s = HalfSpreadSlippage(spread_bps=20.0)
    # 0.5 * 100 * 50 * 20bps = 5.0
    assert s.slippage(price=50.0, qty=100.0, side=-1, adv=None) == pytest.approx(5.0)


def test_half_spread_slippage_series() -> None:
    spread = pd.Series([10.0, 20.0], index=pd.RangeIndex(2))
    s = HalfSpreadSlippage(spread_bps=spread)
    s.set_index_value(0)
    assert s.slippage(price=50.0, qty=100.0, side=1, adv=None) == pytest.approx(2.5)
    s.set_index_value(1)
    assert s.slippage(price=50.0, qty=100.0, side=1, adv=None) == pytest.approx(5.0)


def test_half_spread_requires_index_for_series() -> None:
    spread = pd.Series([10.0], index=pd.RangeIndex(1))
    s = HalfSpreadSlippage(spread_bps=spread)
    with pytest.raises(InputError):
        s.slippage(price=50.0, qty=100.0, side=1, adv=None)


def test_almgren_chriss_sqrt_impact() -> None:
    s = AlmgrenChrissSlippage(eta=10.0, gamma=0.0)
    # impact bps = 10 * sqrt(qty/adv); qty=100, adv=10_000 -> impact_bps = 10*0.1 = 1.0
    # cost = 100 * $50 * 1bp = $0.5
    cost = s.slippage(price=50.0, qty=100.0, side=1, adv=10_000.0)
    assert cost == pytest.approx(0.5)
    # Doubling qty must increase per-share impact by sqrt(2).
    cost_x2 = s.slippage(price=50.0, qty=200.0, side=1, adv=10_000.0)
    per_share_1 = cost / 100.0
    per_share_2 = cost_x2 / 200.0
    assert per_share_2 / per_share_1 == pytest.approx(math.sqrt(2.0))


def test_almgren_chriss_zero_when_no_adv() -> None:
    s = AlmgrenChrissSlippage(eta=10.0)
    assert s.slippage(price=50.0, qty=100.0, side=1, adv=None) == 0.0
    assert s.slippage(price=50.0, qty=100.0, side=1, adv=0.0) == 0.0


def test_fixed_commission() -> None:
    c = FixedCommission(per_trade=1.5)
    assert c.commission(price=10.0, qty=100.0, side=1) == 1.5
    assert c.commission(price=10.0, qty=0.0, side=1) == 0.0


def test_per_share_commission_with_min() -> None:
    c = PerShareCommission(per_share=0.005, min_per_trade=1.0)
    # 10 shares * $0.005 = $0.05 < $1 floor -> floor wins.
    assert c.commission(price=10.0, qty=10.0, side=1) == pytest.approx(1.0)
    # 1000 shares * $0.005 = $5 > floor.
    assert c.commission(price=10.0, qty=1000.0, side=-1) == pytest.approx(5.0)
    assert c.commission(price=10.0, qty=0.0, side=1) == 0.0


def test_constant_borrow_daily_accrual() -> None:
    # rate=100bps, notional=$100k, 1/252 dt_days -> 100_000 * 0.01 * (1/252) / 365? Spec says
    # "1/252 dt -> ~$3.97"; using day count 365 -> 100_000 * 0.01 * (1/252) / 365 != 3.97.
    # The intended formula is short_notional * rate * dt_days/365; with dt_days=1.0 (a single
    # calendar day) the daily charge is $100k * 1% / 365 = $2.74. The "~$3.97" figure in
    # the spec corresponds to dt_days = (365/252) (one trading day expressed in calendar
    # days), which yields exactly 100000 * 0.01 / 252 = $3.968.
    b = ConstantBorrow(rate_bps_annual=100.0)
    dt_days = 365.0 / 252.0
    assert b.borrow_daily(short_notional=100_000.0, dt_days=dt_days) == pytest.approx(
        100_000.0 * 0.01 / 252.0,
        rel=1e-9,
    )


def test_constant_borrow_zero_notional() -> None:
    b = ConstantBorrow(rate_bps_annual=100.0)
    assert b.borrow_daily(short_notional=0.0, dt_days=1.0) == 0.0


def test_profile_borrow_lookup() -> None:
    assert ProfileBorrow("large_cap").rate_bps_annual == 25.0
    assert ProfileBorrow("mid_cap").rate_bps_annual == 100.0
    assert ProfileBorrow("small_cap").rate_bps_annual == 300.0
    with pytest.raises(InputError):
        ProfileBorrow("micro_cap")


def test_profile_borrow_charges_match_constant() -> None:
    p = ProfileBorrow("mid_cap")
    c = ConstantBorrow(rate_bps_annual=100.0)
    assert p.borrow_daily(50_000.0, 1.0) == pytest.approx(c.borrow_daily(50_000.0, 1.0))


def test_two_leg_sizing_dollar_neutral() -> None:
    sa, sb = two_leg_sizing(
        capital=100_000.0,
        price_a=50.0,
        price_b=25.0,
        hedge_ratio=2.0,
        sizing="dollar_neutral",
    )
    # Each leg gets $50k -> 1000 shares of A, 2000 shares of B.
    assert sa == pytest.approx(1000.0)
    assert sb == pytest.approx(2000.0)


def test_two_leg_sizing_unit() -> None:
    sa, sb = two_leg_sizing(
        capital=100_000.0,
        price_a=50.0,
        price_b=25.0,
        hedge_ratio=2.0,
        sizing="unit",
    )
    assert (sa, sb) == (1.0, 2.0)


def test_two_leg_sizing_beta_neutral() -> None:
    sa, sb = two_leg_sizing(
        capital=100.0,
        price_a=10.0,
        price_b=5.0,
        hedge_ratio=2.0,
        sizing="beta_neutral",
    )
    # gross = 10 + 2*5 = 20, scale = 100/20 = 5 -> sa=5, sb=10
    assert sa == pytest.approx(5.0)
    assert sb == pytest.approx(10.0)


def test_two_leg_sizing_rejects_bad_inputs() -> None:
    with pytest.raises(InputError):
        two_leg_sizing(0.0, 1.0, 1.0, 1.0, "dollar_neutral")
    with pytest.raises(InputError):
        two_leg_sizing(1.0, 0.0, 1.0, 1.0, "dollar_neutral")
    with pytest.raises(InputError):
        two_leg_sizing(1.0, 1.0, 1.0, 1.0, "weird")  # type: ignore[arg-type]
