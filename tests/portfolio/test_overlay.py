from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pairs.portfolio import drawdown_killswitch, vol_target_overlay


def test_vol_target_reduces_vol_toward_target() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    # Vol per bar = 0.16 / sqrt(252) ~ 0.0101 (annualised 16%).
    daily_vol = 0.16 / math.sqrt(252)
    returns = pd.Series(rng.normal(scale=daily_vol, size=n), index=pd.bdate_range("2010-01-04", periods=n))
    mult = vol_target_overlay(returns, target_vol=0.08, window=20, clip=(0.1, 2.0))
    scaled = returns * mult
    realised_ann = float(scaled.std(ddof=1)) * math.sqrt(252)
    assert 0.06 <= realised_ann <= 0.10


def test_vol_target_clipped() -> None:
    n = 500
    # Constant zero returns -> infinite multiplier ratio -> must be clipped.
    returns = pd.Series(0.0, index=pd.bdate_range("2020-01-02", periods=n))
    # Make rolling std nonzero but tiny so ratio is huge.
    returns.iloc[::100] = 1e-12
    mult = vol_target_overlay(returns, target_vol=0.08, window=20, clip=(0.5, 1.5))
    assert (mult <= 1.5 + 1e-9).all()
    assert (mult >= 0.5 - 1e-9).all() or (mult.dropna() >= 0.5 - 1e-9).all()


def test_vol_target_no_lookahead() -> None:
    rng = np.random.default_rng(1)
    n = 300
    base = rng.normal(scale=0.01, size=n)
    idx = pd.bdate_range("2020-01-02", periods=n)
    s1 = pd.Series(base.copy(), index=idx)
    s2 = pd.Series(base.copy(), index=idx)
    s2.iloc[200:] += 0.5  # perturb the future
    m1 = vol_target_overlay(s1, target_vol=0.08, window=20)
    m2 = vol_target_overlay(s2, target_vol=0.08, window=20)
    # Multipliers up to bar 200 must be identical.
    pd.testing.assert_series_equal(m1.iloc[:200], m2.iloc[:200])


def test_killswitch_trips_at_threshold() -> None:
    idx = pd.bdate_range("2020-01-02", periods=200)
    # Build an equity curve with a clear 10% drawdown.
    eq = pd.Series(np.linspace(1.0, 1.2, 100).tolist() + np.linspace(1.2, 1.05, 100).tolist(), index=idx)
    mult, events = drawdown_killswitch(eq, dd_threshold=0.08, dd_window=200, ladder_days=10)
    assert any(e.trigger == "dd_threshold" for e in events)
    # Multiplier should drop to 0 at some point after the trip.
    assert (mult == 0.0).any()


def test_killswitch_ladder_back() -> None:
    n = 400
    idx = pd.bdate_range("2020-01-02", periods=n)
    # Up to 1.2, then drop to 1.05 (10% DD), then stay flat at 1.05.
    body = list(np.linspace(1.0, 1.2, 100)) + list(np.linspace(1.2, 1.05, 50)) + [1.05] * (n - 150)
    eq = pd.Series(body, index=idx)
    _mult, events = drawdown_killswitch(eq, dd_threshold=0.08, dd_window=400, ladder_days=10)
    triggers = [e.trigger for e in events]
    assert "dd_threshold" in triggers
    assert "recover_half" in triggers
    assert "recover_full" in triggers


def test_killswitch_no_flap() -> None:
    n = 100
    idx = pd.bdate_range("2020-01-02", periods=n)
    eq = pd.Series(np.linspace(1.0, 1.05, n), index=idx)
    mult, events = drawdown_killswitch(eq, dd_threshold=0.08, dd_window=20, ladder_days=10)
    assert events == []
    assert (mult == 1.0).all()


def test_vol_target_empty_series_returns_empty() -> None:
    s = pd.Series(dtype=float)
    mult = vol_target_overlay(s)
    assert len(mult) == 0
