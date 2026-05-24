from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pairs.portfolio import (
    EqualDollarAllocator,
    HRPAllocator,
    InverseVolAllocator,
    OverlayConfig,
    PairLifecycle,
    PortfolioResult,
    run_multi_pair_backtest,
)


def _build(rng: np.random.Generator):
    n_days, n_pairs = 300, 5
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    pair_results = {}
    for i in range(n_pairs):
        r = pd.Series(rng.normal(scale=0.005, size=n_days), index=idx)
        pair_results[f"P{i}"] = SimpleNamespace(returns=r, equity=(1 + r).cumprod(), metrics={})
    sector_map = {f"P{i}": ("SEC0" if i < 2 else "SEC1") for i in range(n_pairs)}
    asset_legs_map = {f"P{i}": (f"A{i}", f"A{i + 100}") for i in range(n_pairs)}
    lifecycle = PairLifecycle(
        cointegration_retest=lambda *_a, **_k: SimpleNamespace(cointegrated=True),
        half_life_lookup=lambda _pid: 5.0,
        min_cooldown_days=10,
    )
    return pair_results, sector_map, asset_legs_map, lifecycle


@pytest.mark.parametrize(
    "allocator",
    [
        EqualDollarAllocator(),
        InverseVolAllocator(window=60, min_periods=20),
        HRPAllocator(min_history=60),
    ],
    ids=["equal", "invvol", "hrp"],
)
def test_three_allocators_run_on_toy_5pair(allocator) -> None:
    rng = np.random.default_rng(42)
    pair_results, sector_map, asset_legs_map, lifecycle = _build(rng)
    result = run_multi_pair_backtest(
        pair_results,
        prices=pd.DataFrame(),
        allocator=allocator,
        overlay_config=OverlayConfig(),
        lifecycle=lifecycle,
        walk_forward_dates=[],
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
    )
    assert isinstance(result, PortfolioResult)
    assert len(result.equity) == 300
    assert len(result.returns) == 300
    assert result.weights_history.shape == (300, 5)
