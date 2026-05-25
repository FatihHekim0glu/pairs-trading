from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from pairs.portfolio import (
    EqualDollarAllocator,
    OverlayConfig,
    PairLifecycle,
    run_multi_pair_backtest,
)


def _make_lifecycle(always_cointegrated: bool = True) -> PairLifecycle:
    return PairLifecycle(
        cointegration_retest=lambda *_a, **_k: SimpleNamespace(cointegrated=always_cointegrated),
        half_life_lookup=lambda _pid: 5.0,
        min_cooldown_days=10,
    )


def _make_results(rng: np.random.Generator, n_days: int = 300, n_pairs: int = 4):
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    pair_results = {}
    for i in range(n_pairs):
        r = pd.Series(rng.normal(scale=0.005, size=n_days), index=idx)
        pair_results[f"P{i}"] = SimpleNamespace(returns=r, equity=(1 + r).cumprod(), metrics={})
    sector_map = {f"P{i}": ("SEC0" if i < 2 else "SEC1") for i in range(n_pairs)}
    asset_legs_map = {f"P{i}": (f"A{i}", f"A{i + 100}") for i in range(n_pairs)}
    return pair_results, idx, sector_map, asset_legs_map


def test_walkforward_resets_weights_at_quarter_boundary() -> None:
    rng = np.random.default_rng(0)
    pair_results, idx, sector_map, asset_legs_map = _make_results(rng)
    wf_dates = [idx[60], idx[180]]
    result = run_multi_pair_backtest(
        pair_results,
        prices=pd.DataFrame(),
        allocator=EqualDollarAllocator(),
        overlay_config=OverlayConfig(),
        lifecycle=_make_lifecycle(),
        walk_forward_dates=wf_dates,
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
    )
    reselections = [e for e in result.cap_events if e.kind == "reselection"]
    assert len(reselections) == 2


def test_walkforward_logs_reselection_event() -> None:
    rng = np.random.default_rng(1)
    pair_results, idx, sector_map, asset_legs_map = _make_results(rng)
    wf_dates = [idx[50]]
    result = run_multi_pair_backtest(
        pair_results,
        prices=pd.DataFrame(),
        allocator=EqualDollarAllocator(),
        overlay_config=OverlayConfig(),
        lifecycle=_make_lifecycle(),
        walk_forward_dates=wf_dates,
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
    )
    reselects = [e for e in result.cap_events if e.kind == "reselection"]
    assert len(reselects) == 1
    assert reselects[0].asof == idx[50]


def test_runner_single_active_pair_weight_one() -> None:
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2020-01-02", periods=100)
    r = pd.Series(rng.normal(scale=0.002, size=100), index=idx)
    pair_results = {"P0": SimpleNamespace(returns=r, equity=(1 + r).cumprod(), metrics={})}
    sector_map = {"P0": "SEC0"}
    asset_legs_map = {"P0": ("A0", "A1")}
    result = run_multi_pair_backtest(
        pair_results,
        prices=pd.DataFrame(),
        allocator=EqualDollarAllocator(),
        overlay_config=OverlayConfig(),
        lifecycle=_make_lifecycle(),
        walk_forward_dates=[],
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
        cap_kwargs={"max_pairs": 5, "max_sector_gross": 1.0, "max_asset_notional": 1.0},
    )
    # After warmup, P0 should hold full weight 1.0.
    assert np.isclose(result.weights_history["P0"].iloc[-1], 1.0)


def test_runner_all_stopped_out_gross_zero() -> None:
    rng = np.random.default_rng(3)
    pair_results, idx, sector_map, asset_legs_map = _make_results(rng)
    lc = _make_lifecycle(always_cointegrated=False)
    # Stop out everything before run start.
    for pid in pair_results:
        lc.on_stop_out(pid, idx[0] - pd.Timedelta(days=1))
    result = run_multi_pair_backtest(
        pair_results,
        prices=pd.DataFrame(),
        allocator=EqualDollarAllocator(),
        overlay_config=OverlayConfig(),
        lifecycle=lc,
        walk_forward_dates=[],
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
    )
    assert np.isclose(result.gross_history.sum(), 0.0)
    assert np.isclose(result.returns.abs().sum(), 0.0)


def test_runner_no_lookahead_end_to_end() -> None:
    rng = np.random.default_rng(4)
    pair_results_a, _idx, sector_map, asset_legs_map = _make_results(rng, n_days=200, n_pairs=3)
    # Build version B by perturbing returns at t >= 150.
    pair_results_b = {}
    for pid, res in pair_results_a.items():
        r = res.returns.copy()
        r.iloc[150:] += 0.05
        pair_results_b[pid] = SimpleNamespace(returns=r, equity=(1 + r).cumprod(), metrics={})

    def _run(prs):
        return run_multi_pair_backtest(
            prs,
            prices=pd.DataFrame(),
            allocator=EqualDollarAllocator(),
            overlay_config=OverlayConfig(),
            lifecycle=_make_lifecycle(),
            walk_forward_dates=[],
            sector_map=sector_map,
            asset_legs_map=asset_legs_map,
        )

    a = _run(pair_results_a)
    b = _run(pair_results_b)
    # Weights at bar t depend only on history < t, so weights[<150] identical.
    pd.testing.assert_frame_equal(a.weights_history.iloc[:150], b.weights_history.iloc[:150])


def test_runner_metrics_populated() -> None:
    rng = np.random.default_rng(5)
    pair_results, _, sector_map, asset_legs_map = _make_results(rng)
    result = run_multi_pair_backtest(
        pair_results,
        prices=pd.DataFrame(),
        allocator=EqualDollarAllocator(),
        overlay_config=OverlayConfig(),
        lifecycle=_make_lifecycle(),
        walk_forward_dates=[],
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
    )
    for key in ("annualised_return", "annualised_vol", "sharpe", "max_drawdown"):
        assert key in result.metrics
