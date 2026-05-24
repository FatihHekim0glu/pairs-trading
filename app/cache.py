"""Cached adapter between the Streamlit dashboard and the pairs library.

The dashboard wants a tiny surface ("scan / explore / backtest / portfolio")
keyed on date ranges and string options. The library exposes a contract-
heavy API (``Candidate`` lists, ``OverlayConfig``, ``PairLifecycle``, …).
Every function in this module is responsible for translating between the two.

Returning typed library objects (``ScreenResult``, ``BacktestResult``,
``PortfolioResult``) keeps page logic thin and lets us reuse the same
``getattr`` extraction utilities everywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------


_PRICE_FIELD_ALIASES = {
    "close": ("Close", "close", "Adj Close", "adj_close"),
    "adj_close": ("Adj Close", "adj_close", "Close", "close"),
    "open": ("Open", "open"),
    "high": ("High", "high"),
    "low": ("Low", "low"),
    "volume": ("Volume", "volume"),
}


def _flatten_prices(raw: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Collapse the MultiIndex ``(ticker, field)`` returned by ``load_prices``
    to a wide ``[date x ticker]`` frame for a single field.

    The yfinance loader returns its native column names — ``"Close"``,
    ``"Adj Close"``, etc. (capitalised, space-separated). The dashboard speaks
    in lowercase canonical names (``"close"``, ``"adj_close"``), so this
    function does the translation and falls back across the alias list before
    giving up.
    """
    if not isinstance(raw.columns, pd.MultiIndex):
        wide = raw.astype(float).sort_index()
    else:
        available = set(raw.columns.get_level_values(1))
        chosen = next(
            (c for c in _PRICE_FIELD_ALIASES.get(field, (field,)) if c in available),
            None,
        )
        if chosen is None:
            msg = (
                f"price field {field!r} not in frame; available fields at level 1: "
                f"{sorted(available)}"
            )
            raise KeyError(msg)
        wide = raw.xs(chosen, axis=1, level=1).astype(float).sort_index()

    # Library code (cointegration, spread, OU, evaluation) was written against
    # naive indices because "trading days" are a calendar abstraction; yfinance
    # ships a tz-aware UTC index. Strip the tz at the adapter boundary so the
    # comparison `prices.index <= formation_window[1]` (naive Timestamp) works.
    if getattr(wide.index, "tz", None) is not None:
        wide = wide.tz_localize(None)
    return wide


@st.cache_resource
def get_universe_singleton(name: str):
    """Return a cached universe object (loaded once per process)."""
    from pairs.data import load_pair_universe, load_universe

    if name.startswith("curated_") or "pair" in name.lower():
        return load_pair_universe(name)
    return load_universe(name)


@st.cache_data(ttl="1d", show_spinner="Fetching prices...")
def fetch_prices(
    tickers: Sequence[str],
    start: date,
    end: date,
    field: str = "close",
) -> pd.DataFrame:
    """Wide close-price frame ``[date x ticker]`` for the requested tickers."""
    from pairs.data import load_prices

    raw = load_prices(list(tickers), start=start, end=end)
    return _flatten_prices(raw, field=field)


# ---------------------------------------------------------------------------
# Pair-finder screen
# ---------------------------------------------------------------------------


def _universe_pairs(name: str) -> list[tuple[str, str]]:
    from pairs.data import load_pair_universe

    try:
        pu = load_pair_universe(name)
        return [(spec.a, spec.b) for spec in pu.pairs]
    except Exception:
        # If the requested universe is not a pair universe (e.g. an ETF
        # constituent list), enumerate all within-universe pairs.
        from itertools import combinations

        from pairs.data import load_universe

        u = load_universe(name)
        return list(combinations(u.tickers, 2))


@st.cache_data(ttl="1d", show_spinner="Running cointegration scan...")
def run_screen(
    universe_name: str,
    train_start: date,
    train_end: date,
    params_hash: str,  # noqa: ARG001 - included for cache-key stability
):
    """Run the cointegration screen for a universe and training window."""
    from pairs.selection import Candidate, screen_cointegration

    pair_tuples = _universe_pairs(universe_name)
    if not pair_tuples:
        msg = f"universe {universe_name!r} resolved to zero pairs"
        raise ValueError(msg)
    tickers = sorted({t for a, b in pair_tuples for t in (a, b)})
    prices = fetch_prices(tuple(tickers), train_start, train_end)
    candidates = [Candidate(ticker_a=a, ticker_b=b) for a, b in pair_tuples]
    return screen_cointegration(
        candidates,
        prices,
        formation_window=(pd.Timestamp(train_start), pd.Timestamp(train_end)),
        alpha=0.10,
        mtc_method="fdr_bh",
        bootstrap=False,
    )


# ---------------------------------------------------------------------------
# Spread explorer
# ---------------------------------------------------------------------------


@st.cache_data(ttl="1h")
def compute_spread_cached(
    pair_tuple: tuple[str, str],
    lookback_days: int,
) -> dict[str, Any]:
    """Compute spread, z-score, hedge ratio, and half-life for a pair."""
    from pairs.spread import build_spread, fit_ou, half_life, tls_hedge, zscore

    end = date.today()
    start = (pd.Timestamp(end) - pd.Timedelta(days=lookback_days * 2)).date()
    prices = fetch_prices(pair_tuple, start, end)
    if pair_tuple[0] not in prices.columns or pair_tuple[1] not in prices.columns:
        msg = f"missing price data for {pair_tuple}"
        raise ValueError(msg)
    y = prices[pair_tuple[0]].dropna()
    x = prices[pair_tuple[1]].dropna()
    joined = pd.concat([y, x], axis=1, join="inner").dropna()
    y, x = joined.iloc[:, 0], joined.iloc[:, 1]
    hedge = tls_hedge(y, x)
    spread = build_spread(y, x, beta=hedge.beta, alpha=hedge.alpha)
    try:
        ou = fit_ou(spread)
        hl = half_life(spread, n_boot=99).point
        z = zscore(spread, window=None, ou_result=ou)
    except Exception:
        # Slow- or non-reverting samples: fall back to a fixed rolling z-score.
        ou = None
        hl = float("nan")
        z = zscore(spread, window=min(lookback_days, len(spread) // 4))
    return {
        "spread": spread,
        "zscore": z,
        "hedge_ratio": float(hedge.beta),
        "hedge_alpha": float(hedge.alpha),
        "half_life": float(hl),
        "ou": ou,
    }


# ---------------------------------------------------------------------------
# Single-pair backtest
# ---------------------------------------------------------------------------


_SIZING_ALIAS = {
    "fixed_notional": "dollar_neutral",
    "vol_target": "beta_neutral",
    "kelly_capped": "unit",
    "dollar_neutral": "dollar_neutral",
    "beta_neutral": "beta_neutral",
    "unit": "unit",
}


def _build_backtest_inputs(
    prices_a: pd.Series, prices_b: pd.Series, train_frac: float = 0.4
) -> tuple[pd.Series, pd.Series, float, float, pd.Series, pd.Series]:
    """Estimate hedge on a leading slice, then produce signal over the full window."""
    from pairs.spread import build_spread, fit_ou, tls_hedge, zscore
    from pairs.strategy import generate_signals

    n = len(prices_a)
    train_n = max(60, int(n * train_frac))
    y_train, x_train = prices_a.iloc[:train_n], prices_b.iloc[:train_n]
    hedge = tls_hedge(y_train, x_train)
    spread_full = build_spread(prices_a, prices_b, beta=hedge.beta, alpha=hedge.alpha)
    try:
        ou = fit_ou(spread_full.iloc[:train_n])
        z = zscore(spread_full, window=None, ou_result=ou)
        signal = generate_signals(z, half_life=ou.half_life)
    except Exception:
        z = zscore(spread_full, window=60)
        signal = generate_signals(z)
    return prices_a, prices_b, float(hedge.beta), float(hedge.alpha), z, signal


def _run_backtest_window(
    prices_a: pd.Series,
    prices_b: pd.Series,
    cost_profile: str,
    sizing: str,
):
    """Single-window backtest helper. Estimates hedge on the first 40% of the
    window, builds spread + signal, runs `backtest_pair`. Returns the
    `BacktestResult` plus the estimated hedge ratio."""
    from pairs.backtest import backtest_pair, load_profile

    y, x, beta, _alpha, _z, signal = _build_backtest_inputs(prices_a, prices_b)
    profile = load_profile(cost_profile)
    sizing_mapped = _SIZING_ALIAS.get(sizing, "dollar_neutral")
    result = backtest_pair(
        y, x, signal, hedge_ratio=beta, cost_model=profile, sizing=sizing_mapped
    )
    return result, beta


@st.cache_data(ttl="1d", show_spinner="Running backtest...")
def run_backtest_cached(
    pair: tuple[str, str],
    oos_start: date,
    oos_end: date,
    cost_profile: str,
    sizing: str,
) -> dict[str, Any]:
    """Run an IS + OOS pair of backtests and return a dict the dashboard can
    render directly.

    The page-side `_extract_metric` helper does `result[name]` lookup on
    dicts, so every metric the page asks for must be present here as a
    flat key:

    - ``is_sharpe``, ``oos_sharpe`` — Sharpe ratios for the two windows
    - ``max_drawdown`` — peak-to-trough on the OOS equity curve
    - ``turnover`` — OOS annualised position turnover
    - ``equity_curve`` — OOS equity ``pd.Series`` for the drawdown plot
    - ``trades`` — OOS trade log
    - ``is_result``, ``oos_result`` — full ``BacktestResult`` objects for
      power users / future expansion
    """
    # Symmetric IS / OOS split: train on the year before oos_start, then
    # trade on [oos_start, oos_end].
    train_start = (pd.Timestamp(oos_start) - pd.Timedelta(days=365)).date()
    train_end = (pd.Timestamp(oos_start) - pd.Timedelta(days=1)).date()
    full = fetch_prices(pair, train_start, oos_end)
    if pair[0] not in full.columns or pair[1] not in full.columns:
        msg = f"missing price data for {pair}"
        raise ValueError(msg)
    joined = full[list(pair)].dropna()

    train_mask = joined.index <= pd.Timestamp(train_end)
    oos_mask = joined.index >= pd.Timestamp(oos_start)
    train_panel = joined.loc[train_mask]
    oos_panel = joined.loc[oos_mask]
    if len(train_panel) < 60 or len(oos_panel) < 30:
        msg = (
            f"insufficient history for {pair}: train={len(train_panel)} rows, "
            f"oos={len(oos_panel)} rows"
        )
        raise ValueError(msg)

    is_result, _is_beta = _run_backtest_window(
        train_panel[pair[0]], train_panel[pair[1]], cost_profile, sizing
    )
    oos_result, _oos_beta = _run_backtest_window(
        oos_panel[pair[0]], oos_panel[pair[1]], cost_profile, sizing
    )

    is_m = is_result.metrics
    oos_m = oos_result.metrics
    return {
        "is_result": is_result,
        "oos_result": oos_result,
        "is_sharpe": float(is_m.get("sharpe_net", float("nan"))),
        "oos_sharpe": float(oos_m.get("sharpe_net", float("nan"))),
        "sharpe": float(oos_m.get("sharpe_net", float("nan"))),  # for the page's fallback chain
        "max_drawdown": float(oos_m.get("max_drawdown", float("nan"))),
        "turnover": float(oos_m.get("turnover", float("nan"))),
        "equity_curve": oos_result.equity,
        "trades": oos_result.trades,
    }


# ---------------------------------------------------------------------------
# Multi-pair portfolio orchestration
# ---------------------------------------------------------------------------


def _allocator(name: str):
    from pairs.portfolio import EqualDollarAllocator, HRPAllocator, InverseVolAllocator

    return {
        "equal": EqualDollarAllocator(),
        "inverse_vol": InverseVolAllocator(),
        "hrp": HRPAllocator(),
    }.get(name, EqualDollarAllocator())


def _passing_retest(_pid: str, _asof: pd.Timestamp, _prices: pd.DataFrame) -> object:
    """Permissive retest stub: every pair stays cointegrated for the demo run."""

    class _Pass:
        cointegrated = True

    return _Pass()


@st.cache_data(ttl="1d", show_spinner="Running multi-pair portfolio backtest...")
def run_portfolio_cached(
    pairs: Sequence[tuple[str, str]],
    allocator_name: str,
    rebalance: str,
    oos_start: date,
    oos_end: date,
    cost_profile: str,
) -> dict[str, Any]:
    """Run a multi-pair portfolio backtest and return a dict the dashboard can
    render directly (the page reads dict-keys via `_safe_get`)."""
    from pairs.portfolio import OverlayConfig, PairLifecycle, run_multi_pair_backtest

    if not pairs:
        msg = "select at least one pair before running the portfolio backtest"
        raise ValueError(msg)

    # Each per-pair entry must be a BacktestResult (not the dict the page-side
    # adapter returns), so unwrap our IS/OOS bundle to its OOS result.
    pair_results: dict[str, Any] = {}
    per_pair_pnl: dict[str, float] = {}
    per_pair_sharpe: dict[str, float] = {}
    pair_returns_cols: dict[str, pd.Series] = {}
    asset_legs_map: dict[str, tuple[str, str]] = {}
    sector_map: dict[str, str] = {}
    for a, b in pairs:
        pid = f"{a}__{b}"
        bundle = run_backtest_cached((a, b), oos_start, oos_end, cost_profile, "fixed_notional")
        oos_result = bundle["oos_result"]
        pair_results[pid] = oos_result
        per_pair_pnl[pid] = float(oos_result.returns.sum())
        per_pair_sharpe[pid] = float(bundle.get("oos_sharpe", 0.0))
        pair_returns_cols[pid] = oos_result.returns
        asset_legs_map[pid] = (a, b)
        sector_map[pid] = "unknown"

    tickers = sorted({t for pair in pairs for t in pair})
    prices = fetch_prices(tuple(tickers), oos_start, oos_end)

    overlay = OverlayConfig()
    lifecycle = PairLifecycle(
        cointegration_retest=_passing_retest,
        half_life_lookup=lambda _pid: 20.0,
        min_cooldown_days=10,
    )

    rebalance_offset = {"monthly": "MS", "quarterly": "QS", "annual": "YS"}.get(
        rebalance, "QS"
    )
    walk_forward_dates = pd.date_range(
        pd.Timestamp(oos_start), pd.Timestamp(oos_end), freq=rebalance_offset
    ).tolist()

    pr = run_multi_pair_backtest(
        pair_results,
        prices,
        allocator=_allocator(allocator_name),
        overlay_config=overlay,
        lifecycle=lifecycle,
        walk_forward_dates=walk_forward_dates,
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
    )

    m = pr.metrics
    return {
        "portfolio_result": pr,
        "equity_curve": pr.equity,
        "returns": pr.returns,
        "sharpe": float(m.get("sharpe", 0.0)),
        "portfolio_sharpe": float(m.get("sharpe", 0.0)),
        "annual_return": float(m.get("annualised_return", 0.0)),
        "annualised_return": float(m.get("annualised_return", 0.0)),
        "annual_vol": float(m.get("annualised_vol", 0.0)),
        "annualised_vol": float(m.get("annualised_vol", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "per_pair_pnl": per_pair_pnl,
        "per_pair_sharpe": per_pair_sharpe,
        "pair_returns": pd.DataFrame(pair_returns_cols).sort_index(),
    }


__all__ = [
    "compute_spread_cached",
    "fetch_prices",
    "get_universe_singleton",
    "run_backtest_cached",
    "run_portfolio_cached",
    "run_screen",
]
