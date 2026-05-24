"""End-to-end smoke tests for the dashboard adapter (`app/cache.py`).

These tests mock `pairs.data.load_prices` to return a tz-aware MultiIndex
frame matching what yfinance actually emits, then exercise every adapter
function end-to-end. They are the layer that proves the dashboard talks to
the library correctly — the unit suite under `tests/unit/` and the
per-module suites under `tests/<module>/` do not cover this seam.

If something in the adapter pipeline silently breaks (wrong kwarg name,
wrong return type, timezone mismatch, etc.) one of these tests will catch
it before a user clicks the page.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

pytest.importorskip("streamlit", reason="streamlit not installed (in [app] extras)")


# -----------------------------------------------------------------------------
# Synthetic data factories — match the real `load_prices` return shape exactly.
# -----------------------------------------------------------------------------


def _make_yf_panel(
    tickers: list[str],
    n: int = 750,
    *,
    seed: int = 0,
    cointegrated: bool = True,
    start: str = "2018-01-02",
    end: str | None = None,
) -> pd.DataFrame:
    """Build a MultiIndex DataFrame matching what `pairs.data.load_prices`
    returns: tz-aware UTC DatetimeIndex, columns `(ticker, Open/High/Low/Close/Adj Close/Volume)`.

    If `end` is given, the index is the business-day range `[start, end]`
    and `n` is ignored — this lets the mock cover whatever window a caller
    requests.
    """
    if end is not None:
        idx = pd.date_range(start, end, freq="B", tz="UTC")
    else:
        idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    n = len(idx)
    rng = np.random.default_rng(seed)
    common_trend = np.cumsum(rng.standard_normal(n))
    frames = []
    for i, t in enumerate(tickers):
        if cointegrated:
            level = 100.0 + common_trend + rng.standard_normal(n).cumsum() * 0.0
            noise = rng.standard_normal(n) * 0.5
            close = level + noise + i * 5.0
        else:
            close = 100.0 + np.cumsum(rng.standard_normal(n))
        close = np.maximum(close, 1.0)  # keep positive
        df = pd.DataFrame(
            {
                "Open": close * (1.0 + rng.standard_normal(n) * 0.001),
                "High": close * 1.005,
                "Low": close * 0.995,
                "Close": close,
                "Adj Close": close,
                "Volume": rng.integers(1_000_000, 10_000_000, size=n),
            },
            index=idx,
        )
        df.columns = pd.MultiIndex.from_product([[t], df.columns], names=["ticker", "field"])
        frames.append(df)
    return pd.concat(frames, axis=1).sort_index(axis=1)


@pytest.fixture
def yf_panel_factory():
    """Pytest fixture: callable that builds a yf-like panel."""
    return _make_yf_panel


@pytest.fixture
def mocked_load_prices(yf_panel_factory):
    """Patch `pairs.data.load_prices` to return a deterministic yf-shaped panel.

    Yields the patcher so individual tests can inspect call args if needed.
    """
    with patch("pairs.data.load_prices") as mock_lp:

        def _fake(tickers, start=None, end=None, **kw):
            # Cover whatever window the caller asks for so date-range slicing
            # downstream actually finds rows.
            start_str = str(start) if start is not None else "2018-01-02"
            end_str = str(end) if end is not None else "2026-12-31"
            return yf_panel_factory(list(tickers), start=start_str, end=end_str)

        mock_lp.side_effect = _fake
        yield mock_lp


# -----------------------------------------------------------------------------
# fetch_prices — the seam everyone else depends on.
# -----------------------------------------------------------------------------


def test_fetch_prices_returns_wide_naive_index(mocked_load_prices):
    """The adapter must hand downstream code a wide frame with a tz-naive
    index (library code was written against naive indices)."""
    from app.cache import fetch_prices

    wide = fetch_prices.__wrapped__(("KO", "PEP"), date(2020, 1, 1), date(2022, 12, 31))
    assert isinstance(wide, pd.DataFrame)
    assert list(wide.columns) == ["KO", "PEP"]
    assert wide.index.tz is None, (
        f"index must be tz-naive after _flatten_prices, got tz={wide.index.tz}"
    )
    assert not wide.isna().all().any(), "no column should be entirely NaN"


# -----------------------------------------------------------------------------
# run_screen — full cointegration scan.
# -----------------------------------------------------------------------------


def test_run_screen_completes_on_curated_universe(mocked_load_prices):
    """run_screen must hand a tz-naive price panel + properly-constructed
    formation_window to `screen_cointegration` without raising."""
    from app.cache import run_screen

    result = run_screen.__wrapped__(
        "curated_25_v1", date(2020, 1, 1), date(2022, 12, 31), "hash"
    )
    # ScreenResult exposes a `diagnostics` DataFrame.
    diag = getattr(result, "diagnostics", None)
    assert diag is not None, "ScreenResult must expose a diagnostics frame"
    assert isinstance(diag, pd.DataFrame)
    assert len(diag) > 0, "scan should have produced at least one candidate row"


# -----------------------------------------------------------------------------
# compute_spread_cached — spread + zscore + half-life pipeline.
# -----------------------------------------------------------------------------


def test_compute_spread_cached_returns_expected_keys(mocked_load_prices):
    from app.cache import compute_spread_cached

    bundle = compute_spread_cached.__wrapped__(("KO", "PEP"), 252)
    for key in ("spread", "zscore", "hedge_ratio", "hedge_alpha", "half_life", "ou"):
        assert key in bundle, f"compute_spread_cached missing key {key!r}"
    assert isinstance(bundle["spread"], pd.Series)
    assert isinstance(bundle["zscore"], pd.Series)
    assert isinstance(bundle["hedge_ratio"], float)
    # Half-life can be NaN on degenerate inputs but must be a float.
    assert isinstance(bundle["half_life"], float)


# -----------------------------------------------------------------------------
# run_backtest_cached — signal construction + backtest_pair contract.
# -----------------------------------------------------------------------------


def test_run_backtest_cached_returns_dashboard_dict(mocked_load_prices):
    """run_backtest_cached must return a dict whose keys match what the
    Backtest page reads via `_extract_metric`. Without this assertion every
    KPI tile would silently render as 'nan'."""
    from app.cache import run_backtest_cached

    result = run_backtest_cached.__wrapped__(
        ("KO", "PEP"),
        date(2022, 1, 1),
        date(2022, 12, 31),
        "large_cap_realistic",
        "fixed_notional",
    )
    assert isinstance(result, dict)
    for key in (
        "is_result", "oos_result",
        "is_sharpe", "oos_sharpe", "sharpe",
        "max_drawdown", "turnover",
        "equity_curve", "trades",
    ):
        assert key in result, f"backtest bundle missing key {key!r}"
    assert isinstance(result["equity_curve"], pd.Series)
    assert len(result["equity_curve"]) > 0
    assert isinstance(result["is_sharpe"], float)
    assert isinstance(result["oos_sharpe"], float)


def test_run_backtest_cached_degrades_when_train_history_unavailable(yf_panel_factory):
    """REGRESSION: when the library's price cache holds only the OOS window
    (because earlier runs fetched only the OOS range and cache extends right
    not left), the train fetch returns empty after dropna. The adapter must
    NOT raise — it must fall back to OOS-only with `is_sharpe=nan` so the
    page still renders."""
    from unittest.mock import patch as _patch

    from app.cache import run_backtest_cached

    def _train_blind(tickers, start=None, end=None, **kw):
        # Mimic a cache that only holds 2025+ data: train fetches (2024)
        # come back empty.
        s = pd.Timestamp(start) if start is not None else pd.Timestamp("2025-01-01")
        if s.year < 2025:
            return yf_panel_factory(
                list(tickers), start="2025-01-01", end="2025-01-02"
            ).iloc[0:0]
        return yf_panel_factory(list(tickers), start=str(start), end=str(end))

    with _patch("pairs.data.load_prices", side_effect=_train_blind):
        result = run_backtest_cached.__wrapped__(
            ("KO", "PEP"),
            date(2025, 6, 1),
            date(2026, 5, 1),
            "large_cap_realistic",
            "fixed_notional",
        )

    assert isinstance(result, dict)
    assert result["train_available"] is False
    assert pd.isna(result["is_sharpe"])
    assert not pd.isna(result["oos_sharpe"])
    assert isinstance(result["equity_curve"], pd.Series)


@pytest.mark.parametrize("sizing", ["fixed_notional", "vol_target", "kelly_capped"])
def test_run_backtest_cached_accepts_every_ui_sizing(sizing, mocked_load_prices):
    """Every sizing string the UI offers must round-trip through the adapter
    and the underlying engine without raising."""
    from app.cache import run_backtest_cached

    run_backtest_cached.__wrapped__(
        ("KO", "PEP"),
        date(2022, 1, 1),
        date(2022, 12, 31),
        "large_cap_realistic",
        sizing,
    )


# -----------------------------------------------------------------------------
# run_portfolio_cached — full multi-pair orchestration.
# -----------------------------------------------------------------------------


def test_run_portfolio_cached_returns_dashboard_dict(mocked_load_prices):
    """run_portfolio_cached must return a dict whose keys match what the
    Portfolio page reads via `_safe_get` and `getattr(..., key, None)`."""
    from app.cache import run_portfolio_cached

    result = run_portfolio_cached.__wrapped__(
        (("KO", "PEP"), ("XOM", "CVX")),
        "equal",
        "quarterly",
        date(2022, 1, 1),
        date(2022, 12, 31),
        "large_cap_realistic",
    )
    assert isinstance(result, dict)
    for key in (
        "portfolio_result", "equity_curve", "returns",
        "sharpe", "portfolio_sharpe",
        "annual_return", "annualised_return",
        "annual_vol", "annualised_vol",
        "max_drawdown",
        "per_pair_pnl", "per_pair_sharpe", "pair_returns",
    ):
        assert key in result, f"portfolio bundle missing key {key!r}"
    assert isinstance(result["equity_curve"], pd.Series)
    assert isinstance(result["per_pair_pnl"], dict)
    assert isinstance(result["pair_returns"], pd.DataFrame)
    assert not result["pair_returns"].empty


@pytest.mark.parametrize("allocator", ["equal", "inverse_vol", "hrp"])
def test_run_portfolio_cached_each_allocator(allocator, mocked_load_prices):
    """Every allocator the UI offers must compose with the runner."""
    from app.cache import run_portfolio_cached

    run_portfolio_cached.__wrapped__(
        (("KO", "PEP"), ("XOM", "CVX"), ("MSFT", "ORCL")),
        allocator,
        "monthly",
        date(2022, 6, 1),
        date(2022, 12, 31),
        "large_cap_realistic",
    )
