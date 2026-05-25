"""Smoke tests for every ``app/plots/*.py`` render function.

The render functions are thin wrappers around Plotly / pandas Styler — the
kind of presentation code that breaks silently across pandas/plotly major
versions (e.g. ``Styler.applymap`` removed in pandas 3.0). Without this
file, that class of bug only surfaces when a user clicks the page.

For pure-plotly renders we call the function and assert it returns a
``go.Figure`` (i.e., the call did not raise). For renders that touch
``streamlit`` (e.g. ``pair_scan_table.render`` calls ``st.dataframe``,
``st.info``) we mock the streamlit module so the call resolves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# -----------------------------------------------------------------------------
# Shared synthetic data factories
# -----------------------------------------------------------------------------


@pytest.fixture
def synth_equity() -> pd.Series:
    """Equity curve with a deterministic drawdown band."""
    idx = pd.date_range("2024-01-02", periods=250, freq="B")
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(250) * 0.005 + 0.0003
    return pd.Series((1.0 + returns).cumprod(), index=idx, name="equity")


@pytest.fixture
def synth_prices() -> pd.DataFrame:
    """Two-leg price frame for overlay plotting."""
    idx = pd.date_range("2024-01-02", periods=200, freq="B")
    rng = np.random.default_rng(1)
    common = np.cumsum(rng.standard_normal(200))
    return pd.DataFrame(
        {
            "KO": 100.0 + common + rng.standard_normal(200) * 0.5,
            "PEP": 100.0 + 1.2 * common + rng.standard_normal(200) * 0.5,
        },
        index=idx,
    )


@pytest.fixture
def synth_spread_zscore() -> tuple[pd.Series, pd.Series]:
    idx = pd.date_range("2024-01-02", periods=300, freq="B")
    rng = np.random.default_rng(2)
    # OU-like mean-reverting series.
    spread = pd.Series(0.0, index=idx)
    for t in range(1, 300):
        spread.iloc[t] = 0.85 * spread.iloc[t - 1] + rng.standard_normal()
    rolling = spread.rolling(60, min_periods=60)
    z = (spread - rolling.mean()) / rolling.std()
    return spread, z


@pytest.fixture
def synth_diagnostics() -> pd.DataFrame:
    """A ScreenResult-style diagnostics DataFrame.

    This is the EXACT shape `pair_scan_table.render` consumes — used to
    catch styler / display bugs at test time."""
    return pd.DataFrame(
        {
            "pair_id": ["KO__PEP", "MA__V", "XOM__CVX"],
            "ticker_a": ["KO", "MA", "XOM"],
            "ticker_b": ["PEP", "V", "CVX"],
            "p_raw": [0.003, 0.04, 0.21],
            "q_value": [0.009, 0.06, 0.21],
            "survives_mtc": [True, False, False],
            "hedge_ratio": [1.21, 0.97, 1.05],
            "half_life": [12.4, 28.1, 64.0],
        }
    )


# -----------------------------------------------------------------------------
# Streamlit patcher — narrow mock that only replaces the specific st.X calls
# `pair_scan_table.render` makes, leaving the rest of the imported `streamlit`
# module intact so other libraries (st_aggrid, internal logging) keep working.
# -----------------------------------------------------------------------------


@pytest.fixture
def patched_streamlit(monkeypatch):
    """Patch only the st.X functions render code calls, not the whole module."""
    import streamlit as st

    calls: dict[str, list] = {"info": [], "dataframe": [], "success": []}
    monkeypatch.setattr(st, "info", lambda msg, **_: calls["info"].append(msg))
    monkeypatch.setattr(st, "dataframe", lambda *a, **kw: calls["dataframe"].append((a, kw)))
    monkeypatch.setattr(st, "success", lambda msg, **_: calls["success"].append(msg))
    # Ensure session_state has the keys the render touches.
    if "selected_pair" not in st.session_state:
        st.session_state["selected_pair"] = None
    return calls


# -----------------------------------------------------------------------------
# Pure-plotly renders
# -----------------------------------------------------------------------------


def test_equity_drawdown_render_returns_figure(synth_equity):
    from app.plots import equity_drawdown

    fig = equity_drawdown.render(synth_equity)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_equity_drawdown_render_with_benchmark(synth_equity):
    from app.plots import equity_drawdown

    fig = equity_drawdown.render(synth_equity, benchmark=synth_equity * 0.9)
    assert isinstance(fig, go.Figure)


def test_price_overlay_render_returns_figure(synth_prices):
    from app.plots import price_overlay

    fig = price_overlay.render(synth_prices["KO"], synth_prices["PEP"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_spread_zscore_render_returns_figure(synth_spread_zscore):
    from app.plots import spread_zscore

    spread, z = synth_spread_zscore
    fig = spread_zscore.render(
        spread, z, z_entry=2.0, z_exit=0.5, z_stop=3.0, half_life=10.0, hedge_ratio=1.2
    )
    assert isinstance(fig, go.Figure)


def test_is_vs_oos_bars_render_returns_figure():
    from app.plots import is_vs_oos_bars

    df = pd.DataFrame(
        {
            "pair": ["KO/PEP", "MA/V", "XOM/CVX"],
            "is_sharpe": [2.4, 1.8, 1.5],
            "oos_sharpe": [0.7, 0.6, 0.9],
        }
    )
    fig = is_vs_oos_bars.render(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # IS bars + OOS bars


def test_pair_contribution_render_returns_figure():
    from app.plots import pair_contribution

    pnl = {"KO__PEP": 0.04, "MA__V": -0.01, "XOM__CVX": 0.02}
    sharpe = {"KO__PEP": 0.9, "MA__V": -0.2, "XOM__CVX": 0.5}
    fig = pair_contribution.render(pnl, sharpe)
    assert isinstance(fig, go.Figure)


# -----------------------------------------------------------------------------
# Renders that touch streamlit (need the mock) — this includes the styler
# bug that broke production today.
# -----------------------------------------------------------------------------


def test_pair_scan_table_styler_handles_q_value(synth_diagnostics):
    """REGRESSION: catches `Styler.applymap` removal in pandas 3.0.

    The previous implementation called `styler.applymap(...)` which raises
    `AttributeError: 'Styler' object has no attribute 'applymap'` on pandas
    >=3.0. This test exercises the styling helper directly so we never ship
    that bug to a user again."""
    from app.plots.pair_scan_table import _style_qvalue

    styled = _style_qvalue(synth_diagnostics)
    # Force materialisation so any deferred Styler errors fire here.
    rendered = styled.to_html()
    assert "q_value" in rendered
    assert "background-color" in rendered, "q_value column should have colour-coded cells"


def test_pair_scan_table_render_empty(patched_streamlit):
    from app.plots import pair_scan_table

    pair_scan_table.render(pd.DataFrame())
    assert len(patched_streamlit["info"]) >= 1
    assert "No scan results" in patched_streamlit["info"][0]


def test_pair_scan_table_render_full(patched_streamlit, synth_diagnostics, monkeypatch):
    """Force the dataframe-fallback path so AgGrid's JSON-parsing internals
    don't enter the test (they require a real Streamlit websocket context)."""
    from app.plots import pair_scan_table

    monkeypatch.setattr(pair_scan_table, "_try_aggrid", lambda df: None)
    pair_scan_table.render(synth_diagnostics)
    # Dataframe fallback should have rendered the styled diagnostics.
    assert len(patched_streamlit["dataframe"]) == 1


# -----------------------------------------------------------------------------
# Gantt — depends on a walk-forward result object with .fold_train_test_bounds
# -----------------------------------------------------------------------------


def test_gantt_walkforward_render_with_minimal_input():
    """The gantt render coerces whatever it gets into a DataFrame, so a
    plain DataFrame with fold rows is the most defensive test."""
    from app.plots import gantt_walkforward

    folds = pd.DataFrame(
        {
            "fold": [0, 1, 2],
            "train_start": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01"]),
            "train_end": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30"]),
            "test_start": pd.to_datetime(["2020-04-01", "2020-07-01", "2020-10-01"]),
            "test_end": pd.to_datetime(["2020-06-30", "2020-09-30", "2020-12-31"]),
        }
    )
    fig = gantt_walkforward.render(folds)
    assert isinstance(fig, go.Figure)
