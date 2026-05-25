"""Smoke tests for the Streamlit dashboard adapter (`app/cache.py`).

These tests live OUTSIDE `tests/unit/` because they exercise the dashboard
adapter, not the library proper. They are the layer that should have caught
the dashboard-vs-library contract drift up front.

We import the adapter without invoking Streamlit (the `@st.cache_data` /
`@st.cache_resource` decorators are no-ops at import time — they only fire
inside a real Streamlit script-runner context — so the underlying functions
can be smoke-tested directly).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# `app/` lives at the project root, not under `src/`, so the adapter import
# requires the project root to be on sys.path. The Streamlit entrypoint does
# this at runtime; we mirror it here.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


_streamlit = pytest.importorskip("streamlit", reason="streamlit not installed (in [app] extras)")


def _make_yf_like_frame(tickers: list[str], n: int = 10) -> pd.DataFrame:
    """Replica of `pairs.data.load_prices`'s MultiIndex column shape using
    the same yfinance-native field names (`Open`, `High`, `Low`, `Close`,
    `Adj Close`, `Volume`). This is the schema the adapter must accept."""
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    cols = pd.MultiIndex.from_product(
        [tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]],
        names=["ticker", "field"],
    )
    return pd.DataFrame(
        data=1.0
        + 0.0 * (idx.values.reshape(-1, 1) == idx.values.reshape(-1, 1)).repeat(len(cols), axis=1),
        index=idx,
        columns=cols,
    )


# -----------------------------------------------------------------------------
# _flatten_prices field-name handling
# -----------------------------------------------------------------------------


def test_flatten_prices_handles_yfinance_native_close():
    """The yfinance loader emits `Close` (capitalised). The adapter speaks
    `close` (lowercase). Translation must succeed AND the tz must be stripped
    so downstream library code (which assumes naive indices) can compare
    against naive Timestamp formation-window bounds."""
    from app.cache import _flatten_prices

    raw = _make_yf_like_frame(["KO", "PEP"])
    wide = _flatten_prices(raw, field="close")
    assert list(wide.columns) == ["KO", "PEP"]
    assert wide.index.tz is None, "adapter must strip tz before returning to caller"
    # Index dates should match modulo tz.
    assert (wide.index == raw.index.tz_localize(None)).all()


def test_flatten_prices_handles_adj_close_alias():
    """Both lowercase `adj_close` and native `Adj Close` (with space) must
    resolve. Earlier dashboards used the snake-case form."""
    from app.cache import _flatten_prices

    raw = _make_yf_like_frame(["AAPL"])
    wide = _flatten_prices(raw, field="adj_close")
    assert list(wide.columns) == ["AAPL"]


def test_flatten_prices_falls_back_close_to_adj_close():
    """If only `Adj Close` is available, requesting `close` should still
    succeed via the alias chain instead of raising KeyError."""
    from app.cache import _flatten_prices

    idx = pd.date_range("2024-01-02", periods=5, freq="B", tz="UTC")
    raw = pd.DataFrame(
        1.0,
        index=idx,
        columns=pd.MultiIndex.from_product([["KO"], ["Adj Close"]], names=["ticker", "field"]),
    )
    wide = _flatten_prices(raw, field="close")
    assert list(wide.columns) == ["KO"]


def test_flatten_prices_raises_on_unknown_field():
    """Unknown fields should fail loud — not return an empty frame."""
    from app.cache import _flatten_prices

    raw = _make_yf_like_frame(["KO"])
    with pytest.raises(KeyError, match="price field"):
        _flatten_prices(raw, field="dividends")


def test_flatten_prices_passes_through_non_multiindex():
    """If the caller hands the adapter a wide frame already, leave it alone
    (other than the tz-strip applied to every output for consistency)."""
    from app.cache import _flatten_prices

    idx = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
    wide_in = pd.DataFrame({"KO": [1.0, 1.1, 1.2], "PEP": [2.0, 2.1, 2.2]}, index=idx)
    wide_out = _flatten_prices(wide_in)
    assert wide_out.index.tz is None
    assert list(wide_out.columns) == ["KO", "PEP"]
    assert (wide_out.values == wide_in.astype(float).sort_index().values).all()


# -----------------------------------------------------------------------------
# Sizing-alias contract: the UI strings must all map to real backtest sizings.
# -----------------------------------------------------------------------------


def test_sizing_alias_covers_all_ui_options():
    """Page 03 offers three sizing options; every one must map to a valid
    `backtest_pair` sizing keyword."""
    from app.cache import _SIZING_ALIAS

    ui_options = {"fixed_notional", "vol_target", "kelly_capped"}
    library_sizings = {"dollar_neutral", "beta_neutral", "unit"}

    for ui in ui_options:
        assert ui in _SIZING_ALIAS, f"UI option {ui!r} has no library mapping"
        assert _SIZING_ALIAS[ui] in library_sizings, (
            f"{ui!r} maps to {_SIZING_ALIAS[ui]!r}, which is not a valid backtest_pair sizing"
        )


# -----------------------------------------------------------------------------
# Universe-name contract: every option in page 01's selectbox must resolve.
# -----------------------------------------------------------------------------


def test_pair_finder_universe_options_all_load():
    """Every universe listed in the Pair Finder selectbox must actually load
    from `pairs.data`. Catches the kind of drift where someone adds a UI
    option without shipping the YAML."""
    from app.cache import _universe_pairs

    page_options = ["curated_25_v1", "xlk_v1"]
    for name in page_options:
        pairs_list = _universe_pairs(name)
        assert len(pairs_list) > 0, f"universe {name!r} resolved to zero pairs"
        for a, b in pairs_list:
            assert isinstance(a, str) and isinstance(b, str)
            assert a != b


# -----------------------------------------------------------------------------
# Cost-profile contract: every option in page 03's selectbox must load.
# -----------------------------------------------------------------------------


def test_backtest_cost_profile_options_all_load():
    """Every cost profile listed in the Backtest selectbox must be on disk."""
    from pairs.backtest import load_profile

    profiles = [
        "large_cap_optimistic",
        "large_cap_realistic",
        "mid_cap_realistic",
        "small_cap_conservative",
    ]
    for name in profiles:
        prof = load_profile(name)
        assert prof is not None
