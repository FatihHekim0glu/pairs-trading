"""Tests for :mod:`pairs.selection.pre_screen`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pairs._exceptions import InputError
from pairs._rng import default_rng
from pairs.selection.pre_screen import apply_pre_screen
from pairs.selection.results import Candidate


def _make_panel(
    n_tickers: int = 6,
    T: int = 400,
    *,
    seed: int = 20260530,
) -> pd.DataFrame:
    rng = default_rng(seed=seed)
    index = pd.date_range("2020-01-01", periods=T, freq="B")
    latent = rng.standard_normal((T, 1))
    idio = rng.standard_normal((T, n_tickers))
    shocks = 0.7 * latent + 0.3 * idio
    log_prices = 4.0 + np.cumsum(shocks * 0.01, axis=0)
    prices = np.exp(log_prices)
    tickers = [f"T{ix:02d}" for ix in range(n_tickers)]
    return pd.DataFrame(prices, index=index, columns=tickers)


def _all_pairs(panel: pd.DataFrame, adv: float = 1e7) -> list[Candidate]:
    out: list[Candidate] = []
    cols = list(panel.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            out.append(Candidate(ticker_a=a, ticker_b=b, adv_a=adv, adv_b=adv))
    return out


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    floor_a=st.floats(min_value=1e6, max_value=1e7),
    floor_b=st.floats(min_value=1e7, max_value=1e9),
)
def test_monotone_in_adv_floor(seed: int, floor_a: float, floor_b: float) -> None:
    """Raising the ADV floor never increases the count of survivors."""
    rng = default_rng(seed=seed)
    panel = _make_panel(seed=seed)
    candidates: list[Candidate] = []
    for c in _all_pairs(panel):
        # Random per-pair ADV chosen from the broader bracket.
        adv = float(rng.uniform(1e5, 1e10))
        candidates.append(Candidate(ticker_a=c.ticker_a, ticker_b=c.ticker_b, adv_a=adv, adv_b=adv))
    window = (panel.index[0], panel.index[-1])
    low = min(floor_a, floor_b)
    high = max(floor_a, floor_b)
    n_low = len(apply_pre_screen(candidates, panel, formation_window=window, adv_floor=low))
    n_high = len(apply_pre_screen(candidates, panel, formation_window=window, adv_floor=high))
    assert n_high <= n_low


def test_every_reject_has_reason() -> None:
    panel = _make_panel()
    candidates = _all_pairs(panel, adv=1e9)
    window = (panel.index[0], panel.index[-1])
    annotated = apply_pre_screen(
        candidates,
        panel,
        formation_window=window,
        adv_floor=1e15,
        return_rejects=True,
    )
    for cand in annotated:
        assert cand.exclusion_reason, "rejected candidate must carry reason"


def test_counts_sum_to_input() -> None:
    panel = _make_panel()
    candidates = _all_pairs(panel)
    window = (panel.index[0], panel.index[-1])
    all_annotated = apply_pre_screen(
        candidates, panel, formation_window=window, return_rejects=True
    )
    passes_only = apply_pre_screen(candidates, panel, formation_window=window)
    passes = sum(1 for c in all_annotated if not c.exclusion_reason)
    assert len(all_annotated) == len(candidates)
    assert passes == len(passes_only)


def test_correlation_band_inclusive() -> None:
    """Boundary correlations should be accepted (the band is inclusive)."""
    panel = _make_panel()
    window = (panel.index[0], panel.index[-1])
    cands = _all_pairs(panel)
    # Use a very wide band that should accept all valid pairs.
    accepted = apply_pre_screen(
        cands, panel, formation_window=window, corr_band=(-1.0, 1.0), hurst_max=1.5
    )
    assert len(accepted) > 0
    # Now use the exact returns-correlation as both lower and upper bounds for
    # the first pair to exercise the inclusive ends.
    first = cands[0]
    sub = panel[[first.ticker_a, first.ticker_b]].pct_change().dropna()
    rho = float(sub.iloc[:, 0].corr(sub.iloc[:, 1]))
    one = apply_pre_screen(
        [first], panel, formation_window=window, corr_band=(rho, rho), hurst_max=1.5
    )
    assert len(one) == 1


def test_continuous_listing_excludes_missing_bars() -> None:
    panel = _make_panel()
    panel.iloc[10:20, 0] = np.nan
    window = (panel.index[0], panel.index[-1])
    cands = [Candidate(ticker_a="T00", ticker_b="T01", adv_a=1e9, adv_b=1e9)]
    annotated = apply_pre_screen(cands, panel, formation_window=window, return_rejects=True)
    assert annotated[0].exclusion_reason
    assert "continuous_listing" in annotated[0].exclusion_reason


def test_price_floor_excludes_penny_stocks() -> None:
    panel = _make_panel()
    panel.loc[:, "T00"] = 0.5
    window = (panel.index[0], panel.index[-1])
    cands = [Candidate(ticker_a="T00", ticker_b="T01", adv_a=1e9, adv_b=1e9)]
    annotated = apply_pre_screen(
        cands, panel, formation_window=window, return_rejects=True, price_floor=5.0
    )
    assert "price_floor" in annotated[0].exclusion_reason


def test_missing_ticker_recorded() -> None:
    panel = _make_panel()
    window = (panel.index[0], panel.index[-1])
    cands = [Candidate(ticker_a="ZZZ", ticker_b="T01", adv_a=1e9, adv_b=1e9)]
    annotated = apply_pre_screen(cands, panel, formation_window=window, return_rejects=True)
    assert "missing_ticker" in annotated[0].exclusion_reason


def test_invalid_window_raises() -> None:
    panel = _make_panel()
    with pytest.raises(InputError):
        apply_pre_screen(
            [],
            panel,
            formation_window=(panel.index[-1], panel.index[0]),
        )


def test_invalid_corr_band_raises() -> None:
    panel = _make_panel()
    with pytest.raises(InputError):
        apply_pre_screen(
            [],
            panel,
            formation_window=(panel.index[0], panel.index[-1]),
            corr_band=(0.9, 0.1),
        )


def test_non_dataframe_raises() -> None:
    with pytest.raises(InputError):
        apply_pre_screen(
            [],
            np.zeros((10, 2)),  # type: ignore[arg-type]
            formation_window=(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")),
        )


def test_adv_floor_independent_of_other_filters() -> None:
    """ADV floor alone should suffice to reject when other checks would pass."""
    panel = _make_panel()
    cands = [Candidate(ticker_a="T00", ticker_b="T01", adv_a=1.0, adv_b=1.0)]
    window = (panel.index[0], panel.index[-1])
    annotated = apply_pre_screen(
        cands,
        panel,
        formation_window=window,
        adv_floor=1e9,
        corr_band=(-1.0, 1.0),
        hurst_max=1.5,
        price_floor=0.0,
        return_rejects=True,
    )
    assert "adv_floor" in annotated[0].exclusion_reason


def test_price_floor_independent_of_other_filters() -> None:
    panel = _make_panel()
    panel.loc[:, "T00"] = 0.5
    panel.loc[:, "T01"] = 0.5
    cands = [Candidate(ticker_a="T00", ticker_b="T01", adv_a=1e9, adv_b=1e9)]
    window = (panel.index[0], panel.index[-1])
    annotated = apply_pre_screen(
        cands,
        panel,
        formation_window=window,
        price_floor=5.0,
        adv_floor=0.0,
        corr_band=(-1.0, 1.0),
        hurst_max=1.5,
        return_rejects=True,
    )
    assert "price_floor" in annotated[0].exclusion_reason


def test_correlation_band_rejects_outside() -> None:
    panel = _make_panel()
    cands = [Candidate(ticker_a="T00", ticker_b="T01", adv_a=1e9, adv_b=1e9)]
    window = (panel.index[0], panel.index[-1])
    # Demand near-perfect correlation; reject.
    annotated = apply_pre_screen(
        cands,
        panel,
        formation_window=window,
        adv_floor=0.0,
        price_floor=0.0,
        corr_band=(0.999, 1.0),
        hurst_max=1.5,
        return_rejects=True,
    )
    assert "correlation_band" in annotated[0].exclusion_reason


def test_adv_falls_through_when_none_and_no_volume() -> None:
    """Candidates without ADV metadata pass the ADV check (delegated upstream)."""
    panel = _make_panel()
    cands = [Candidate(ticker_a="T00", ticker_b="T01")]  # adv_a/adv_b == None
    window = (panel.index[0], panel.index[-1])
    annotated = apply_pre_screen(
        cands,
        panel,
        formation_window=window,
        adv_floor=1e15,
        price_floor=0.0,
        corr_band=(-1.0, 1.0),
        hurst_max=1.5,
        return_rejects=True,
    )
    assert "adv_floor" not in annotated[0].exclusion_reason


def test_hurst_max_rejects_random_walks() -> None:
    rng = default_rng(seed=42)
    T = 600
    index = pd.date_range("2020-01-01", periods=T, freq="B")
    rw_a = np.exp(4.0 + np.cumsum(rng.standard_normal(T) * 0.01))
    rw_b = np.exp(4.0 + np.cumsum(rng.standard_normal(T) * 0.01))
    panel = pd.DataFrame({"A": rw_a, "B": rw_b}, index=index)
    cands = [Candidate(ticker_a="A", ticker_b="B", adv_a=1e9, adv_b=1e9)]
    annotated = apply_pre_screen(
        cands,
        panel,
        formation_window=(panel.index[0], panel.index[-1]),
        corr_band=(-1.0, 1.0),
        hurst_max=0.05,  # extremely strict -> reject
        return_rejects=True,
    )
    # At least one reason should be present (likely hurst).
    assert annotated[0].exclusion_reason
