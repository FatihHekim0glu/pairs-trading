from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pairs.portfolio import EqualDollarAllocator, HRPAllocator, InverseVolAllocator


def _make_panel(rng: np.random.Generator, n_days: int = 200, n_pairs: int = 5) -> pd.DataFrame:
    data = rng.normal(scale=0.01, size=(n_days, n_pairs))
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    cols = [f"P{i}" for i in range(n_pairs)]
    return pd.DataFrame(data, index=idx, columns=cols)


@given(active_bits=st.lists(st.booleans(), min_size=5, max_size=5))
def test_equal_dollar_sums_to_one_over_active(active_bits: list[bool]) -> None:
    rng = np.random.default_rng(0)
    panel = _make_panel(rng)
    mask = pd.Series(active_bits, index=panel.columns)
    w = EqualDollarAllocator().weights(panel, mask)
    if mask.any():
        assert np.isclose(w.sum(), 1.0)
        n_active = int(mask.sum())
        np.testing.assert_allclose(w[mask].to_numpy(), 1.0 / n_active)
        np.testing.assert_allclose(w[~mask].to_numpy(), 0.0)
    else:
        assert np.isclose(w.sum(), 0.0)


@given(scale_factor=st.floats(min_value=1.5, max_value=5.0))
def test_inverse_vol_monotone(scale_factor: float) -> None:
    rng = np.random.default_rng(7)
    n = 200
    a = rng.normal(scale=0.01, size=n)
    b = rng.normal(scale=0.01 * scale_factor, size=n)
    idx = pd.bdate_range("2020-01-02", periods=n)
    panel = pd.DataFrame({"P0": a, "P1": b}, index=idx)
    mask = pd.Series([True, True], index=panel.columns)
    w = InverseVolAllocator(window=60, min_periods=20).weights(panel, mask)
    assert w["P0"] > w["P1"]
    expected_ratio = 1.0 / scale_factor
    actual_ratio = w["P1"] / w["P0"]
    assert actual_ratio < 1.0
    assert abs(actual_ratio - expected_ratio) < 0.3


def test_hrp_weights_simplex() -> None:
    rng = np.random.default_rng(11)
    panel = _make_panel(rng, n_days=200, n_pairs=6)
    mask = pd.Series([True] * 6, index=panel.columns)
    w = HRPAllocator(min_history=60).weights(panel, mask)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0.0).all()


def test_hrp_no_short() -> None:
    rng = np.random.default_rng(13)
    panel = _make_panel(rng, n_days=200, n_pairs=4)
    mask = pd.Series([True] * 4, index=panel.columns)
    w = HRPAllocator(min_history=60).weights(panel, mask)
    assert (w >= 0.0).all()


def test_inverse_vol_handles_nan() -> None:
    rng = np.random.default_rng(3)
    panel = _make_panel(rng, n_days=200, n_pairs=3)
    panel.iloc[:, 0] = np.nan  # insufficient history
    mask = pd.Series([True, True, True], index=panel.columns)
    w = InverseVolAllocator(window=60, min_periods=20).weights(panel, mask)
    assert w.iloc[0] == 0.0
    assert np.isclose(w.iloc[1:].sum(), 1.0)


def test_hrp_max_weight_cap() -> None:
    rng = np.random.default_rng(17)
    panel = _make_panel(rng, n_days=300, n_pairs=4)
    mask = pd.Series([True] * 4, index=panel.columns)
    w = HRPAllocator(min_history=60, max_weight=0.4).weights(panel, mask)
    assert (w <= 0.4 + 1e-9).all()
    assert np.isclose(w.sum(), 1.0)


def test_hrp_falls_back_with_short_history() -> None:
    rng = np.random.default_rng(19)
    panel = _make_panel(rng, n_days=20, n_pairs=4)
    mask = pd.Series([True] * 4, index=panel.columns)
    w = HRPAllocator(min_history=60).weights(panel, mask)
    # Falls back to equal dollar.
    np.testing.assert_allclose(w.to_numpy(), 0.25)


def test_allocator_rejects_misaligned_mask() -> None:
    rng = np.random.default_rng(23)
    panel = _make_panel(rng, n_days=100, n_pairs=3)
    bad_mask = pd.Series([True, True], index=["X", "Y"])
    with pytest.raises(Exception):  # noqa: B017
        EqualDollarAllocator().weights(panel, bad_mask)
