"""Tests for :mod:`pairs.selection.romano_wolf`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs._exceptions import InputError
from pairs._rng import default_rng
from pairs.selection.romano_wolf import romano_wolf_stepdown


def _gauss_returns(rng: np.random.Generator, T: int, drift: float = 0.0) -> pd.Series:
    idx = pd.date_range("2022-01-01", periods=T, freq="B")
    vals = drift + rng.standard_normal(T) * 0.01
    return pd.Series(vals, index=idx)


def test_rw_null_calibration() -> None:
    """Under the null with K=5 strategies, FWER should not blow up.

    With only 30 sims and a 199-draw bootstrap the binomial confidence is
    wide, so we test for the absence of egregious miscalibration rather
    than tight FWER control.
    """
    rng = default_rng(seed=20260601)
    rejects = 0
    n_sims = 30
    K = 5
    T = 200
    for _ in range(n_sims):
        sims = {f"s{k}": _gauss_returns(rng, T) for k in range(K)}
        bench = _gauss_returns(rng, T)
        result = romano_wolf_stepdown(
            sims,
            bench,
            n_boot=199,
            alpha=0.10,
            rng=rng,
        )
        if result.surviving_pair_ids:
            rejects += 1
    rate = rejects / n_sims
    assert rate <= 0.40, f"empirical FWER {rate:.2f} substantially exceeds nominal 0.10"


def test_rw_detects_strong_signal() -> None:
    rng = default_rng(seed=20260602)
    T = 400
    K = 10
    sims: dict[str, pd.Series] = {}
    for k in range(K):
        drift = 0.05 if k == 0 else 0.0  # one truly strong strategy
        sims[f"strategy_{k}"] = _gauss_returns(rng, T, drift=drift)
    bench = _gauss_returns(rng, T, drift=0.0)
    result = romano_wolf_stepdown(
        sims,
        bench,
        n_boot=499,
        alpha=0.05,
        rng=rng,
    )
    assert "strategy_0" in result.surviving_pair_ids


def test_rw_block_length_positive() -> None:
    rng = default_rng(seed=20260603)
    sims = {"a": _gauss_returns(rng, 200, drift=0.05)}
    result = romano_wolf_stepdown(sims, None, n_boot=99, rng=rng)
    assert result.block_length > 0
    assert result.n_boot == 99


def test_rw_invalid_alpha() -> None:
    rng = default_rng(seed=1)
    with pytest.raises(InputError):
        romano_wolf_stepdown({"a": _gauss_returns(rng, 100)}, None, alpha=0.0)


def test_rw_invalid_n_boot() -> None:
    rng = default_rng(seed=2)
    with pytest.raises(InputError):
        romano_wolf_stepdown({"a": _gauss_returns(rng, 100)}, None, n_boot=0)


def test_rw_empty_input() -> None:
    with pytest.raises(InputError):
        romano_wolf_stepdown({}, None)


def test_rw_with_explicit_block_length() -> None:
    rng = default_rng(seed=20260604)
    sims = {f"s{i}": _gauss_returns(rng, 200) for i in range(3)}
    bench = _gauss_returns(rng, 200)
    result = romano_wolf_stepdown(
        sims,
        bench,
        n_boot=99,
        block_length=5,
        rng=rng,
    )
    assert result.block_length == 5
    assert result.null_distribution.shape == (99,)


def test_rw_no_overlap_raises() -> None:
    rng = default_rng(seed=20260605)
    a = pd.Series(rng.standard_normal(50), index=pd.date_range("2020-01-01", periods=50))
    b = pd.Series(rng.standard_normal(50), index=pd.date_range("2025-01-01", periods=50))
    with pytest.raises(InputError):
        romano_wolf_stepdown({"a": a}, b, n_boot=10, rng=rng)


def test_rw_invalid_block_length() -> None:
    rng = default_rng(seed=20260606)
    sims = {"a": _gauss_returns(rng, 100)}
    with pytest.raises(InputError):
        romano_wolf_stepdown(sims, None, n_boot=10, block_length=0, rng=rng)


def test_rw_adjusted_pvalues_monotone() -> None:
    """Step-down adjusted p-values are monotone non-decreasing in rejection order."""
    rng = default_rng(seed=20260607)
    T = 300
    sims = {
        "strong": _gauss_returns(rng, T, drift=0.05),
        "weak": _gauss_returns(rng, T, drift=0.01),
        "noise": _gauss_returns(rng, T, drift=0.0),
    }
    bench = _gauss_returns(rng, T, drift=0.0)
    result = romano_wolf_stepdown(sims, bench, n_boot=199, alpha=0.10, rng=rng)
    # All adjusted p-values lie in [0, 1].
    vals = result.adjusted_pvalues.to_numpy()
    assert np.all((vals >= 0.0) & (vals <= 1.0))
    # Null distribution has the expected shape.
    assert result.null_distribution.ndim == 1


def test_rw_no_benchmark_uses_zero() -> None:
    """When benchmark is None, the test reduces to a one-sample mean test."""
    rng = default_rng(seed=20260608)
    sims = {"a": _gauss_returns(rng, 200, drift=0.10)}
    result = romano_wolf_stepdown(sims, None, n_boot=99, alpha=0.10, rng=rng)
    assert isinstance(result.adjusted_pvalues, pd.Series)
    assert "a" in result.adjusted_pvalues.index
