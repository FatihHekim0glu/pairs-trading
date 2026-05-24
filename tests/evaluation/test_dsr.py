"""Tests for the deflated / probabilistic Sharpe ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from pairs.evaluation import (
    deflated_sharpe_ratio,
    effective_n_trials,
    probabilistic_sharpe_ratio,
)


def test_psr_formula_known_value() -> None:
    # Gaussian returns, SR=0.5, benchmark=0, n=252.
    n = 252
    sr_hat = 0.5
    psr = probabilistic_sharpe_ratio(sr_hat=sr_hat, sr_benchmark=0.0, n=n)
    # Expected = Phi(sr_hat * sqrt(n-1) / sqrt(1 + sr_hat^2/4 * 2))... compute directly.
    denom = np.sqrt(1.0 - 0.0 + (3.0 - 1.0) / 4.0 * sr_hat * sr_hat)
    expected = float(stats.norm.cdf(sr_hat * np.sqrt(n - 1) / denom))
    assert psr == pytest.approx(expected, rel=1e-10)


@given(
    n_trials=st.integers(min_value=2, max_value=50),
)
@settings(max_examples=20, deadline=None)
def test_dsr_monotone_decreasing_in_trials(n_trials: int) -> None:
    base = deflated_sharpe_ratio(
        realized_sr=0.10,
        n_trials_eff=float(n_trials),
        sr_trial_variance=0.01,
        sample_size=252,
    )
    more = deflated_sharpe_ratio(
        realized_sr=0.10,
        n_trials_eff=float(n_trials + 5),
        sr_trial_variance=0.01,
        sample_size=252,
    )
    # More trials => higher threshold => lower DSR.
    assert more.dsr <= base.dsr + 1e-9


def test_effective_n_trials_perfectly_correlated_is_one() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=300)
    df = pd.DataFrame({f"c{i}": base for i in range(5)})
    n_eff = effective_n_trials(df)
    assert n_eff == pytest.approx(1.0, abs=1e-6)


def test_effective_n_trials_orthogonal_is_n() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(5000, 8)), columns=[f"c{i}" for i in range(8)])
    n_eff = effective_n_trials(df)
    assert n_eff == pytest.approx(8.0, rel=0.1)
