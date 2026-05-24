"""Tests for :mod:`pairs.selection.effective_n`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs._exceptions import InputError, InsufficientDataError
from pairs._rng import default_rng
from pairs.selection.effective_n import estimate_effective_n


def test_effective_n_perfectly_correlated_is_one() -> None:
    rng = default_rng(seed=20260526)
    base = rng.standard_normal(500)
    frame = pd.DataFrame({f"s{i}": base for i in range(5)})
    n_pca = estimate_effective_n(frame, method="pca")
    n_avg = estimate_effective_n(frame, method="avg_corr")
    assert n_pca == pytest.approx(1.0, abs=1e-6)
    assert n_avg == pytest.approx(1.0, abs=1e-6)


def test_effective_n_independent_is_n() -> None:
    rng = default_rng(seed=20260527)
    frame = pd.DataFrame(rng.standard_normal((400, 20)))
    n_pca = estimate_effective_n(frame, method="pca")
    n_avg = estimate_effective_n(frame, method="avg_corr")
    # Closed-form is exactly N when correlations vanish; sample noise leaves
    # us with a value close to but not equal to N.
    assert 15.0 < n_pca <= 20.0
    assert 15.0 < n_avg <= 20.0


def test_effective_n_intermediate_closed_form() -> None:
    """For an exact rho-bar = 0.5, N=4 ; N/(1+(N-1)*0.5) = 1.6."""
    n = 4
    corr = np.full((n, n), 0.5)
    np.fill_diagonal(corr, 1.0)
    # Build a sample with exactly this empirical correlation via Cholesky.
    rng = default_rng(seed=20260528)
    chol = np.linalg.cholesky(corr)
    samples = rng.standard_normal((20_000, n)) @ chol.T
    frame = pd.DataFrame(samples, columns=[f"s{i}" for i in range(n)])
    n_avg = estimate_effective_n(frame, method="avg_corr")
    assert n_avg == pytest.approx(1.6, rel=0.1)


def test_single_column_returns_one() -> None:
    frame = pd.DataFrame({"only": [0.1, 0.2, 0.3]})
    assert estimate_effective_n(frame) == 1.0


def test_unknown_method_raises() -> None:
    frame = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    with pytest.raises(InputError):
        estimate_effective_n(frame, method="bogus")  # type: ignore[arg-type]


def test_non_dataframe_raises() -> None:
    with pytest.raises(InputError):
        estimate_effective_n(np.zeros((10, 2)))  # type: ignore[arg-type]


def test_too_few_rows_raises() -> None:
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(InsufficientDataError):
        estimate_effective_n(frame)


def test_zero_columns_raises() -> None:
    with pytest.raises(InputError):
        estimate_effective_n(pd.DataFrame(index=[0, 1, 2]))


def test_pca_handles_constant_columns() -> None:
    """Constant columns produce NaN in DataFrame.corr; the helper substitutes I."""
    rng = default_rng(seed=20260529)
    n = 4
    arr = rng.standard_normal((200, n))
    arr[:, 0] = 1.0  # constant
    frame = pd.DataFrame(arr, columns=[f"s{i}" for i in range(n)])
    n_eff = estimate_effective_n(frame, method="pca")
    assert 1.0 <= n_eff <= float(n)


def test_avg_corr_handles_constant_columns() -> None:
    rng = default_rng(seed=20260530)
    n = 4
    arr = rng.standard_normal((200, n))
    arr[:, 0] = 1.0
    frame = pd.DataFrame(arr, columns=[f"s{i}" for i in range(n)])
    n_eff = estimate_effective_n(frame, method="avg_corr")
    assert 1.0 <= n_eff <= float(n)
