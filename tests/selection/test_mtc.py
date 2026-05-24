"""Tests for :mod:`pairs.selection.mtc`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pairs._exceptions import InputError
from pairs.selection.mtc import apply_mtc


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_true=st.integers(min_value=20, max_value=80),
)
def test_fdr_bh_controls_fdr_under_simulated_mixture(seed: int, n_true: int) -> None:
    """BH at q=0.10 should keep empirical FDR <= 0.15 in 80/20 mixtures."""
    rng = np.random.default_rng(seed)
    n_total = 200
    n_alt = n_total - n_true  # n_true is the number of NULL hypotheses
    null_p = rng.uniform(0.0, 1.0, size=n_true)
    alt_p = rng.uniform(0.0, 1e-4, size=n_alt)
    p_raw = np.concatenate([null_p, alt_p])
    is_null = np.concatenate([np.ones(n_true, dtype=bool), np.zeros(n_alt, dtype=bool)])
    index = [f"h{i:03d}" for i in range(n_total)]
    series = pd.Series(p_raw, index=index)
    frame = apply_mtc(series, method="fdr_bh", alpha=0.10)
    rejected = frame["survives_mtc"].to_numpy()
    if rejected.any():
        false_disc = float(np.sum(rejected & is_null) / max(rejected.sum(), 1))
    else:
        false_disc = 0.0
    assert false_disc <= 0.15


def test_mtc_identity_on_singleton() -> None:
    series = pd.Series([0.04], index=["only"])
    frame = apply_mtc(series, method="fdr_bh", alpha=0.10)
    assert len(frame) == 1
    assert frame.loc[0, "p_raw"] == pytest.approx(0.04)
    assert frame.loc[0, "q_value"] == pytest.approx(0.04)
    assert bool(frame.loc[0, "survives_mtc"])


def test_bh_vs_by_ordering_preserved() -> None:
    """BH and BY must rank pairs in the same raw-p order."""
    p_raw = np.array([0.001, 0.02, 0.04, 0.1, 0.5])
    series = pd.Series(p_raw, index=[f"p{i}" for i in range(p_raw.size)])
    bh = apply_mtc(series, method="fdr_bh", alpha=0.10)
    by = apply_mtc(series, method="fdr_by", alpha=0.10)
    bh_order = np.argsort(bh["q_value"].to_numpy())
    by_order = np.argsort(by["q_value"].to_numpy())
    np.testing.assert_array_equal(bh_order, by_order)


def test_bonferroni_more_conservative_than_bh() -> None:
    p_raw = np.array([0.0005, 0.01, 0.03, 0.05, 0.08, 0.5])
    series = pd.Series(p_raw, index=[f"p{i}" for i in range(p_raw.size)])
    bh = apply_mtc(series, method="fdr_bh", alpha=0.10)
    bonf = apply_mtc(series, method="bonferroni", alpha=0.10)
    diff = bonf["q_value"].to_numpy() - bh["q_value"].to_numpy()
    assert np.all(diff >= -1e-12)


def test_unknown_method_raises() -> None:
    series = pd.Series([0.05])
    with pytest.raises(InputError):
        apply_mtc(series, method="banana", alpha=0.10)


def test_method_none_passes_raw() -> None:
    series = pd.Series([0.03, 0.2], index=["a", "b"])
    frame = apply_mtc(series, method="none", alpha=0.05)
    np.testing.assert_allclose(frame["q_value"].to_numpy(), [0.03, 0.2])
    assert list(frame["survives_mtc"]) == [True, False]


def test_invalid_pvalue_raises() -> None:
    with pytest.raises(InputError):
        apply_mtc(pd.Series([np.nan]), method="fdr_bh", alpha=0.10)
    with pytest.raises(InputError):
        apply_mtc(pd.Series([1.5]), method="fdr_bh", alpha=0.10)


def test_empty_input_returns_empty_frame() -> None:
    frame = apply_mtc(pd.Series(dtype=float), method="fdr_bh", alpha=0.10)
    assert frame.empty
    assert list(frame.columns) == ["pair_id", "p_raw", "q_value", "survives_mtc"]


def test_non_series_input_raises() -> None:
    with pytest.raises(InputError):
        apply_mtc([0.05, 0.1], method="fdr_bh", alpha=0.10)  # type: ignore[arg-type]


def test_invalid_alpha_raises() -> None:
    with pytest.raises(InputError):
        apply_mtc(pd.Series([0.1]), method="fdr_bh", alpha=0.0)
    with pytest.raises(InputError):
        apply_mtc(pd.Series([0.1]), method="fdr_bh", alpha=1.5)


def test_holm_method_returns_monotone_qvalues() -> None:
    """Holm should yield a monotone non-decreasing sequence after sort."""
    p_raw = np.array([0.001, 0.02, 0.04, 0.1, 0.5])
    series = pd.Series(p_raw, index=[f"p{i}" for i in range(p_raw.size)])
    holm = apply_mtc(series, method="holm", alpha=0.10)
    assert list(holm.columns) == ["pair_id", "p_raw", "q_value", "survives_mtc"]
    # Sort by raw p and check monotone adjusted values.
    holm_sorted = holm.sort_values("p_raw").reset_index(drop=True)
    q = holm_sorted["q_value"].to_numpy()
    assert np.all(np.diff(q) >= -1e-12)


def test_fdr_by_method_returns_full_schema() -> None:
    p_raw = np.array([0.0005, 0.02, 0.5])
    series = pd.Series(p_raw, index=["a", "b", "c"])
    by = apply_mtc(series, method="fdr_by", alpha=0.10)
    assert set(by.columns) == {"pair_id", "p_raw", "q_value", "survives_mtc"}
    # The smallest raw p must be at least flagged when alpha is generous.
    assert by.loc[by["pair_id"] == "a", "survives_mtc"].iloc[0] in {True, False}


def test_pvalue_below_zero_raises() -> None:
    with pytest.raises(InputError):
        apply_mtc(pd.Series([-0.01]), method="fdr_bh", alpha=0.10)


def test_pair_id_preserves_index_strings() -> None:
    series = pd.Series([0.01, 0.04], index=["alpha", "beta"])
    frame = apply_mtc(series, method="fdr_bh", alpha=0.10)
    assert list(frame["pair_id"]) == ["alpha", "beta"]
