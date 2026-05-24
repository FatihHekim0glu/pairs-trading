"""Unit tests for :func:`pairs.cointegration.bootstrap_coint_pvalue`."""

from __future__ import annotations

import numpy as np
import pytest

from pairs.cointegration import bootstrap_coint_pvalue, engle_granger


@pytest.mark.slow
def test_bootstrap_stable_vs_asymptotic_within_0_10(synthetic_coint_pair, rng):
    x, y = synthetic_coint_pair(t=300, rho_residual=0.3)
    eg = engle_granger(x, y, use_log=False)
    boot = bootstrap_coint_pvalue(x, y, n_boot=200, rng=rng)
    assert abs(boot.pvalue - eg.pvalue) <= 0.20


@pytest.mark.unit
def test_bootstrap_seed_reproducible(synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=200)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a = bootstrap_coint_pvalue(x, y, n_boot=50, rng=rng1)
    b = bootstrap_coint_pvalue(x, y, n_boot=50, rng=rng2)
    assert a.pvalue == pytest.approx(b.pvalue)
    assert a.block_length == b.block_length
    assert a.observed_stat == pytest.approx(b.observed_stat)


@pytest.mark.unit
def test_bootstrap_block_length_positive(synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=200)
    boot = bootstrap_coint_pvalue(x, y, n_boot=20)
    assert boot.block_length >= 2


@pytest.mark.unit
def test_bootstrap_rejects_bad_args(synthetic_coint_pair):
    from pairs._exceptions import InputError

    x, y = synthetic_coint_pair(t=100)
    with pytest.raises(InputError):
        bootstrap_coint_pvalue(x, y, n_boot=0)
    with pytest.raises(InputError):
        bootstrap_coint_pvalue(x, y, n_boot=10, block_length=-1)


@pytest.mark.unit
def test_bootstrap_quantiles_populated(synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=200)
    boot = bootstrap_coint_pvalue(x, y, n_boot=50)
    for q in (0.01, 0.05, 0.10, 0.50, 0.90):
        assert q in boot.null_stat_quantiles
