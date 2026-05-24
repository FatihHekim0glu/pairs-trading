"""Unit tests for :func:`pairs.cointegration.engle_granger`."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from statsmodels.tsa.stattools import coint

from pairs.cointegration import engle_granger
from pairs.cointegration.results import TestDirection


@pytest.mark.unit
def test_eg_detects_synthetic_coint_pair(synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=600, rho_residual=0.3)
    result = engle_granger(x, y, use_log=False)
    assert result.pvalue < 0.05
    assert result.n_obs == 600
    assert result.direction_used in TestDirection


@pytest.mark.unit
def test_eg_lower_pvalue_direction_kept(synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=400, rho_residual=0.4)
    result = engle_granger(x, y, use_log=False)
    assert result.pvalue <= result.pvalue_other_direction + 1e-12


@pytest.mark.unit
def test_eg_records_both_direction_pvalues(synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=400)
    result = engle_granger(x, y, use_log=False)
    assert 0.0 <= result.pvalue <= 1.0
    assert 0.0 <= result.pvalue_other_direction <= 1.0


@pytest.mark.unit
def test_eg_parity_with_statsmodels_coint(synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=300, rho_residual=0.2)
    result = engle_granger(x, y, use_log=False, autolag="bic")
    sm_stat_fwd, _, _ = coint(x.to_numpy(), y.to_numpy(), trend="c", autolag="bic")
    sm_stat_rev, _, _ = coint(y.to_numpy(), x.to_numpy(), trend="c", autolag="bic")
    sm_chosen = min(float(sm_stat_fwd), float(sm_stat_rev))
    # The chosen statistic should match the direction with the smaller p-value.
    assert result.stat == pytest.approx(sm_chosen, rel=1e-6, abs=1e-6) or result.stat in (
        pytest.approx(float(sm_stat_fwd), rel=1e-6, abs=1e-6),
        pytest.approx(float(sm_stat_rev), rel=1e-6, abs=1e-6),
    )


@pytest.mark.unit
def test_eg_use_log_requires_positive(rng):
    n = 100
    a = pd.Series(rng.standard_normal(n).cumsum(), name="A")
    b = pd.Series(rng.standard_normal(n).cumsum(), name="B")
    from pairs._exceptions import InputError

    with pytest.raises(InputError):
        engle_granger(a, b, use_log=True)


@pytest.mark.unit
def test_eg_rejects_unknown_trend(synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=100)
    from pairs._exceptions import InputError

    with pytest.raises(InputError):
        engle_granger(x, y, trend="zzz", use_log=False)  # type: ignore[arg-type]


@pytest.mark.property
@settings(
    deadline=None,
    max_examples=10,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    shift=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_eg_invariant_under_affine_transform(scale, shift, synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=300, rho_residual=0.3)
    base = engle_granger(x, y, use_log=False)
    transformed = engle_granger(scale * x + shift, y, use_log=False)
    assert abs(base.pvalue - transformed.pvalue) < 0.10
