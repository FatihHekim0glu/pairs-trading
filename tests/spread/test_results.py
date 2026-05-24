"""Validation-path tests for the frozen result dataclasses.

These tests exercise the ``__post_init__`` invariants that the regular
estimator code paths never trigger, and so are not otherwise covered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs._exceptions import InputError
from pairs.spread.results import (
    HalfLifeResult,
    HedgeResult,
    KalmanHedgeResult,
    OUDiagnostics,
    OUResult,
)


def _empty_series(name: str = "r") -> pd.Series:
    """Single-row series with a fixed name (avoids zero-length edge cases)."""

    return pd.Series([0.0], index=pd.RangeIndex(1), name=name)


def test_hedge_result_rejects_r_squared_out_of_range() -> None:
    with pytest.raises(InputError):
        HedgeResult(
            alpha=0.0,
            beta=1.0,
            residuals=_empty_series(),
            r_squared=1.5,
            method="ols",
            direction="y~x",
            use_log=True,
            n_obs=1,
        )
    with pytest.raises(InputError):
        HedgeResult(
            alpha=0.0,
            beta=1.0,
            residuals=_empty_series(),
            r_squared=-0.01,
            method="ols",
            direction="y~x",
            use_log=True,
            n_obs=1,
        )


def test_hedge_result_rejects_bad_method() -> None:
    with pytest.raises(InputError):
        HedgeResult(
            alpha=0.0,
            beta=1.0,
            residuals=_empty_series(),
            r_squared=0.5,
            method="lasso",  # type: ignore[arg-type]
            direction="y~x",
            use_log=True,
            n_obs=1,
        )


def test_hedge_result_rejects_non_positive_n_obs() -> None:
    with pytest.raises(InputError):
        HedgeResult(
            alpha=0.0,
            beta=1.0,
            residuals=_empty_series(),
            r_squared=0.5,
            method="ols",
            direction="y~x",
            use_log=True,
            n_obs=0,
        )


def test_hedge_result_accepts_r_squared_at_one_with_tolerance() -> None:
    # The post-init permits 1 + 1e-9 to absorb FP noise from statsmodels.
    res = HedgeResult(
        alpha=0.0,
        beta=1.0,
        residuals=_empty_series(),
        r_squared=1.0 + 5e-10,
        method="tls",
        direction="y~x",
        use_log=False,
        n_obs=10,
    )
    assert res.method == "tls"


def _make_ou(**overrides: float) -> dict:
    base = dict(
        theta=0.1,
        mu=0.0,
        sigma=1.0,
        sigma_eq=1.0,
        half_life=7.0,
        phi=0.9,
        intercept=0.0,
        residuals=_empty_series("r_ou"),
        log_likelihood=0.0,
        dt=1.0,
        n_obs=1,
    )
    base.update(overrides)
    return base


def test_ou_result_rejects_non_positive_sigma() -> None:
    with pytest.raises(InputError):
        OUResult(**_make_ou(sigma=0.0))


def test_ou_result_rejects_phi_outside_unit_interval() -> None:
    with pytest.raises(InputError):
        OUResult(**_make_ou(phi=0.0))
    with pytest.raises(InputError):
        OUResult(**_make_ou(phi=1.0))
    with pytest.raises(InputError):
        OUResult(**_make_ou(phi=-0.1))


def test_ou_result_rejects_non_positive_dt() -> None:
    with pytest.raises(InputError):
        OUResult(**_make_ou(dt=0.0))
    with pytest.raises(InputError):
        OUResult(**_make_ou(dt=-1.0))


def test_ou_result_clamps_theta_low() -> None:
    res = OUResult(**_make_ou(theta=1e-12))
    assert res.theta == pytest.approx(1e-6)


def test_ou_result_clamps_theta_high() -> None:
    res = OUResult(**_make_ou(theta=1_000.0))
    assert res.theta == pytest.approx(100.0)


def test_half_life_result_rejects_point_outside_ci() -> None:
    with pytest.raises(InputError):
        HalfLifeResult(
            point=10.0, ci_lower=12.0, ci_upper=15.0, ci_level=0.95,
            n_boot=100, method="bootstrap",
        )
    with pytest.raises(InputError):
        HalfLifeResult(
            point=20.0, ci_lower=12.0, ci_upper=15.0, ci_level=0.95,
            n_boot=100, method="bootstrap",
        )


def test_half_life_result_rejects_ci_level_outside_unit_open_interval() -> None:
    with pytest.raises(InputError):
        HalfLifeResult(
            point=10.0, ci_lower=5.0, ci_upper=15.0, ci_level=0.0,
            n_boot=100, method="bootstrap",
        )
    with pytest.raises(InputError):
        HalfLifeResult(
            point=10.0, ci_lower=5.0, ci_upper=15.0, ci_level=1.0,
            n_boot=100, method="bootstrap",
        )


def test_half_life_result_rejects_negative_n_boot() -> None:
    with pytest.raises(InputError):
        HalfLifeResult(
            point=10.0, ci_lower=5.0, ci_upper=15.0, ci_level=0.95,
            n_boot=-1, method="bootstrap",
        )


def _kalman_payload(**overrides) -> dict:
    n = 3
    idx = pd.RangeIndex(n)
    base = dict(
        beta_series=pd.Series(np.zeros(n), index=idx, name="b"),
        alpha_series=pd.Series(np.zeros(n), index=idx, name="a"),
        dynamic_spread=pd.Series(np.zeros(n), index=idx, name="s"),
        dynamic_zscore=pd.Series(np.zeros(n), index=idx, name="z"),
        innovations=pd.Series(np.zeros(n), index=idx, name="i"),
        log_likelihood=0.0,
        delta=1e-3,
        backend="numpy",
    )
    base.update(overrides)
    return base


def test_kalman_result_rejects_bad_backend() -> None:
    with pytest.raises(InputError):
        KalmanHedgeResult(**_kalman_payload(backend="theano"))  # type: ignore[arg-type]


def test_kalman_result_rejects_non_positive_delta() -> None:
    with pytest.raises(InputError):
        KalmanHedgeResult(**_kalman_payload(delta=0.0))
    with pytest.raises(InputError):
        KalmanHedgeResult(**_kalman_payload(delta=-1e-3))


def test_ou_diagnostics_passed_property_round_trip() -> None:
    passing = OUDiagnostics(
        phi_significance_pvalue=0.01,
        adf_pvalue=0.01,
        ljung_box_pvalue=0.5,
        half_life_to_sample_ratio=0.05,
        reject_reason=None,
    )
    failing = OUDiagnostics(
        phi_significance_pvalue=0.01,
        adf_pvalue=0.01,
        ljung_box_pvalue=0.5,
        half_life_to_sample_ratio=0.5,
        reject_reason="half_life_too_long",
    )
    assert passing.passed is True
    assert failing.passed is False
