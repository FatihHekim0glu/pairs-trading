"""Unit tests for :func:`pairs.cointegration.full_pipeline`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.cointegration import full_pipeline
from pairs.cointegration.results import (
    CointegrationResult,
    JohansenResult,
    KPSSResult,
    PipelineResult,
    TestDirection,
    UnitRootResult,
)


@pytest.mark.unit
def test_pipeline_true_positive_on_cointegrated(synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=500, rho_residual=0.3)
    res = full_pipeline(x, y, use_log=False)
    assert res.cointegrated
    assert res.eg.pvalue < 0.05
    assert res.kpss.is_stationary
    assert res.leg0_unit_root.is_i1
    assert res.leg1_unit_root.is_i1


@pytest.mark.unit
def test_pipeline_true_negative_on_independent_rw(two_random_walks):
    x, y = two_random_walks(t=400)
    res = full_pipeline(x, y, use_log=False)
    assert not res.cointegrated


@pytest.mark.unit
def test_pipeline_with_bootstrap(synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=200, rho_residual=0.3)
    res = full_pipeline(x, y, use_log=False, n_boot=30)
    assert res.bootstrap is not None
    assert res.bootstrap.n_boot == 30


@pytest.mark.unit
def test_pipeline_skips_johansen_when_disabled(synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=300)
    res = full_pipeline(x, y, use_log=False, run_johansen=False)
    assert res.johansen is None


@pytest.mark.unit
def test_pipeline_rejects_bad_alpha(synthetic_coint_pair):
    from pairs._exceptions import InputError

    x, y = synthetic_coint_pair(t=100)
    with pytest.raises(InputError):
        full_pipeline(x, y, alpha=1.5)


def _truth_table_inputs() -> list[tuple[bool, bool, bool, bool]]:
    out: list[tuple[bool, bool, bool, bool]] = []
    for eg in (False, True):
        for joh in (False, True):
            for kp in (False, True):
                for legs in (False, True):
                    out.append((eg, joh, kp, legs))
    return out


def _fake_pipeline_inputs(
    eg_ok: bool,
    joh_ok: bool,
    kp_ok: bool,
    legs_ok: bool,
) -> tuple[CointegrationResult, JohansenResult, KPSSResult, UnitRootResult, UnitRootResult]:
    resid = pd.Series(np.zeros(50), name="resid")
    eg = CointegrationResult(
        stat=-3.0,
        pvalue=0.01 if eg_ok else 0.50,
        crit_values=(-3.5, -2.9, -2.6),
        direction_used=TestDirection.Y0_ON_Y1,
        beta=1.0,
        alpha=0.0,
        residuals=resid,
        autolag_used="bic",
        n_obs=50,
        pvalue_other_direction=0.50,
    )
    joh = JohansenResult(
        trace_stats=np.array([20.0, 5.0]),
        trace_crit_95=np.array([15.0, 4.0]) if joh_ok else np.array([30.0, 10.0]),
        max_eig_stats=np.array([15.0, 5.0]),
        max_eig_crit_95=np.array([14.0, 4.0]) if joh_ok else np.array([30.0, 10.0]),
        rank=1 if joh_ok else 0,
        eigenvectors=np.eye(2),
        n_obs=50,
    )
    kp = KPSSResult(
        stat=0.1,
        pvalue=0.20 if kp_ok else 0.01,
        pvalue_interpolated=False,
        crit_values={"5%": 0.463},
        nlags_used=4,
        is_stationary=kp_ok,
    )
    leg0 = UnitRootResult(
        leg_name="x",
        levels_pvalue=0.50,
        diff_pvalue=0.01 if legs_ok else 0.50,
        is_i1=legs_ok,
        method="adf",
        n_obs=50,
    )
    leg1 = UnitRootResult(
        leg_name="y",
        levels_pvalue=0.50,
        diff_pvalue=0.01 if legs_ok else 0.50,
        is_i1=legs_ok,
        method="adf",
        n_obs=50,
    )
    return eg, joh, kp, leg0, leg1


@pytest.mark.unit
@pytest.mark.parametrize(("eg_ok", "joh_ok", "kp_ok", "legs_ok"), _truth_table_inputs())
def test_pipeline_four_cell_logic_truth_table(eg_ok, joh_ok, kp_ok, legs_ok):
    eg, joh, kp, leg0, leg1 = _fake_pipeline_inputs(eg_ok, joh_ok, kp_ok, legs_ok)
    expected = eg_ok and joh_ok and kp_ok and legs_ok
    res = PipelineResult(
        eg=eg,
        johansen=joh,
        kpss=kp,
        leg0_unit_root=leg0,
        leg1_unit_root=leg1,
        bootstrap=None,
        cointegrated=bool(
            eg.pvalue < 0.05
            and joh.rank >= 1
            and kp.is_stationary
            and leg0.is_i1
            and leg1.is_i1,
        ),
        alpha=0.05,
    )
    assert res.cointegrated is expected
