"""Tests for the bootstrap half-life CI."""

from __future__ import annotations

import numpy as np
import pytest

from pairs._exceptions import InputError
from pairs._rng import default_rng
from pairs.spread.half_life import half_life


@pytest.mark.slow
def test_bootstrap_ci_covers_true_value(simulated_ou) -> None:
    theta_true = 0.05
    true_hl = float(np.log(2.0) / theta_true)
    n_mc = 60
    covered = 0
    for k in range(n_mc):
        rng = default_rng(2000 + k)
        spread = simulated_ou(rng, theta=theta_true, mu=0.0, sigma=1.0, n=1000)
        res = half_life(spread, n_boot=200, rng=default_rng(3000 + k))
        if res.ci_lower <= true_hl <= res.ci_upper:
            covered += 1
    coverage = covered / n_mc
    assert coverage >= 0.80


def test_bootstrap_determinism_with_seed(simulated_ou) -> None:
    rng = default_rng(50)
    spread = simulated_ou(rng, theta=0.05, n=600)
    r1 = half_life(spread, n_boot=50, rng=default_rng(7))
    r2 = half_life(spread, n_boot=50, rng=default_rng(7))
    assert r1.ci_lower == pytest.approx(r2.ci_lower, rel=0, abs=1e-12)
    assert r1.ci_upper == pytest.approx(r2.ci_upper, rel=0, abs=1e-12)
    assert r1.point == pytest.approx(r2.point, rel=0, abs=1e-12)


def test_ci_lower_le_point_le_upper(simulated_ou) -> None:
    rng = default_rng(60)
    spread = simulated_ou(rng, theta=0.05, n=800)
    res = half_life(spread, n_boot=80, rng=default_rng(8))
    assert res.ci_lower <= res.point <= res.ci_upper
    assert res.ci_level == 0.95
    assert res.method == "bootstrap"


def test_invalid_ci_method(simulated_ou) -> None:
    rng = default_rng(61)
    spread = simulated_ou(rng, theta=0.05, n=400)
    with pytest.raises(InputError):
        half_life(spread, ci_method="profile")


def test_invalid_n_boot(simulated_ou) -> None:
    rng = default_rng(62)
    spread = simulated_ou(rng, theta=0.05, n=400)
    with pytest.raises(InputError):
        half_life(spread, n_boot=1)
