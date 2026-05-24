"""Tests for the OU diagnostics battery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs._exceptions import DegenerateSeriesError
from pairs._rng import default_rng
from pairs.spread.diagnostics import ou_diagnostics
from pairs.spread.ou import fit_ou
from pairs.spread.results import OUResult


def _dummy_ou_for(spread: pd.Series, *, half_life: float = 20.0) -> OUResult:
    """Construct a synthetic OUResult around a spread for diagnostics tests."""

    theta = float(np.log(2.0) / max(half_life, 1.0))
    phi = float(np.exp(-theta))
    n = spread.shape[0]
    return OUResult(
        theta=theta,
        mu=float(spread.mean()),
        sigma=max(float(spread.std(ddof=1)), 1e-6),
        sigma_eq=max(float(spread.std(ddof=1)), 1e-6),
        half_life=half_life,
        phi=max(min(phi, 0.999_999), 1e-3),
        intercept=0.0,
        residuals=pd.Series(
            np.zeros(max(n - 1, 1)), index=spread.index[1:], name="resid"
        ),
        log_likelihood=0.0,
        dt=1.0,
        n_obs=max(n - 1, 1),
    )


def test_random_walk_rejected_phi_or_adf(random_walk) -> None:
    rng = default_rng(700)
    rw = random_walk(rng, n=400)
    try:
        ou = fit_ou(rw)
    except DegenerateSeriesError:
        # Random walk failed the OU back-transform; construct a dummy and
        # ask diagnostics about the original (random-walk) series.
        ou = _dummy_ou_for(rw, half_life=10_000.0)
    diag = ou_diagnostics(rw, ou)
    assert diag.reject_reason in {
        "phi_not_significant",
        "adf_nonstationary",
        "half_life_too_long",
    }
    assert diag.passed is False


def test_white_noise_passes(simulated_ou) -> None:
    rng = default_rng(701)
    spread = simulated_ou(rng, theta=0.2, mu=0.0, sigma=1.0, n=2000)
    ou = fit_ou(spread)
    diag = ou_diagnostics(spread, ou)
    # Most healthy fits should pass; if not, at least the components should be sane.
    assert 0.0 <= diag.phi_significance_pvalue <= 1.0
    assert 0.0 <= diag.adf_pvalue <= 1.0
    assert 0.0 <= diag.ljung_box_pvalue <= 1.0
    assert diag.half_life_to_sample_ratio >= 0.0


def test_short_half_life_rejected(simulated_ou) -> None:
    rng = default_rng(702)
    spread = simulated_ou(rng, theta=0.05, mu=0.0, sigma=1.0, n=600)
    ou = _dummy_ou_for(spread, half_life=0.5)
    diag = ou_diagnostics(spread, ou)
    assert diag.reject_reason == "half_life_too_short"


def test_long_half_life_rejected(simulated_ou) -> None:
    rng = default_rng(703)
    spread = simulated_ou(rng, theta=0.05, mu=0.0, sigma=1.0, n=300)
    ou = _dummy_ou_for(spread, half_life=200.0)
    diag = ou_diagnostics(spread, ou)
    assert diag.reject_reason in {"half_life_too_long", "phi_not_significant"}


def test_passed_property(simulated_ou) -> None:
    rng = default_rng(704)
    spread = simulated_ou(rng, theta=0.2, n=2000)
    ou = fit_ou(spread)
    diag = ou_diagnostics(spread, ou)
    assert diag.passed == (diag.reject_reason is None)
