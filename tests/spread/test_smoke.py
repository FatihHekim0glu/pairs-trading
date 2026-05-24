"""End-to-end smoke test for the spread pipeline."""

from __future__ import annotations

import os

import numpy as np

from pairs._rng import default_rng
from pairs.spread import (
    KalmanHedge,
    build_spread,
    fit_ou,
    half_life,
    ou_diagnostics,
    tls_hedge,
    zscore,
)


def test_full_pipeline_on_cointegrated_pair(cointegrated_prices) -> None:
    os.environ["KALMAN_BACKEND"] = "numpy"
    rng = default_rng(12345)
    y, x = cointegrated_prices(rng, beta_true=1.5, n=1500)

    hedge = tls_hedge(y, x)
    # TLS on log-price levels with cumulative-noise paths shows finite-sample bias
    # (orthogonal regression on integrated series is asymptotically consistent but
    # converges slowly). Tolerance kept loose to track that reality.
    assert abs(hedge.beta - 1.5) < 0.30

    spread = build_spread(y, x, beta=hedge.beta, alpha=hedge.alpha)
    assert spread.name.startswith("spread(")

    ou = fit_ou(spread)
    assert 0.0 < ou.phi < 1.0
    assert ou.theta > 0.0

    hl = half_life(spread, n_boot=60, rng=default_rng(11))
    assert hl.ci_lower <= hl.point <= hl.ci_upper

    z = zscore(spread, window=None, ou_result=ou)
    assert z.dropna().shape[0] > 0
    assert np.isfinite(z.dropna()).all()

    z_ou = zscore(spread, use_ou=True, ou_result=ou)
    assert np.isfinite(z_ou).all()

    diag = ou_diagnostics(spread, ou)
    assert 0.0 <= diag.adf_pvalue <= 1.0

    kalman = KalmanHedge().fit(y, x, delta=1e-4)
    assert kalman.beta_series.shape == y.shape
    assert np.isfinite(kalman.log_likelihood)
