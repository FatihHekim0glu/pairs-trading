"""Unit tests for :func:`pairs.cointegration.johansen`."""

from __future__ import annotations

import pandas as pd
import pytest

from pairs.cointegration import johansen


@pytest.mark.unit
def test_johansen_rank_1_for_synthetic_pair(synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=500, rho_residual=0.3)
    res = johansen(pd.concat({"x": x, "y": y}, axis=1))
    assert res.rank >= 1
    assert res.n_obs == 500
    assert res.trace_stats.shape == res.trace_crit_95.shape


@pytest.mark.unit
def test_johansen_rank_0_for_random_walks(two_random_walks):
    x, y = two_random_walks(t=500)
    res = johansen(pd.concat({"x": x, "y": y}, axis=1))
    assert res.rank == 0


@pytest.mark.integration
def test_johansen_handles_3_asset_basket(rng):
    t = 500
    x = pd.Series(rng.standard_normal(t).cumsum() + 100, name="x")
    y = pd.Series(rng.standard_normal(t).cumsum() + 100, name="y")
    # z is a cointegrating combination of x and y plus a small AR(1) residual.
    eps = rng.standard_normal(t) * 0.2
    z = pd.Series(0.5 * x.to_numpy() + 0.5 * y.to_numpy() + eps, name="z")
    df = pd.concat([x, y, z], axis=1)
    res = johansen(df, det_order=0, k_ar_diff=1)
    assert res.rank >= 1


@pytest.mark.unit
def test_johansen_rejects_bad_det_order(synthetic_coint_pair):
    from pairs._exceptions import InputError

    x, y = synthetic_coint_pair(t=100)
    with pytest.raises(InputError):
        johansen(pd.concat({"x": x, "y": y}, axis=1), det_order=5)


@pytest.mark.unit
def test_johansen_requires_two_columns(rng):
    from pairs._exceptions import InputError

    with pytest.raises(InputError):
        johansen(pd.DataFrame({"x": rng.standard_normal(100)}))
