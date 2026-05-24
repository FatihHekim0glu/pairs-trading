"""Regression tests covering audit-critical invariants.

These tests guard against regressions in:

* :func:`pairs.cointegration.engle_granger` direction selection
  (both regressions actually run, the smaller p-value wins, and the
  unchosen direction's p-value is preserved).
* :func:`pairs.cointegration.bootstrap_coint_pvalue` honouring the
  Davison-Hinkley ``(1 + count) / (1 + n_boot)`` continuity correction
  and producing geometric block lengths with wraparound indices.
* :func:`pairs.cointegration.unit_root_check` rejecting returns while
  accepting log-prices.
* :class:`pairs.cointegration.results.TestDirection` exposing both
  members with the documented string form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs._exceptions import InputError
from pairs.cointegration import (
    bootstrap_coint_pvalue,
    engle_granger,
    unit_root_check,
)
from pairs.cointegration.bootstrap import _stationary_indices
from pairs.cointegration.results import TestDirection


# ---------------------------------------------------------------------------
# Engle-Granger: direction selection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_eg_other_direction_pvalue_matches_reverse_regression(synthetic_coint_pair):
    """The recorded ``pvalue_other_direction`` is the unchosen regression's p."""
    from statsmodels.tsa.stattools import coint

    x, y = synthetic_coint_pair(t=400, rho_residual=0.3)
    res = engle_granger(x, y, use_log=False, autolag="bic")

    _, sm_p_fwd, _ = coint(x.to_numpy(), y.to_numpy(), trend="c", autolag="bic")
    _, sm_p_rev, _ = coint(y.to_numpy(), x.to_numpy(), trend="c", autolag="bic")
    sm_min = min(float(sm_p_fwd), float(sm_p_rev))
    sm_max = max(float(sm_p_fwd), float(sm_p_rev))

    assert res.pvalue == pytest.approx(sm_min, rel=1e-6, abs=1e-6)
    assert res.pvalue_other_direction == pytest.approx(sm_max, rel=1e-6, abs=1e-6)


@pytest.mark.unit
def test_eg_direction_flips_when_reverse_pvalue_is_lower(synthetic_coint_pair):
    """Swapping arguments must swap ``direction_used`` accordingly."""
    x, y = synthetic_coint_pair(t=400, rho_residual=0.3)
    forward = engle_granger(x, y, use_log=False)
    reversed_ = engle_granger(y, x, use_log=False)

    # The minimum p-value across directions is invariant to argument order.
    assert forward.pvalue == pytest.approx(reversed_.pvalue, abs=1e-9)
    # And ``pvalue_other_direction`` is too.
    assert forward.pvalue_other_direction == pytest.approx(
        reversed_.pvalue_other_direction,
        abs=1e-9,
    )
    # But the recorded direction label must mirror the swap.
    if forward.direction_used is TestDirection.Y0_ON_Y1:
        assert reversed_.direction_used is TestDirection.Y1_ON_Y0
    else:
        assert reversed_.direction_used is TestDirection.Y0_ON_Y1


@pytest.mark.unit
def test_eg_pvalue_is_minimum_of_two_regressions(synthetic_coint_pair):
    """The chosen p-value is always <= the recorded other-direction p-value."""
    x, y = synthetic_coint_pair(t=300, rho_residual=0.4)
    res = engle_granger(x, y, use_log=False)
    assert res.pvalue <= res.pvalue_other_direction + 1e-12


# ---------------------------------------------------------------------------
# TestDirection enum
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_test_direction_has_both_members():
    """Both directions must be present and string-format to documented values."""
    assert TestDirection.Y0_ON_Y1.value == "y0_on_y1"
    assert TestDirection.Y1_ON_Y0.value == "y1_on_y0"
    # StrEnum -> str(member) returns the value, not the qualified name.
    assert str(TestDirection.Y0_ON_Y1) == "y0_on_y1"
    assert str(TestDirection.Y1_ON_Y0) == "y1_on_y0"
    assert {m.value for m in TestDirection} == {"y0_on_y1", "y1_on_y0"}


# ---------------------------------------------------------------------------
# Bootstrap: stationary-bootstrap fallback invariants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bootstrap_pvalue_uses_davison_hinkley_correction(synthetic_coint_pair):
    """``pvalue == (1 + count) / (1 + n_boot)`` so it's never exactly zero."""
    x, y = synthetic_coint_pair(t=200, rho_residual=0.3)
    boot = bootstrap_coint_pvalue(x, y, n_boot=99, rng=np.random.default_rng(0))
    # (1 + count) / (1 + 99) is always >= 1/100.
    assert boot.pvalue >= 1.0 / (1 + 99)


@pytest.mark.unit
def test_bootstrap_pvalue_resolution_matches_n_boot():
    """The smallest representable p-value must be ``1 / (1 + n_boot)``."""
    # Use clearly cointegrated data so the observed stat is very negative and
    # essentially no bootstrap replicate is at least as extreme.
    rng = np.random.default_rng(7)
    t = 250
    x = pd.Series(100 + rng.standard_normal(t).cumsum(), name="x")
    # y exactly equals x plus tiny noise -> very strong cointegration.
    y = pd.Series(x.to_numpy() + 0.01 * rng.standard_normal(t), name="y")
    boot = bootstrap_coint_pvalue(x, y, n_boot=49, rng=np.random.default_rng(1))
    # The minimum achievable value is 1 / (1 + 49) = 0.02 thanks to the
    # Davison-Hinkley correction; without it, the p-value could be exactly 0.
    assert boot.pvalue >= 1.0 / (1 + 49) - 1e-12
    assert boot.pvalue > 0.0


@pytest.mark.unit
def test_stationary_indices_wrap_around_and_stay_in_bounds():
    """Geometric blocks may extend past ``n``; output indices must wrap modulo n."""
    n = 50
    gen = np.random.default_rng(123)
    idx = _stationary_indices(n, block_length=10, rng=gen)
    assert idx.shape == (n,)
    assert idx.min() >= 0
    assert idx.max() < n


@pytest.mark.unit
def test_stationary_indices_block_length_approximates_target():
    """Average run length should be close to the requested expected block size."""
    n = 2_000
    target = 20
    gen = np.random.default_rng(0)
    idx = _stationary_indices(n, block_length=target, rng=gen)
    # Count "breaks" where the index is not a +1 wraparound from the previous.
    breaks = 0
    for i in range(1, n):
        if idx[i] != (idx[i - 1] + 1) % n:
            breaks += 1
    # Number of blocks ~= breaks + 1.  Mean run length ~= n / (breaks + 1).
    mean_run = n / (breaks + 1)
    # Generous tolerance: geometric variance is large, but we still expect
    # mean run length within a factor of two of the target on average.
    assert 0.5 * target <= mean_run <= 2.0 * target


# ---------------------------------------------------------------------------
# Unit-root check: returns vs log-prices guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unit_root_log_prices_accepted_returns_rejected():
    """Levels -> pass; first differences (returns) -> InputError."""
    gen = np.random.default_rng(2026)
    prices = 100 + gen.standard_normal(400).cumsum()
    log_prices = np.log(prices)

    res = unit_root_check(log_prices, leg_name="px")
    assert res.is_i1
    assert res.leg_name == "px"

    returns = np.diff(log_prices)
    with pytest.raises(InputError, match="already stationary"):
        unit_root_check(returns)


@pytest.mark.unit
def test_unit_root_explicit_method_paths():
    """Cover the ``method='adf'`` explicit branch."""
    gen = np.random.default_rng(11)
    log_prices = np.log(100 + gen.standard_normal(300).cumsum())
    res = unit_root_check(log_prices, method="adf")
    assert res.method == "adf"


@pytest.mark.unit
def test_unit_root_rejects_bad_alpha_and_min_obs():
    gen = np.random.default_rng(3)
    prices = 100 + gen.standard_normal(200).cumsum()
    with pytest.raises(InputError):
        unit_root_check(prices, alpha=0.0)
    with pytest.raises(InputError):
        unit_root_check(prices, alpha=1.0)
    with pytest.raises(InputError):
        unit_root_check(prices, min_obs_for_dfgls=0)


@pytest.mark.unit
def test_unit_root_rejects_2d_input(rng):
    arr = rng.standard_normal((100, 2))
    with pytest.raises(InputError, match="1-D"):
        unit_root_check(arr)


# ---------------------------------------------------------------------------
# Bootstrap argument-handling coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bootstrap_block_length_override_honoured(synthetic_coint_pair):
    x, y = synthetic_coint_pair(t=150)
    boot = bootstrap_coint_pvalue(
        x,
        y,
        n_boot=20,
        block_length=12,
        rng=np.random.default_rng(0),
    )
    assert boot.block_length == 12
