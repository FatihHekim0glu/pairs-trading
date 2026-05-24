"""Tests for the Memmel (2003) Sharpe-equality test."""

from __future__ import annotations

import numpy as np

from pairs.evaluation import memmel_test


def test_memmel_under_null_correct_size() -> None:
    rng = np.random.default_rng(123)
    n_sim = 200
    t = 252
    rejected = 0
    for _ in range(n_sim):
        eps = rng.normal(size=(t, 2))
        a = eps[:, 0]
        b = 0.6 * eps[:, 0] + 0.8 * eps[:, 1]
        result = memmel_test(a, b)
        if result.p_value < 0.05:
            rejected += 1
    rate = rejected / n_sim
    assert 0.01 <= rate <= 0.12


def test_memmel_power_when_clearly_different() -> None:
    rng = np.random.default_rng(7)
    t = 1000
    a = rng.normal(0.20, 1.0, size=t)
    b = rng.normal(-0.20, 1.0, size=t)
    result = memmel_test(a, b)
    assert result.p_value < 0.05
    assert result.sr_a > result.sr_b
