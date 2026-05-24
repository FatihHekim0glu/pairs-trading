"""Tests for Hansen's SPA test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.evaluation import hansen_spa


@pytest.mark.slow
def test_spa_null_rejection_rate() -> None:
    rng = np.random.default_rng(99)
    trials = 30
    rejections = 0
    for _ in range(trials):
        strat = pd.DataFrame(rng.normal(size=(200, 5)))
        bench = pd.Series(rng.normal(size=200))
        result = hansen_spa(strat, bench, n_boot=200, rng=rng)
        if result.p_value_consistent < 0.05:
            rejections += 1
    # Under the null, ~5% rejection expected; allow generous slack.
    assert rejections / trials <= 0.30


def test_spa_returns_best_model() -> None:
    rng = np.random.default_rng(3)
    bench = pd.Series(rng.normal(0.0, 1.0, size=300))
    strat = pd.DataFrame(rng.normal(0.0, 1.0, size=(300, 4)), columns=["a", "b", "c", "d"])
    strat["b"] += 0.3  # boost column "b"
    result = hansen_spa(strat, bench, n_boot=200, rng=rng)
    assert result.best_model == "b"
    assert result.n_models == 4
