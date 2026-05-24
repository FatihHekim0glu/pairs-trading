"""Fixtures for strategy-layer tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def flat_zscore() -> pd.Series:
    """Return a zero z-score series 200 bars long."""
    return pd.Series(np.zeros(200), index=pd.RangeIndex(200), dtype=float, name="z")


@pytest.fixture
def step_zscore() -> pd.Series:
    """Return a z-score with one entry and one exit cycle.

    Schedule: 0 for 10 bars, +3 for 5 bars (entry), then 0 for 5 bars (exit).
    """
    z = np.zeros(20)
    z[10:15] = 3.0
    return pd.Series(z, index=pd.RangeIndex(20), dtype=float, name="z")


@pytest.fixture
def oscillating_zscore() -> pd.Series:
    """Return an oscillating z-score around the entry band."""
    rng = np.random.default_rng(7)
    base = 0.4 * np.sin(np.linspace(0, 12.0, 200))
    noise = 0.05 * rng.standard_normal(200)
    z = 3.0 * np.sin(np.linspace(0, 4.0 * np.pi, 200)) + noise + base
    return pd.Series(z, index=pd.RangeIndex(200), dtype=float, name="z")
