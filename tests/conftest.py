"""Root pytest fixtures and Hypothesis profiles.

Profiles are registered eagerly so individual test modules do not need to know
about them. The active profile is selected via the ``HYPOTHESIS_PROFILE``
environment variable, defaulting to ``"dev"`` for local runs. CI sets the
variable to ``"ci"`` to widen the search.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, Verbosity, settings

try:
    import matplotlib as mpl

    mpl.use("Agg")
except ModuleNotFoundError:  # matplotlib is optional in [dev]; only used for plot-heavy paths
    pass


_SEED: int = 20260523


settings.register_profile(
    "dev",
    max_examples=25,
    deadline=None,
    derandomize=False,
    suppress_health_check=[HealthCheck.too_slow],
    verbosity=Verbosity.normal,
)

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    verbosity=Verbosity.normal,
)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture
def seed() -> int:
    """Return the deterministic test seed."""
    return _SEED


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    """Return a fresh ``Generator`` seeded with :data:`_SEED`."""
    return np.random.default_rng(seed)


@pytest.fixture
def tmp_manifest_dir(tmp_path: Path) -> Path:
    """Return a unique manifest directory rooted at ``tmp_path``."""
    target: Path = tmp_path / "manifests"
    target.mkdir(parents=True, exist_ok=True)
    return target
