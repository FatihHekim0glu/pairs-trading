"""Shared fixtures for ``tests/data``."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pairs._rng import default_rng
from pairs.data.manifest import ManifestEntry


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic generator for test stochasticity."""
    return default_rng(seed=20260523)


@pytest.fixture
def synthetic_prices(rng: np.random.Generator) -> pd.DataFrame:
    """Geometric-Brownian-motion price frame with MultiIndex columns.

    500 business days, 3 tickers, two fields each (``Adj Close``, ``Volume``),
    tz-aware UTC index.
    """
    n_days = 500
    tickers = ["AAA", "BBB", "CCC"]
    idx = pd.date_range("2020-01-02", periods=n_days, freq="B", tz="UTC", name="date")
    drift = 0.0002
    sigma = 0.02
    increments = rng.normal(loc=drift, scale=sigma, size=(n_days, len(tickers)))
    prices = 100.0 * np.exp(np.cumsum(increments, axis=0))
    vols = rng.integers(1_000_000, 10_000_000, size=(n_days, len(tickers)))
    cols = pd.MultiIndex.from_product([tickers, ["Adj Close", "Volume"]])
    data = np.empty((n_days, len(tickers) * 2))
    for i in range(len(tickers)):
        data[:, 2 * i] = prices[:, i]
        data[:, 2 * i + 1] = vols[:, i]
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture
def tmp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch cache dir pointed at by a patched ``get_settings``."""
    cache = tmp_path / "cache"
    (cache / "prices").mkdir(parents=True)

    class _StubSettings:
        cache_dir: Path = cache
        offline: bool = False

    def _get_settings() -> Any:
        return _StubSettings()

    monkeypatch.setattr("pairs._config.get_settings", _get_settings)
    monkeypatch.setattr("pairs.data.loader.get_settings", _get_settings)
    monkeypatch.setattr("pairs.data._yfinance_adapter.get_settings", _get_settings)
    return cache


@pytest.fixture
def frozen_manifest_entry() -> ManifestEntry:
    """A ``ManifestEntry`` with a known SHA-256 for round-trip tests."""
    return ManifestEntry(
        relpath="prices/AAA.parquet",
        sha256="0" * 64,
        bytes=12345,
        rows=500,
        start="2020-01-02T00:00:00+00:00",
        end="2022-01-02T00:00:00+00:00",
        provider="yfinance",
        written_at=datetime(2026, 5, 23, tzinfo=UTC).isoformat(),
    )
