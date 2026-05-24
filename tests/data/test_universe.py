"""Tests for ``pairs.data.universe``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pairs._exceptions import InputError
from pairs.data import universe as universe_mod
from pairs.data.universe import (
    PairSpec,
    PairUniverse,
    load_pair_universe,
    load_universe,
)


def test_curated_25_loads_returns_25_pairs() -> None:
    universe = load_pair_universe("curated_25_v1")
    assert isinstance(universe, PairUniverse)
    assert len(universe.pairs) == 25
    assert universe.universe_id == "curated_25_v1"
    assert universe.schema_version == 1


def test_curated_25_pairs_are_tuples_uppercase() -> None:
    universe = load_pair_universe("curated_25_v1")
    for spec in universe.pairs:
        assert isinstance(spec, PairSpec)
        assert spec.a == spec.a.upper() and spec.a.isalnum()
        assert spec.b == spec.b.upper() and spec.b.isalnum()
        assert spec.a != spec.b
        assert spec.rationale  # non-empty


def test_no_duplicate_pairs() -> None:
    universe = load_pair_universe("curated_25_v1")
    seen: set[frozenset[str]] = set()
    for spec in universe.pairs:
        key = frozenset({spec.a, spec.b})
        assert key not in seen, f"duplicate pair: {spec.a}/{spec.b}"
        seen.add(key)


def test_xlk_v1_nonempty() -> None:
    universe = load_universe("xlk_v1")
    assert universe.universe_id == "xlk_v1"
    assert universe.schema_version == 1
    assert len(universe.tickers) == 20
    assert "AAPL" in universe.tickers
    assert all(t == t.upper() for t in universe.tickers)
    # Sorted invariant from loader.
    assert list(universe.tickers) == sorted(universe.tickers)


def test_unknown_universe_raises() -> None:
    with pytest.raises(InputError):
        load_universe("does_not_exist_v999")
    with pytest.raises(InputError):
        load_pair_universe("also_missing_v0")
