from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pairs._exceptions import InputError
from pairs.portfolio import apply_caps


def _maps(n: int = 10):
    sector_map = {f"P{i}": ("SEC0" if i < 5 else "SEC1") for i in range(n)}
    asset_legs_map = {f"P{i}": (f"A{i}", f"A{i + 100}") for i in range(n)}
    return sector_map, asset_legs_map


@given(
    raw=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=10,
    )
)
def test_no_sector_exceeds_cap_post_apply(raw: list[float]) -> None:
    sector_map, asset_legs_map = _maps(10)
    weights = pd.Series(raw, index=[f"P{i}" for i in range(10)])
    out, _ = apply_caps(
        weights,
        max_pairs=10,
        max_sector_gross=0.30,
        max_asset_notional=0.10,
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
    )
    for sector in ("SEC0", "SEC1"):
        load = float(
            sum(abs(out.loc[p]) for p in out.index if sector_map[p] == sector)
        )
        assert load <= 0.30 + 1e-9


def test_max_pairs_truncates_smallest() -> None:
    sector_map, asset_legs_map = _maps(10)
    raw = pd.Series(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
        index=[f"P{i}" for i in range(10)],
    )
    out, events = apply_caps(
        raw,
        max_pairs=3,
        max_sector_gross=1.0,
        max_asset_notional=1.0,
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
    )
    nonzero = out[out != 0.0]
    assert len(nonzero) == 3
    assert set(nonzero.index) == {"P7", "P8", "P9"}
    assert any(e.kind == "max_pairs" for e in events)


def test_per_asset_notional_cap() -> None:
    sector_map = {"P0": "SEC0", "P1": "SEC1"}
    asset_legs_map = {"P0": ("X", "Y"), "P1": ("X", "Z")}  # share asset X
    raw = pd.Series([0.5, 0.5], index=["P0", "P1"])
    out, events = apply_caps(
        raw,
        max_pairs=10,
        max_sector_gross=1.0,
        max_asset_notional=0.10,
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
    )
    # Asset X carries P0 + P1, must be <= 0.10.
    assert abs(out["P0"]) + abs(out["P1"]) <= 0.10 + 1e-9
    assert any(e.kind == "asset_notional" for e in events)


def test_cap_events_logged() -> None:
    sector_map, asset_legs_map = _maps(10)
    raw = pd.Series(np.full(10, 0.20), index=[f"P{i}" for i in range(10)])
    _, events = apply_caps(
        raw,
        max_pairs=5,
        max_sector_gross=0.30,
        max_asset_notional=0.10,
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
    )
    assert len(events) > 0
    kinds = {e.kind for e in events}
    assert "max_pairs" in kinds


def test_no_op_when_all_constraints_satisfied() -> None:
    sector_map, asset_legs_map = _maps(2)
    raw = pd.Series([0.05, 0.05], index=["P0", "P1"])
    out, events = apply_caps(
        raw,
        max_pairs=10,
        max_sector_gross=0.30,
        max_asset_notional=0.10,
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
    )
    np.testing.assert_allclose(out.to_numpy(), raw.to_numpy())
    assert events == []


def test_apply_caps_rejects_non_series() -> None:
    sector_map, asset_legs_map = _maps(2)
    with pytest.raises(InputError, match="pandas Series"):
        apply_caps(
            [0.1, 0.2],  # type: ignore[arg-type]
            sector_map=sector_map,
            asset_legs_map=asset_legs_map,
        )


def test_apply_caps_rejects_nan_weights() -> None:
    sector_map, asset_legs_map = _maps(2)
    bad = pd.Series([0.1, np.nan], index=["P0", "P1"])
    with pytest.raises(InputError, match="NaN"):
        apply_caps(bad, sector_map=sector_map, asset_legs_map=asset_legs_map)


def test_apply_caps_missing_sector_entry() -> None:
    sector_map = {"P0": "SEC0"}  # P1 missing
    asset_legs_map = {"P0": ("A", "B"), "P1": ("C", "D")}
    raw = pd.Series([0.1, 0.1], index=["P0", "P1"])
    with pytest.raises(InputError, match="sector_map missing"):
        apply_caps(raw, sector_map=sector_map, asset_legs_map=asset_legs_map)


def test_apply_caps_missing_asset_entry() -> None:
    sector_map = {"P0": "SEC0", "P1": "SEC1"}
    asset_legs_map = {"P0": ("A", "B")}  # P1 missing
    raw = pd.Series([0.1, 0.1], index=["P0", "P1"])
    with pytest.raises(InputError, match="asset_legs_map missing"):
        apply_caps(raw, sector_map=sector_map, asset_legs_map=asset_legs_map)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"max_pairs": 0}, "max_pairs"),
        ({"max_sector_gross": 0.0}, "max_sector_gross"),
        ({"max_sector_gross": 1.5}, "max_sector_gross"),
        ({"max_asset_notional": -0.1}, "max_asset_notional"),
        ({"max_asset_notional": 2.0}, "max_asset_notional"),
    ],
)
def test_apply_caps_validates_kwargs(kwargs, match) -> None:
    sector_map, asset_legs_map = _maps(2)
    raw = pd.Series([0.1, 0.1], index=["P0", "P1"])
    with pytest.raises(InputError, match=match):
        apply_caps(
            raw, sector_map=sector_map, asset_legs_map=asset_legs_map, **kwargs
        )


def test_sector_cap_emits_events() -> None:
    sector_map, asset_legs_map = _maps(10)
    # Heavy SEC0 loading so sector cap binds but asset notional does not.
    raw = pd.Series(
        [0.20, 0.20, 0.20, 0.20, 0.20, 0.01, 0.01, 0.01, 0.01, 0.01],
        index=[f"P{i}" for i in range(10)],
    )
    out, events = apply_caps(
        raw,
        max_pairs=10,
        max_sector_gross=0.30,
        max_asset_notional=1.0,
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
    )
    sec0 = sum(abs(out.loc[p]) for p in out.index if sector_map[p] == "SEC0")
    assert sec0 <= 0.30 + 1e-9
    assert any(e.kind == "sector_gross" for e in events)


def test_apply_caps_renormalises_above_budget() -> None:
    sector_map, asset_legs_map = _maps(2)
    # Heavy weights that survive every per-pair cap, will renormalise.
    raw = pd.Series([0.9, 0.9], index=["P0", "P1"])
    out, _ = apply_caps(
        raw,
        max_pairs=10,
        max_sector_gross=1.0,
        max_asset_notional=1.0,
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
    )
    assert float(out.abs().sum()) <= 1.0 + 1e-9


def test_apply_caps_asof_stamped_on_events() -> None:
    sector_map, asset_legs_map = _maps(10)
    raw = pd.Series(np.full(10, 0.20), index=[f"P{i}" for i in range(10)])
    stamp = pd.Timestamp("2024-06-15")
    _, events = apply_caps(
        raw,
        max_pairs=5,
        max_sector_gross=0.30,
        max_asset_notional=0.10,
        sector_map=sector_map,
        asset_legs_map=asset_legs_map,
        asof=stamp,
    )
    assert events
    assert all(e.asof == stamp for e in events)
