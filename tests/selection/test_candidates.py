"""Tests for :mod:`pairs.selection.candidates`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from pairs._exceptions import InputError
from pairs.selection import candidates as candidates_module
from pairs.selection.candidates import generate_candidates


@dataclass(frozen=True)
class _PairSpecStub:
    a: str
    b: str
    rationale: str = ""


@dataclass(frozen=True)
class _PairUniverseStub:
    pairs: tuple[_PairSpecStub, ...]


def _curated_25() -> _PairUniverseStub:
    pairs = tuple(
        _PairSpecStub(a=f"A{i:02d}", b=f"B{i:02d}")
        for i in range(25)
    )
    return _PairUniverseStub(pairs=pairs)


def _make_panel(n_tickers: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {f"T{i:02d}": [1.0] * 10 for i in range(n_tickers)},
        index=idx,
    )


def _patch_data(monkeypatch: Any, pair_universe: _PairUniverseStub) -> None:
    monkeypatch.setattr(
        candidates_module, "load_pair_universe", lambda _name: pair_universe
    )
    monkeypatch.setattr(
        candidates_module, "is_blacklisted", lambda _t, _asof: False
    )


def test_curated_mode_returns_25_pairs(monkeypatch: Any) -> None:
    _patch_data(monkeypatch, _curated_25())
    panel = _make_panel(0)  # not used in curated mode
    cands = generate_candidates(
        "curated_25_v1",
        panel,
        mode="curated",
        asof=pd.Timestamp("2024-01-01"),
    )
    assert len(cands) == 25


def test_within_sector_unknown_group_returns_all_pairs(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        candidates_module, "is_blacklisted", lambda _t, _asof: False
    )
    panel = _make_panel(5)
    cands = generate_candidates(
        "ignored",
        panel,
        mode="within_sector",
        asof=pd.Timestamp("2024-01-01"),
    )
    # 5 choose 2 == 10
    assert len(cands) == 10
    for c in cands:
        assert c.sector == "unknown"


def test_force_required_above_1500_pairs(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        candidates_module, "is_blacklisted", lambda _t, _asof: False
    )
    # 60 choose 2 == 1770 > 1500
    panel = _make_panel(60)
    with pytest.raises(InputError):
        generate_candidates(
            "ignored",
            panel,
            mode="within_sub_industry",
            asof=pd.Timestamp("2024-01-01"),
        )
    # Force = True bypasses the gate.
    cands = generate_candidates(
        "ignored",
        panel,
        mode="within_sub_industry",
        asof=pd.Timestamp("2024-01-01"),
        force=True,
    )
    assert len(cands) == 60 * 59 // 2


def test_unknown_mode_raises(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        candidates_module, "is_blacklisted", lambda _t, _asof: False
    )
    panel = _make_panel(3)
    with pytest.raises(InputError):
        generate_candidates(
            "ignored",
            panel,
            mode="banana",  # type: ignore[arg-type]
            asof=pd.Timestamp("2024-01-01"),
        )


def test_non_dataframe_raises() -> None:
    with pytest.raises(InputError):
        generate_candidates(
            "ignored",
            [["foo"]],  # type: ignore[arg-type]
            mode="within_sector",
            asof=pd.Timestamp("2024-01-01"),
        )


def test_within_sector_respects_sector_map(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        candidates_module, "is_blacklisted", lambda _t, _asof: False
    )
    panel = _make_panel(4)
    sector_map = {"T00": "Tech", "T01": "Tech", "T02": "Energy", "T03": "Energy"}
    cands = generate_candidates(
        "ignored",
        panel,
        mode="within_sector",
        asof=pd.Timestamp("2024-01-01"),
        sector_map=sector_map,
    )
    # 1 pair within Tech + 1 pair within Energy = 2.
    assert len(cands) == 2
    for c in cands:
        assert c.sector in {"Tech", "Energy"}


def test_candidate_with_reason_and_pair_id() -> None:
    from pairs.selection.results import Candidate as _C

    c = _C(ticker_a="aapl", ticker_b="msft")
    assert c.pair_id == "AAPL__MSFT"
    annotated = c.with_reason("adv_floor")
    assert annotated.exclusion_reason == ("adv_floor",)
    chained = annotated.with_reason("corr_band")
    assert chained.exclusion_reason == ("adv_floor", "corr_band")


def test_candidate_rejects_self_pair() -> None:
    from pairs.selection.results import Candidate as _C

    with pytest.raises(ValueError, match="self-pair"):
        _C(ticker_a="AAA", ticker_b="AAA")


def test_candidate_rejects_empty_ticker() -> None:
    from pairs.selection.results import Candidate as _C

    with pytest.raises(ValueError, match="non-empty"):
        _C(ticker_a="", ticker_b="MSFT")


def test_curated_filters_blacklisted(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        candidates_module, "load_pair_universe", lambda _name: _curated_25()
    )
    # Blacklist every A-prefix ticker.
    monkeypatch.setattr(
        candidates_module,
        "is_blacklisted",
        lambda t, _asof: t.startswith("A"),
    )
    panel = _make_panel(0)
    cands = generate_candidates(
        "curated_25_v1",
        panel,
        mode="curated",
        asof=pd.Timestamp("2024-01-01"),
    )
    assert cands == []
