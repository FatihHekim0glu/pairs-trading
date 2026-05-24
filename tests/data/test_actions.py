"""Tests for ``pairs.data.actions`` (corporate-action blacklist)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from pairs._exceptions import InputError
from pairs.data import actions as actions_mod
from pairs.data.actions import blacklist_entries, is_blacklisted


def test_blacklist_yaml_parses() -> None:
    entries = blacklist_entries()
    assert len(entries) >= 1
    tickers = {e.ticker for e in entries}
    assert "ABBV" in tickers
    for entry in entries:
        assert entry.ticker == entry.ticker.upper()
        assert entry.window_days >= 0
        assert isinstance(entry.date, date)


def test_is_blacklisted_abbv_before_2013_01_02_true() -> None:
    # Within the symmetric 30-day window.
    assert is_blacklisted("ABBV", "2012-12-20") is True
    assert is_blacklisted("ABBV", "2013-01-02") is True


def test_is_blacklisted_abbv_after_window_false() -> None:
    # Far outside the 30-day window.
    assert is_blacklisted("ABBV", "2013-06-01") is False
    assert is_blacklisted("ABBV", "2010-01-01") is False


def test_is_blacklisted_unknown_ticker_false() -> None:
    assert is_blacklisted("ZZZZ", "2020-01-01") is False


@given(k=st.integers(min_value=0, max_value=30))
def test_blacklist_window_symmetric(k: int) -> None:
    """If a ticker is blacklisted ``k`` days after the event, it must also be
    blacklisted ``k`` days before."""
    entries = blacklist_entries()
    for entry in entries:
        if k > entry.window_days:
            continue
        after = entry.date + timedelta(days=k)
        before = entry.date - timedelta(days=k)
        assert is_blacklisted(entry.ticker, after) == is_blacklisted(entry.ticker, before)


def test_blacklist_window_boundary_inclusive() -> None:
    """Day exactly ``window_days`` away from the event must be inclusive on both sides."""
    for entry in blacklist_entries():
        edge_before = entry.date - timedelta(days=entry.window_days)
        edge_after = entry.date + timedelta(days=entry.window_days)
        one_past_before = entry.date - timedelta(days=entry.window_days + 1)
        one_past_after = entry.date + timedelta(days=entry.window_days + 1)
        assert is_blacklisted(entry.ticker, edge_before) is True
        assert is_blacklisted(entry.ticker, edge_after) is True
        assert is_blacklisted(entry.ticker, one_past_before) is False
        assert is_blacklisted(entry.ticker, one_past_after) is False


def test_is_blacklisted_accepts_timestamp_and_date_inputs() -> None:
    assert is_blacklisted("ABBV", pd.Timestamp("2013-01-02")) is True
    assert is_blacklisted("abbv", date(2013, 1, 2)) is True  # lowercase ticker normalises


def test_is_blacklisted_rejects_unparseable_asof() -> None:
    with pytest.raises(InputError):
        is_blacklisted("ABBV", 12345)  # type: ignore[arg-type]


def test_blacklist_yaml_roundtrips_via_disk(tmp_path: Path) -> None:
    """Reading the packaged YAML and writing it back out yields equivalent data."""
    from importlib import resources

    src_path = Path(
        str(resources.files("pairs.data.data.blacklist").joinpath("structural_breaks.yaml")),
    )
    payload = yaml.safe_load(src_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert isinstance(payload["entries"], list)
    out = tmp_path / "rt.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    reparsed = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert reparsed == payload


def test_blacklist_non_mapping_payload_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_load_yaml`` itself rejects a top-level list."""
    actions_mod._load_yaml.cache_clear()
    bad = tmp_path / "blacklist.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")
    from importlib import resources

    class _Fake:
        @staticmethod
        def joinpath(_: str) -> Path:
            return bad

    monkeypatch.setattr(resources, "files", lambda _name: _Fake())  # type: ignore[arg-type]
    with pytest.raises(InputError, match="mapping"):
        actions_mod._load_yaml()
    actions_mod._load_yaml.cache_clear()


def test_blacklist_missing_file_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import resources

    actions_mod._load_yaml.cache_clear()
    actions_mod.blacklist_entries.cache_clear()

    class _Fake:
        @staticmethod
        def joinpath(_: str) -> Path:
            return tmp_path / "does_not_exist.yaml"

    monkeypatch.setattr(resources, "files", lambda _name: _Fake())  # type: ignore[arg-type]
    with pytest.raises(InputError, match="not found"):
        actions_mod._load_yaml()
    actions_mod._load_yaml.cache_clear()


def test_blacklist_wrong_schema_version_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions_mod._load_yaml.cache_clear()
    actions_mod.blacklist_entries.cache_clear()
    bad = tmp_path / "blacklist.yaml"
    bad.write_text("schema_version: 99\nentries: []\n", encoding="utf-8")
    from importlib import resources

    class _Fake:
        @staticmethod
        def joinpath(_: str) -> Path:
            return bad

    monkeypatch.setattr(resources, "files", lambda _name: _Fake())  # type: ignore[arg-type]
    with pytest.raises(InputError, match="schema_version"):
        actions_mod._load_yaml()
    actions_mod._load_yaml.cache_clear()


def test_blacklist_entry_missing_key_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions_mod._load_yaml.cache_clear()
    actions_mod.blacklist_entries.cache_clear()
    monkeypatch.setattr(
        actions_mod,
        "_load_yaml",
        lambda: {"schema_version": 1, "entries": [{"ticker": "AAA", "event": "x"}]},
    )
    with pytest.raises(InputError, match="missing"):
        actions_mod.blacklist_entries()
    actions_mod.blacklist_entries.cache_clear()


def test_blacklist_entry_not_mapping_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    actions_mod.blacklist_entries.cache_clear()
    monkeypatch.setattr(
        actions_mod,
        "_load_yaml",
        lambda: {"schema_version": 1, "entries": ["not a dict"]},
    )
    with pytest.raises(InputError, match="not a mapping"):
        actions_mod.blacklist_entries()
    actions_mod.blacklist_entries.cache_clear()


def test_blacklist_entries_not_list_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    actions_mod.blacklist_entries.cache_clear()
    monkeypatch.setattr(
        actions_mod,
        "_load_yaml",
        lambda: {"schema_version": 1, "entries": {"oops": 1}},
    )
    with pytest.raises(InputError, match="must be a list"):
        actions_mod.blacklist_entries()
    actions_mod.blacklist_entries.cache_clear()
