"""Profile-loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pairs._exceptions import InputError
from pairs.backtest import load_profile
from pairs.backtest.accounting import PerShareCommission
from pairs.backtest.borrow import ConstantBorrow
from pairs.backtest.costs import CompositeCost
from pairs.backtest.profiles import PROFILES_ROOT, available_profiles
from pairs.backtest.slippage import HalfSpreadSlippage


def test_profiles_directory_exists() -> None:
    assert PROFILES_ROOT.is_dir(), f"profiles directory missing: {PROFILES_ROOT}"
    names = available_profiles()
    assert {"large_cap_optimistic", "large_cap_realistic", "mid_cap_realistic"} <= set(names)


def test_load_profile_large_cap_realistic() -> None:
    profile = load_profile("large_cap_realistic")
    assert isinstance(profile, CompositeCost)
    assert profile.name == "large_cap_realistic"
    assert isinstance(profile.slippage_model, HalfSpreadSlippage)
    assert isinstance(profile.commission_model, PerShareCommission)
    assert isinstance(profile.borrow, ConstantBorrow)
    assert profile.borrow.rate_bps_annual == 25.0


def test_load_profile_unknown_raises() -> None:
    with pytest.raises(InputError, match="not found"):
        load_profile("nonexistent_profile_xyz")


def test_profile_yaml_roundtrip(tmp_path: Path) -> None:
    # Write a custom YAML, load it via load_profile with root override, and verify the fields.
    p = tmp_path / "custom.yaml"
    p.write_text(
        "name: custom\n"
        "slippage: {type: HalfSpreadSlippage, spread_bps: 7.5}\n"
        "commission: {type: PerShareCommission, per_share: 0.003, min_per_trade: 0.5}\n"
        "borrow: {type: ConstantBorrow, rate_bps_annual: 75}\n",
        encoding="utf-8",
    )
    profile = load_profile("custom", root=tmp_path)
    assert profile.name == "custom"
    assert profile.borrow.rate_bps_annual == 75.0
    assert isinstance(profile.slippage_model, HalfSpreadSlippage)


def test_profile_rejects_unknown_slippage_type(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        "name: bad\n"
        "slippage: {type: NotAClass, bps: 1.0}\n"
        "commission: {type: PerShareCommission, per_share: 0.003}\n"
        "borrow: {type: ConstantBorrow, rate_bps_annual: 10}\n",
        encoding="utf-8",
    )
    with pytest.raises(InputError, match="unknown slippage"):
        load_profile("bad", root=tmp_path)


def test_profile_rejects_missing_section(tmp_path: Path) -> None:
    p = tmp_path / "no_borrow.yaml"
    p.write_text(
        "name: no_borrow\n"
        "slippage: {type: ConstantBpsSlippage, bps: 1.0}\n"
        "commission: {type: FixedCommission, per_trade: 0.0}\n",
        encoding="utf-8",
    )
    with pytest.raises(InputError, match="missing required key"):
        load_profile("no_borrow", root=tmp_path)


def test_available_profiles_returns_empty_when_root_missing(tmp_path: Path) -> None:
    assert available_profiles(tmp_path / "does_not_exist") == []
