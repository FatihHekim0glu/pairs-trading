from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from pairs.portfolio import PairLifecycle


def _passing_retest(*_args, **_kwargs):
    return SimpleNamespace(cointegrated=True)


def _failing_retest(*_args, **_kwargs):
    return SimpleNamespace(cointegrated=False)


def test_cooldown_blocks_reentry() -> None:
    lc = PairLifecycle(
        cointegration_retest=_passing_retest,
        half_life_lookup=lambda _pid: 5.0,
        min_cooldown_days=10,
    )
    asof = pd.Timestamp("2024-01-15")
    lc.on_stop_out("P0", asof)
    next_day = asof + pd.Timedelta(days=1)
    assert lc.cooldown_remaining("P0", next_day) > 0
    assert not lc.can_reenter("P0", next_day, pd.DataFrame())


def test_cooldown_expires() -> None:
    lc = PairLifecycle(
        cointegration_retest=_passing_retest,
        half_life_lookup=lambda _pid: 5.0,
        min_cooldown_days=10,
    )
    asof = pd.Timestamp("2024-01-15")
    lc.on_stop_out("P0", asof)
    later = asof + pd.Timedelta(days=15)
    assert lc.cooldown_remaining("P0", later) == 0
    assert lc.can_reenter("P0", later, pd.DataFrame())


def test_cointegration_retest_gates_reentry() -> None:
    lc = PairLifecycle(
        cointegration_retest=_failing_retest,
        half_life_lookup=lambda _pid: 1.0,
        min_cooldown_days=1,
    )
    asof = pd.Timestamp("2024-01-15")
    lc.on_stop_out("P0", asof)
    later = asof + pd.Timedelta(days=30)
    assert lc.cooldown_remaining("P0", later) == 0
    assert not lc.can_reenter("P0", later, pd.DataFrame())


def test_walkforward_resets_cooldown() -> None:
    lc = PairLifecycle(
        cointegration_retest=_passing_retest,
        half_life_lookup=lambda _pid: 5.0,
        min_cooldown_days=10,
    )
    asof = pd.Timestamp("2024-01-15")
    lc.on_stop_out("P0", asof)
    # Walk-forward keeps P0 in the universe; cooldown is cleared.
    lc.on_walkforward_reselect(["P1", "P2"], asof + pd.Timedelta(days=5))
    # P0 was removed from universe -> cooldown registry purged.
    assert lc.cooldown_remaining("P0", asof + pd.Timedelta(days=6)) == 0


def test_active_set_filters_cooldowns() -> None:
    lc = PairLifecycle(
        cointegration_retest=_passing_retest,
        half_life_lookup=lambda _pid: 5.0,
        min_cooldown_days=20,
    )
    asof = pd.Timestamp("2024-01-15")
    lc.on_stop_out("P0", asof)
    active = lc.active_set(["P0", "P1"], asof + pd.Timedelta(days=1), pd.DataFrame())
    assert active == {"P1"}


def test_active_set_readmits_after_cooldown() -> None:
    lc = PairLifecycle(
        cointegration_retest=_passing_retest,
        half_life_lookup=lambda _pid: 5.0,
        min_cooldown_days=10,
    )
    asof = pd.Timestamp("2024-01-15")
    lc.on_stop_out("P0", asof)
    later = asof + pd.Timedelta(days=30)
    active = lc.active_set(["P0", "P1"], later, pd.DataFrame())
    assert active == {"P0", "P1"}


def test_retest_exception_blocks_reentry() -> None:
    def raising_retest(*_args, **_kwargs):
        raise RuntimeError("boom")

    lc = PairLifecycle(
        cointegration_retest=raising_retest,
        half_life_lookup=lambda _pid: 1.0,
        min_cooldown_days=1,
    )
    asof = pd.Timestamp("2024-01-15")
    lc.on_stop_out("P0", asof)
    later = asof + pd.Timedelta(days=10)
    assert not lc.can_reenter("P0", later, pd.DataFrame())
