"""Direct tests for :mod:`pairs.strategy.rules`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs._exceptions import InputError
from pairs.strategy.rules import apply_rules


def test_state_machine_transitions() -> None:
    z = np.array([0.0, 2.5, 1.5, 0.4, -2.5, -2.5, -0.2])
    series = pd.Series(z, dtype=float)
    out = apply_rules(series, z_entry=2.0, z_exit=0.5, z_stop=5.0, time_stop_bars=None)
    # Enter short at idx 1, hold idx 2 (1.5 in band), exit at idx 3 (|z|=0.4 < 0.5),
    # re-enter long at idx 4, hold idx 5, exit at idx 6.
    np.testing.assert_array_equal(out.positions, np.array([0, -1, -1, 0, 1, 1, 0], dtype=np.int8))
    np.testing.assert_array_equal(out.entries, np.array([0, 1, 0, 0, 1, 0, 0], dtype=bool))
    np.testing.assert_array_equal(out.exits, np.array([0, 0, 0, 1, 0, 0, 1], dtype=bool))


def test_time_stop_auto_from_half_life() -> None:
    z = np.full(10, 3.0)
    z[0] = 0.0
    series = pd.Series(z, dtype=float)
    out = apply_rules(series, z_entry=2.0, z_exit=0.5, z_stop=5.0, time_stop_bars=3)
    # Enter at idx 1, bars_in_position counts 1,2,3 across idx 1..3, time-stop fires at idx 4.
    assert out.positions[1] == -1
    assert out.positions[4] == 0
    assert out.time_stops[4]
    assert out.exits[4]


def test_time_stop_explicit_override() -> None:
    z = np.full(8, 3.0)
    z[0] = 0.0
    series = pd.Series(z, dtype=float)
    short = apply_rules(series, z_entry=2.0, z_exit=0.5, z_stop=5.0, time_stop_bars=2)
    long_ts = apply_rules(series, z_entry=2.0, z_exit=0.5, z_stop=5.0, time_stop_bars=5)
    # Tighter time-stop must trigger sooner.
    assert int(short.time_stops.sum()) >= 1
    assert int(long_ts.time_stops.sum()) <= int(short.time_stops.sum())


def test_z_stop_triggers_exit_with_lag() -> None:
    z = np.array([0.0, 2.5, 6.0, 0.0])
    series = pd.Series(z, dtype=float)
    out = apply_rules(series, z_entry=2.0, z_exit=0.5, z_stop=4.0, time_stop_bars=None)
    assert out.positions[1] == -1
    assert out.positions[2] == 0
    assert bool(out.stops[2])
    # The stop must also be recorded as an exit.
    assert bool(out.exits[2])


def test_apply_rules_handles_nan_inside_trade() -> None:
    z = np.array([0.0, 2.5, np.nan, np.nan, 0.4])
    series = pd.Series(z, dtype=float)
    out = apply_rules(series, z_entry=2.0, z_exit=0.5, z_stop=5.0, time_stop_bars=None)
    # Position held through NaN bars, exits on the 0.4 bar.
    assert out.positions[2] == -1
    assert out.positions[3] == -1
    assert out.positions[4] == 0


def test_apply_rules_rejects_bad_thresholds() -> None:
    with pytest.raises(InputError, match="thresholds"):
        apply_rules(
            pd.Series([0.0]),
            z_entry=0.5,
            z_exit=2.0,
            z_stop=3.0,
            time_stop_bars=None,
        )


def test_apply_rules_rejects_non_series() -> None:
    with pytest.raises(InputError, match="must be a pandas Series"):
        apply_rules(
            np.array([0.0, 1.0]),  # type: ignore[arg-type]
            z_entry=2.0,
            z_exit=0.5,
            z_stop=3.0,
            time_stop_bars=None,
        )
