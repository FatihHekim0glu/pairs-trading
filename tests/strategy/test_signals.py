"""Signal-generation tests for :mod:`pairs.strategy.signals`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pairs._exceptions import InputError
from pairs.strategy import generate_signals
from pairs.strategy.signals import default_time_stop


def test_basic_entry_exit(step_zscore: pd.Series) -> None:
    pos = generate_signals(step_zscore, z_entry=2.0, z_exit=0.5, z_stop=5.0)
    # Entry happens when z first crosses +2 (bar 10), exit when z drops below 0.5 (bar 15).
    assert pos.iloc[9] == 0
    assert pos.iloc[10] == -1
    assert pos.iloc[14] == -1
    assert pos.iloc[15] == 0


def test_hysteresis_no_exit_above_z_exit() -> None:
    z = np.array([0.0, 0.0, 2.5, 1.5, 1.0, 0.7, 0.4, 0.4])
    series = pd.Series(z, dtype=float)
    pos = generate_signals(series, z_entry=2.0, z_exit=0.5, z_stop=5.0)
    # Entry at index 2 (z=2.5 > 2.0). Should remain in trade while 0.5 < |z| <= 2.0.
    assert pos.iloc[2] == -1
    assert pos.iloc[3] == -1
    assert pos.iloc[4] == -1
    assert pos.iloc[5] == -1
    # Exit at index 6 (|z| = 0.4 < 0.5).
    assert pos.iloc[6] == 0
    assert pos.iloc[7] == 0


def test_z_stop_overrides_z_exit() -> None:
    z = np.array([0.0, 2.5, 2.6, 4.5, 0.0])
    series = pd.Series(z, dtype=float)
    pos = generate_signals(series, z_entry=2.0, z_exit=0.5, z_stop=4.0)
    # Enter at index 1, get stopped out at index 3 because |z| > z_stop.
    assert pos.iloc[1] == -1
    assert pos.iloc[2] == -1
    assert pos.iloc[3] == 0


def test_default_time_stop_from_half_life() -> None:
    assert default_time_stop(0.4) == 2  # below floor
    assert default_time_stop(5.0) == 10
    assert default_time_stop(7.5) == 16  # round-half-to-even: round(7.5) == 8


def test_signals_input_validation() -> None:
    with pytest.raises(InputError, match="must be a pandas Series"):
        generate_signals([0.0, 1.0])  # type: ignore[arg-type]


def test_signals_output_index_and_dtype(oscillating_zscore: pd.Series) -> None:
    pos = generate_signals(oscillating_zscore)
    assert pos.index.equals(oscillating_zscore.index)
    assert str(pos.dtype) == "int8"
    assert set(np.unique(pos.to_numpy())).issubset({-1, 0, 1})


@given(
    hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=5, max_value=50),
        elements=st.floats(
            min_value=-5.0,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
)
def test_symmetry_negation(values: np.ndarray) -> None:
    """generate_signals(-z) == -generate_signals(z) (sign-flip invariance)."""
    z = pd.Series(values, dtype=float)
    pos_pos = generate_signals(z, z_entry=2.0, z_exit=0.5, z_stop=4.0)
    pos_neg = generate_signals(-z, z_entry=2.0, z_exit=0.5, z_stop=4.0)
    np.testing.assert_array_equal(pos_pos.to_numpy(), -pos_neg.to_numpy())


def test_time_stop_auto_from_half_life() -> None:
    # Five bars of strong z-score, no exit-band crossing; with half_life=1 the
    # auto time-stop should be max(2, 2*1) = 2 -> position closes at bar 4.
    z = np.zeros(10)
    z[2:8] = 3.0
    series = pd.Series(z, dtype=float)
    pos = generate_signals(series, z_entry=2.0, z_exit=0.5, z_stop=5.0, half_life=1.0)
    assert pos.iloc[2] == -1
    assert pos.iloc[3] == -1
    assert pos.iloc[4] == 0  # bars_in_position would hit 3 > 2 -> time-stop
