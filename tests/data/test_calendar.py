"""Tests for ``pairs.data.calendar``.

Skipped wholesale when ``pandas_market_calendars`` is unavailable, since it
lives in the ``[app]`` extra.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

_calendar_available = importlib.util.find_spec("pandas_market_calendars") is not None

pytestmark = pytest.mark.skipif(
    not _calendar_available,
    reason="pandas_market_calendars not installed (in [app] extra)",
)

if _calendar_available:
    from pairs.data.calendar import align_to_sessions, nyse_sessions


def test_nyse_excludes_weekends() -> None:
    sessions = nyse_sessions("2023-01-01", "2023-01-31")
    weekday_codes = sessions.weekday
    assert (weekday_codes < 5).all()


def test_nyse_excludes_christmas_2023() -> None:
    sessions = nyse_sessions("2023-12-20", "2023-12-31")
    dates = {s.date().isoformat() for s in sessions}
    assert "2023-12-25" not in dates


def test_nyse_includes_half_day_2023_11_24() -> None:
    sessions = nyse_sessions("2023-11-20", "2023-11-30")
    dates = {s.date().isoformat() for s in sessions}
    assert "2023-11-24" in dates


def test_align_drops_weekend_rows() -> None:
    idx = pd.date_range("2023-01-02", "2023-01-15", freq="D", tz="UTC")
    df = pd.DataFrame({"x": range(len(idx))}, index=idx)
    aligned = align_to_sessions(df)
    assert (aligned.index.weekday < 5).all()


def test_align_forward_fill_caps_at_one_session() -> None:
    idx = pd.date_range("2023-01-02", "2023-02-01", freq="B", tz="UTC")
    df = pd.DataFrame({"x": range(len(idx))}, index=idx, dtype=float)
    # Knock out a wide consecutive band so the assertions survive the NYSE
    # holiday-aware reindex (which drops Jan 2 New Year observed + Jan 16 MLK
    # Day from the business-day range, shifting positional indices).
    df.iloc[5:12] = float("nan")
    aligned = align_to_sessions(df)
    # ffill(limit=1) refills exactly the first NaN; subsequent NaN sessions
    # in the same gap must stay NaN — that is the "cap at one session" claim.
    nan_count = aligned.iloc[:, 0].isna().sum()
    assert nan_count >= 3, (
        f"expected at least 3 surviving NaN rows after ffill(limit=1) on a "
        f"7-row NaN band; got {nan_count}"
    )


def test_align_preserves_tz_awareness() -> None:
    idx = pd.date_range("2023-01-02", "2023-01-15", freq="B", tz="UTC")
    df = pd.DataFrame({"x": range(len(idx))}, index=idx)
    aligned = align_to_sessions(df)
    assert aligned.index.tz is not None
    assert str(aligned.index.tz) == "UTC"
