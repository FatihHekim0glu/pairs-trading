"""Session state initialization for the dashboard."""

from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

DEFAULTS: dict[str, Any] = {
    "universe_name": "curated_25_v1",
    "train_start": date(2018, 1, 1),
    "train_end": date(2022, 12, 31),
    "oos_start": date(2023, 1, 1),
    "oos_end": date.today(),
    "selected_pair": None,
    "cost_profile": "large_cap_realistic",
    "slippage_bps": 2,
    "z_entry": 2.0,
    "z_exit": 0.5,
    "z_stop": 3.0,
    "lookback_days": 252,
    "last_scan_result": None,
    "last_backtest_result": None,
    "disclaimer_dismissed": False,
}


def init_state() -> None:
    """Populate ``st.session_state`` with default keys if missing."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
