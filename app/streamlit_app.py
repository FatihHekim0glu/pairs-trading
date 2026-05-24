"""Streamlit entrypoint for the pairs trading dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/streamlit_app.py` puts only `app/` on sys.path, not the
# project root, so `from app.X import Y` cannot find a top-level `app` package.
# Prepend the project root so `app.state`, `app.components`, etc. resolve from
# any of the four page modules loaded by st.navigation.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.state import init_state

st.set_page_config(
    page_title="Pairs Trading",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

pages = [
    st.Page("pages/01_pair_finder.py", title="Pair Finder", icon=None),
    st.Page("pages/02_spread_explorer.py", title="Spread Explorer"),
    st.Page("pages/03_backtest.py", title="Backtest"),
    st.Page("pages/04_portfolio.py", title="Portfolio"),
]

nav = st.navigation(pages)
nav.run()
