"""Dismissible research-use disclaimer banner."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the disclaimer until the user dismisses it."""
    if st.session_state.get("disclaimer_dismissed"):
        return
    container = st.container()
    with container:
        st.warning(
            "Research and educational use only. Not investment advice. "
            "Past performance does not guarantee future results."
        )
        if st.button("Dismiss", key="disclaimer_dismiss_btn"):
            st.session_state.disclaimer_dismissed = True
            st.rerun()
