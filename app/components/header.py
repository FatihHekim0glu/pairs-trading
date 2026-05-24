"""Page header with title and shields.io badge row."""

from __future__ import annotations

import streamlit as st

OWNER = "noxire-dev"
REPO = "pairs-trading"

BADGES = [
    (
        "CI",
        f"https://img.shields.io/github/actions/workflow/status/{OWNER}/{REPO}/ci.yml"
        f"?branch=main&label=CI",
        f"https://github.com/{OWNER}/{REPO}/actions",
    ),
    (
        "Coverage",
        f"https://img.shields.io/codecov/c/github/{OWNER}/{REPO}?label=coverage",
        f"https://codecov.io/gh/{OWNER}/{REPO}",
    ),
    (
        "License",
        f"https://img.shields.io/github/license/{OWNER}/{REPO}",
        f"https://github.com/{OWNER}/{REPO}/blob/main/LICENSE",
    ),
    (
        "Demo",
        "https://img.shields.io/badge/demo-HF%20Spaces-yellow",
        f"https://huggingface.co/spaces/{OWNER}/{REPO}",
    ),
]


def render(page: str) -> None:
    """Render the dashboard header for a given page name."""
    st.title("Pairs Trading Research Lab")
    st.caption(page)
    badge_md = " ".join(
        f"[![{label}]({src})]({href})" for label, src, href in BADGES
    )
    st.markdown(badge_md)
    st.divider()
