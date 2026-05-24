"""Muted footer with provenance and build info."""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime

import streamlit as st


def _resolve_version() -> str:
    try:
        from pairs import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _resolve_git_sha() -> str:
    sha = os.environ.get("GIT_SHA")
    if sha:
        return sha[:7]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("utf-8").strip()[:7]
    except Exception:
        return "unknown"


def render() -> None:
    """Render a one-line footer with data source and build metadata."""
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    version = _resolve_version()
    sha = _resolve_git_sha()
    st.divider()
    st.caption(
        f"Data: Yahoo Finance via yfinance | Fetched: {ts} | "
        f"pairs v{version} | Build {sha}"
    )
