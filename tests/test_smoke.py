"""Smoke tests for the public API.

These tests must remain dependency-light: they only validate import paths and
the version string so they can run before any other agent's work is wired up.
"""

from __future__ import annotations

import importlib
import re

import pytest

# PEP 440 release form, including pre/post/dev/local segments.
_PEP440: re.Pattern[str] = re.compile(
    r"^([1-9][0-9]*!)?"
    r"(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*"
    r"((a|b|rc)(0|[1-9][0-9]*))?"
    r"(\.post(0|[1-9][0-9]*))?"
    r"(\.dev(0|[1-9][0-9]*))?"
    r"(\+([a-z0-9]+(\.[a-z0-9]+)*))?$",
    re.IGNORECASE,
)


@pytest.mark.unit
def test_import_version() -> None:
    """``pairs.__version__`` is importable and PEP 440 compliant."""
    pairs = importlib.import_module("pairs")
    version: str = pairs.__version__
    assert isinstance(version, str)
    assert version, "version must be non-empty"
    assert _PEP440.match(version), f"{version!r} is not PEP 440 compliant"


@pytest.mark.unit
def test_public_api_complete() -> None:
    """Every name in ``pairs.__all__`` resolves to a real attribute."""
    pairs = importlib.import_module("pairs")
    missing: list[str] = [name for name in pairs.__all__ if not hasattr(pairs, name)]
    assert not missing, f"missing public symbols: {missing}"
