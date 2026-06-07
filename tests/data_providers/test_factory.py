"""Unit tests for :mod:`pairs.data_providers.factory`."""

from __future__ import annotations

import pytest

from pairs.data_providers.factory import make_provider
from pairs.data_providers.polygon import PolygonProvider
from pairs.data_providers.yfinance import YFinanceProvider


pytestmark = pytest.mark.unit


def test_make_provider_returns_yfinance_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``POLYGON_API_KEY``, the factory must return the yfinance fallback."""
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    provider = make_provider()
    assert isinstance(provider, YFinanceProvider)


def test_make_provider_returns_polygon_when_env_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-empty env key must select the real Polygon provider."""
    monkeypatch.setenv("POLYGON_API_KEY", "live-key")
    provider = make_provider()
    assert isinstance(provider, PolygonProvider)


def test_make_provider_explicit_key_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``api_key`` argument takes precedence over the env var."""
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    provider = make_provider(api_key="explicit-key")
    assert isinstance(provider, PolygonProvider)


def test_make_provider_blank_env_key_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty/whitespace env key must be treated as unset."""
    monkeypatch.setenv("POLYGON_API_KEY", "   ")
    provider = make_provider()
    assert isinstance(provider, YFinanceProvider)
