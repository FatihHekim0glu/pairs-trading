"""Unit tests for :mod:`pairs._config`.

Covers defaults, environment-variable overrides via the ``PAIRS_`` prefix,
the frozen-model invariant, and the :func:`get_settings` LRU cache lifecycle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pairs._config import RuntimeSettings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Ensure :func:`get_settings` is rebuilt for every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_runtime_settings_defaults_are_sensible() -> None:
    """Default-constructed settings have expected shape and types."""
    settings = RuntimeSettings()
    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.cache_dir, Path)
    assert settings.log_level == "INFO"
    assert isinstance(settings.default_seed, int)
    assert settings.offline is False


def test_runtime_settings_is_frozen() -> None:
    """Mutating a field on the frozen model raises a validation error."""
    settings = RuntimeSettings()
    with pytest.raises((ValueError, TypeError)):
        settings.default_seed = 999  # type: ignore[misc]


def test_env_override_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """``PAIRS_LOG_LEVEL`` overrides the default."""
    monkeypatch.setenv("PAIRS_LOG_LEVEL", "DEBUG")
    settings = RuntimeSettings()
    assert settings.log_level == "DEBUG"


def test_env_override_default_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """``PAIRS_DEFAULT_SEED`` is parsed as an int."""
    monkeypatch.setenv("PAIRS_DEFAULT_SEED", "1234")
    settings = RuntimeSettings()
    assert settings.default_seed == 1234


def test_env_override_offline_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """``PAIRS_OFFLINE`` is parsed as a bool."""
    monkeypatch.setenv("PAIRS_OFFLINE", "true")
    settings = RuntimeSettings()
    assert settings.offline is True


def test_env_override_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``PAIRS_DATA_DIR`` overrides the per-user default."""
    custom = tmp_path / "data"
    monkeypatch.setenv("PAIRS_DATA_DIR", str(custom))
    settings = RuntimeSettings()
    assert settings.data_dir == custom


def test_env_extra_keys_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown ``PAIRS_*`` keys are silently dropped (``extra='ignore'``)."""
    monkeypatch.setenv("PAIRS_UNKNOWN_KEY_FOR_TEST", "x")
    # Should not raise:
    settings = RuntimeSettings()
    assert not hasattr(settings, "unknown_key_for_test")


def test_get_settings_returns_cached_singleton() -> None:
    """Repeated calls return the *same* instance until the cache is cleared."""
    first = get_settings()
    second = get_settings()
    assert first is second


def test_get_settings_cache_clear_returns_fresh_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After ``cache_clear`` env changes take effect."""
    first = get_settings()
    get_settings.cache_clear()
    monkeypatch.setenv("PAIRS_DEFAULT_SEED", "987654")
    second = get_settings()
    assert second is not first
    assert second.default_seed == 987654


def test_runtime_settings_invalid_seed_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-integer ``PAIRS_DEFAULT_SEED`` raises a validation error."""
    monkeypatch.setenv("PAIRS_DEFAULT_SEED", "not-an-int")
    with pytest.raises((ValueError, Exception)):
        RuntimeSettings()
