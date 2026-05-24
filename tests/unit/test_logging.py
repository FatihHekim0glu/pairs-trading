"""Unit tests for :mod:`pairs._logging`.

Covers idempotent handler attachment, level/format updates on repeated calls,
custom stream and format strings, and that the library does not poison the
root logger.
"""

from __future__ import annotations

import io
import logging

import pytest

from pairs._logging import enable_default


@pytest.fixture(autouse=True)
def _reset_pairs_logger() -> None:
    """Strip the ``pairs`` logger of any handlers between tests."""
    logger = logging.getLogger("pairs")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = True
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = True


def test_enable_default_attaches_named_handler() -> None:
    """First call attaches exactly one named handler."""
    handler = enable_default("INFO")
    logger = logging.getLogger("pairs")
    assert handler in logger.handlers
    assert handler.name == "pairs._logging.default"
    assert sum(h.name == "pairs._logging.default" for h in logger.handlers) == 1


def test_enable_default_is_idempotent() -> None:
    """Subsequent calls do not stack handlers."""
    first = enable_default("INFO")
    second = enable_default("INFO")
    third = enable_default("INFO")
    logger = logging.getLogger("pairs")
    assert first is second is third
    assert sum(h.name == "pairs._logging.default" for h in logger.handlers) == 1


def test_enable_default_updates_level_on_repeat_call() -> None:
    """A second call with a different level mutates the existing handler."""
    enable_default("INFO")
    handler = enable_default("DEBUG")
    assert handler.level == logging.DEBUG
    assert logging.getLogger("pairs").level == logging.DEBUG


def test_enable_default_updates_format_on_repeat_call() -> None:
    """Format changes are applied to the existing handler in place."""
    enable_default("INFO", fmt="A %(message)s")
    handler = enable_default("INFO", fmt="B %(message)s")
    assert handler.formatter is not None
    assert handler.formatter._fmt == "B %(message)s"


def test_enable_default_writes_to_supplied_stream() -> None:
    """The custom stream receives formatted records."""
    sink = io.StringIO()
    enable_default("INFO", stream=sink)
    logging.getLogger("pairs.test").info("hello")
    output = sink.getvalue()
    assert "hello" in output
    assert "INFO" in output


def test_enable_default_disables_propagation() -> None:
    """The ``pairs`` logger does not propagate to root after configuration."""
    enable_default("INFO")
    assert logging.getLogger("pairs").propagate is False


def test_enable_default_accepts_string_or_int_levels() -> None:
    """Both ``"DEBUG"`` and :data:`logging.DEBUG` work."""
    handler_str = enable_default("DEBUG")
    assert handler_str.level == logging.DEBUG
    handler_int = enable_default(logging.WARNING)
    assert handler_int.level == logging.WARNING


def test_enable_default_uses_default_format_when_none() -> None:
    """When ``fmt`` is ``None`` the documented default is applied."""
    handler = enable_default("INFO")
    assert handler.formatter is not None
    assert "%(asctime)s" in handler.formatter._fmt
    assert "%(levelname)" in handler.formatter._fmt


def test_enable_default_does_not_pollute_root_logger() -> None:
    """The root logger gets no handler from ``enable_default``."""
    root_before = list(logging.getLogger().handlers)
    enable_default("INFO")
    root_after = list(logging.getLogger().handlers)
    assert root_before == root_after
