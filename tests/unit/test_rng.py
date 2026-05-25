"""Unit tests for :mod:`pairs._rng`.

Covers determinism, default-seed wiring through
:class:`pairs._config.RuntimeSettings`, spawn-based child independence and
input validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pairs._config import get_settings
from pairs._rng import default_rng, derive_rng


def test_default_rng_explicit_seed_is_deterministic() -> None:
    """Same explicit seed yields the same first draws."""
    a = default_rng(42).standard_normal(8)
    b = default_rng(42).standard_normal(8)
    np.testing.assert_array_equal(a, b)


def test_default_rng_different_seeds_differ() -> None:
    """Different seeds yield different draws."""
    a = default_rng(1).standard_normal(8)
    b = default_rng(2).standard_normal(8)
    assert not np.array_equal(a, b)


def test_default_rng_returns_generator_instance() -> None:
    """Helper returns a NumPy ``Generator``."""
    assert isinstance(default_rng(0), np.random.Generator)


def test_default_rng_seed_none_uses_configured_default() -> None:
    """``seed=None`` resolves to ``RuntimeSettings.default_seed``."""
    get_settings.cache_clear()
    configured = get_settings().default_seed
    implicit = default_rng().standard_normal(8)
    explicit = default_rng(configured).standard_normal(8)
    np.testing.assert_array_equal(implicit, explicit)


def test_derive_rng_returns_generator() -> None:
    """A derived child is a ``Generator``."""
    parent = default_rng(0)
    child = derive_rng(parent, "child")
    assert isinstance(child, np.random.Generator)


def test_derive_rng_label_changes_stream() -> None:
    """Different labels from the same parent yield different streams."""
    parent_a = default_rng(0)
    parent_b = default_rng(0)
    a = derive_rng(parent_a, "alpha").standard_normal(8)
    b = derive_rng(parent_b, "beta").standard_normal(8)
    assert not np.array_equal(a, b)


def test_derive_rng_label_is_deterministic() -> None:
    """Same label + same parent seed reproduces the same child stream."""
    a = derive_rng(default_rng(0), "alpha").standard_normal(8)
    b = derive_rng(default_rng(0), "alpha").standard_normal(8)
    np.testing.assert_array_equal(a, b)


def test_derive_rng_long_label_is_accepted() -> None:
    """Labels of arbitrary length hash to a valid offset."""
    parent = default_rng(0)
    child = derive_rng(parent, "x" * 4096)
    assert isinstance(child, np.random.Generator)
    assert child.standard_normal(1).shape == (1,)


def test_derive_rng_empty_label_is_accepted() -> None:
    """The empty string is a valid (if uncommon) label."""
    child = derive_rng(default_rng(0), "")
    assert isinstance(child, np.random.Generator)


def test_derive_rng_rejects_non_generator_parent() -> None:
    """Plain ``RandomState`` or ``int`` parents raise ``TypeError``."""
    with pytest.raises(TypeError, match=r"numpy\.random\.Generator"):
        derive_rng(42, "child")  # type: ignore[arg-type]


def test_derive_rng_rejects_legacy_random_state() -> None:
    """Legacy ``RandomState`` is not a ``Generator`` and must be rejected."""
    with pytest.raises(TypeError):
        derive_rng(np.random.RandomState(0), "child")  # type: ignore[arg-type]


def test_derive_rng_child_independent_of_parent_after_spawn() -> None:
    """Drawing from a child does not consume the parent's stream."""
    parent_a = default_rng(0)
    parent_b = default_rng(0)
    derive_rng(parent_a, "ignored").standard_normal(100)
    # parent_a and parent_b should still produce the same next draws.
    np.testing.assert_array_equal(parent_a.standard_normal(8), parent_b.standard_normal(8))
