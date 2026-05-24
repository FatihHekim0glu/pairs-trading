"""Unit tests for :mod:`pairs._exceptions`.

Verifies the hierarchy (every subclass derives from :class:`PairsError`
which itself derives from :class:`Exception`), public ``__all__`` coverage,
and that instances are raisable and carry messages.
"""

from __future__ import annotations

import pytest

from pairs import _exceptions as exc
from pairs._exceptions import (
    DegenerateSeriesError,
    InputError,
    InsufficientDataError,
    ManifestError,
    NonStationaryError,
    OOSReuseError,
    PairsError,
)

_SUBCLASSES = (
    InputError,
    NonStationaryError,
    DegenerateSeriesError,
    InsufficientDataError,
    OOSReuseError,
    ManifestError,
)


def test_pairs_error_is_exception_subclass() -> None:
    """The root error inherits from :class:`Exception`."""
    assert issubclass(PairsError, Exception)


@pytest.mark.parametrize("subclass", _SUBCLASSES)
def test_every_subclass_inherits_from_pairs_error(
    subclass: type[PairsError],
) -> None:
    """Each library exception derives from :class:`PairsError`."""
    assert issubclass(subclass, PairsError)


@pytest.mark.parametrize("subclass", _SUBCLASSES)
def test_each_exception_can_be_raised_and_caught_as_pairs_error(
    subclass: type[PairsError],
) -> None:
    """Each subclass is catchable as the root type."""
    with pytest.raises(PairsError):
        raise subclass("boom")


@pytest.mark.parametrize("subclass", _SUBCLASSES)
def test_each_exception_preserves_message(subclass: type[PairsError]) -> None:
    """The message argument round-trips through ``str()``."""
    instance = subclass("a message")
    assert str(instance) == "a message"


def test_subclasses_are_distinct_types() -> None:
    """Subclasses do not collapse into each other (no accidental aliases)."""
    types = set(_SUBCLASSES)
    assert len(types) == len(_SUBCLASSES)


def test_input_error_is_not_manifest_error() -> None:
    """Independent siblings are not mistakenly related."""
    with pytest.raises(InputError):
        raise InputError("x")
    assert not issubclass(InputError, ManifestError)
    assert not issubclass(ManifestError, InputError)


def test_dunder_all_matches_public_exports() -> None:
    """``__all__`` lists every public exception class exactly once."""
    expected = {
        "DegenerateSeriesError",
        "InputError",
        "InsufficientDataError",
        "ManifestError",
        "NonStationaryError",
        "OOSReuseError",
        "PairsError",
    }
    assert set(exc.__all__) == expected
    assert len(exc.__all__) == len(set(exc.__all__))


def test_dunder_all_attributes_exist_on_module() -> None:
    """Every name advertised in ``__all__`` is an attribute on the module."""
    for name in exc.__all__:
        assert hasattr(exc, name)


def test_exceptions_chain_with_from() -> None:
    """The ``raise X from Y`` idiom preserves ``__cause__``."""
    cause = ValueError("root")
    try:
        try:
            raise cause
        except ValueError as e:
            raise ManifestError("wrapped") from e
    except ManifestError as wrapped:
        assert wrapped.__cause__ is cause
