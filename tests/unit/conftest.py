"""Unit-test conftest.

Marks every test under ``tests/unit`` with :pytest.mark.unit so the suite can
be filtered with ``pytest -m unit``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit
