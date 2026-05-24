"""Property-test conftest.

Marks every test under ``tests/property`` with :pytest.mark.property.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.property
