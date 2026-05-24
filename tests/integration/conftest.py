"""Integration-test conftest.

Marks every test under ``tests/integration`` with :pytest.mark.integration.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration
