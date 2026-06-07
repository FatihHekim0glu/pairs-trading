"""Typed provider error hierarchy.

Lets call sites distinguish auth (401/403) and rate-limit (429) failures from
generic data/parse errors. All inherit :class:`ProviderError` so a single
``except ProviderError`` still catches everything.

The names are vendor-neutral on purpose -- the yfinance fallback and any
future provider can raise the same hierarchy.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for every data-provider failure."""


class ProviderAuthError(ProviderError):
    """Raised on HTTP 401/403 -- missing or invalid API key."""


class ProviderRateLimitError(ProviderError):
    """Raised after exhausting retries on HTTP 429."""


class ProviderDataError(ProviderError):
    """Raised when the response payload is missing expected fields or malformed."""
