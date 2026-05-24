"""Tests for the yfinance retry/backoff adapter.

The real network path is gated behind ``@pytest.mark.network``; CI skips it by
default. Everything here exercises the deterministic retry logic with an
rng-injected jitter and a stub sleep, so no wall time is consumed.
"""

from __future__ import annotations

from typing import Any

import pytest

from pairs._exceptions import InputError
from pairs._rng import default_rng
from pairs.data import _yfinance_adapter as adapter
from pairs.data._yfinance_adapter import _batch_download, retry_with_backoff


def test_backoff_retries_then_succeeds() -> None:
    sleeps: list[float] = []
    rng = default_rng(seed=42)
    calls: list[int] = []

    @retry_with_backoff(max_attempts=4, base=2.0, jitter=True, rng=rng, sleep=sleeps.append)
    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("transient")
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 3
    # Two intermediate sleeps before the successful third attempt.
    assert len(sleeps) == 2
    assert all(s > 0 for s in sleeps)


def test_backoff_gives_up_after_max_attempts() -> None:
    sleeps: list[float] = []
    rng = default_rng(seed=7)
    calls: list[int] = []

    @retry_with_backoff(max_attempts=3, base=1.2, jitter=False, rng=rng, sleep=sleeps.append)
    def always_fails() -> None:
        calls.append(1)
        raise ValueError("nope")

    with pytest.raises(ValueError):
        always_fails()
    assert len(calls) == 3
    # Sleeps between attempts 1->2 and 2->3 only.
    assert len(sleeps) == 2


def test_offline_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stub:
        offline: bool = True

    def _stub_settings() -> Any:
        return _Stub()

    monkeypatch.setattr(adapter, "get_settings", _stub_settings)
    with pytest.raises(InputError, match="offline"):
        _batch_download(["AAA"], "2020-01-01", "2020-02-01")


def test_retry_with_backoff_rejects_zero_attempts() -> None:
    with pytest.raises(InputError):
        retry_with_backoff(max_attempts=0)


@pytest.mark.network
def test_yfinance_real_download() -> None:  # pragma: no cover - network only
    """Smoke test that hits real yfinance. Skipped by default."""
    pytest.importorskip("yfinance")
    df = _batch_download(["AAPL"], "2024-01-02", "2024-01-10")
    assert not df.empty
