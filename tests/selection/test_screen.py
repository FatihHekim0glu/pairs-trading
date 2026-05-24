"""Tests for :mod:`pairs.selection.screen`.

The cointegration battery is mocked so the screen tests don't depend on
the cointegration sub-package being implemented yet.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pairs._rng import default_rng
from pairs.selection.results import Candidate
from pairs.selection.screen import screen_cointegration


@dataclass
class _StubEG:
    pvalue: float
    beta: float
    residuals: pd.Series


@dataclass
class _StubPipelineResult:
    eg: _StubEG
    cointegrated: bool = True


def _install_stub_pipeline(monkeypatch: pytest.MonkeyPatch, pvalues: dict[str, float]) -> None:
    """Inject a stub ``pairs.cointegration.full_pipeline`` into ``sys.modules``."""

    def _fake_full_pipeline(
        y0: pd.Series,
        y1: pd.Series,
        *,
        alpha: float,
        use_log: bool = True,
        n_boot: int = 0,
    ) -> _StubPipelineResult:
        a = str(y0.name).upper()
        b = str(y1.name).upper()
        key = f"{a}__{b}"
        pv = pvalues.get(key, 0.5)
        resid = pd.Series(np.zeros(len(y0)), index=y0.index)
        return _StubPipelineResult(eg=_StubEG(pvalue=pv, beta=1.0, residuals=resid))

    fake_module = types.ModuleType("pairs.cointegration")
    fake_module.full_pipeline = _fake_full_pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pairs.cointegration", fake_module)


def _make_panel(tickers: list[str]) -> pd.DataFrame:
    rng = default_rng(seed=20260531)
    T = 200
    idx = pd.date_range("2022-01-01", periods=T, freq="B")
    data = {t: np.exp(4.0 + np.cumsum(rng.standard_normal(T) * 0.01)) for t in tickers}
    return pd.DataFrame(data, index=idx)


def test_screen_cointegration_returns_screenresult_with_required_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tickers = ["AAA", "BBB", "CCC"]
    panel = _make_panel(tickers)
    candidates = [
        Candidate(ticker_a="AAA", ticker_b="BBB"),
        Candidate(ticker_a="AAA", ticker_b="CCC"),
    ]
    _install_stub_pipeline(
        monkeypatch,
        {
            "AAA__BBB": 0.001,
            "AAA__CCC": 0.5,
        },
    )
    window = (panel.index[0], panel.index[-1])
    result = screen_cointegration(
        candidates,
        panel,
        formation_window=window,
        alpha=0.10,
        mtc_method="fdr_bh",
    )
    expected = {
        "pair_id",
        "ticker_a",
        "ticker_b",
        "p_raw",
        "hedge_ratio",
        "half_life",
        "q_value",
        "survives_mtc",
    }
    assert expected.issubset(set(result.diagnostics.columns))
    assert len(result.diagnostics) == 2
    assert result.method == "fdr_bh"
    assert result.alpha == 0.10


def test_surviving_pairs_property(monkeypatch: pytest.MonkeyPatch) -> None:
    tickers = ["AAA", "BBB", "CCC"]
    panel = _make_panel(tickers)
    candidates = [
        Candidate(ticker_a="AAA", ticker_b="BBB"),
        Candidate(ticker_a="AAA", ticker_b="CCC"),
    ]
    _install_stub_pipeline(
        monkeypatch,
        {
            "AAA__BBB": 0.0001,
            "AAA__CCC": 0.9,
        },
    )
    window = (panel.index[0], panel.index[-1])
    result = screen_cointegration(
        candidates,
        panel,
        formation_window=window,
        alpha=0.10,
    )
    pairs = result.surviving_pairs
    assert ("AAA", "BBB") in pairs
    assert ("AAA", "CCC") not in pairs


def test_screen_handles_empty_candidate_list(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pipeline must NOT be called for empty input.
    fake_module = types.ModuleType("pairs.cointegration")

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("full_pipeline should not be called for empty input")

    fake_module.full_pipeline = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pairs.cointegration", fake_module)

    panel = _make_panel(["AAA"])
    window = (panel.index[0], panel.index[-1])
    result = screen_cointegration([], panel, formation_window=window)
    assert result.diagnostics.empty
    assert result.surviving_pairs == []


def test_invalid_window_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_pipeline(monkeypatch, {})
    panel = _make_panel(["AAA", "BBB"])
    with pytest.raises(Exception):  # noqa: B017 -- selection raises InputError
        screen_cointegration(
            [Candidate(ticker_a="AAA", ticker_b="BBB")],
            panel,
            formation_window=(panel.index[-1], panel.index[0]),
        )


def test_screen_diagnostic_schema_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    """The diagnostics frame must carry exactly the documented columns."""
    tickers = ["AAA", "BBB"]
    panel = _make_panel(tickers)
    _install_stub_pipeline(monkeypatch, {"AAA__BBB": 0.001})
    result = screen_cointegration(
        [Candidate(ticker_a="AAA", ticker_b="BBB")],
        panel,
        formation_window=(panel.index[0], panel.index[-1]),
        alpha=0.10,
    )
    expected = [
        "pair_id", "ticker_a", "ticker_b",
        "p_raw", "hedge_ratio", "half_life",
        "q_value", "survives_mtc",
    ]
    assert list(result.diagnostics.columns) == expected
    assert result.diagnostics["survives_mtc"].dtype == bool


def test_screen_handles_pipeline_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the cointegration pipeline raises PairsError the pair is skipped."""
    from pairs._exceptions import PairsError

    def _bad_pipeline(*_args: Any, **_kwargs: Any) -> Any:
        msg = "synthetic failure"
        raise PairsError(msg)

    fake_module = types.ModuleType("pairs.cointegration")
    fake_module.full_pipeline = _bad_pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pairs.cointegration", fake_module)

    panel = _make_panel(["AAA", "BBB"])
    result = screen_cointegration(
        [Candidate(ticker_a="AAA", ticker_b="BBB")],
        panel,
        formation_window=(panel.index[0], panel.index[-1]),
    )
    # All candidates failed, so the diagnostics frame is empty.
    assert result.diagnostics.empty


def test_screen_handles_missing_eg_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pipeline result without EG attributes triggers a warning and skip."""

    class _NoEG:
        pass

    def _pipeline(*_args: Any, **_kwargs: Any) -> Any:
        return _NoEG()

    fake_module = types.ModuleType("pairs.cointegration")
    fake_module.full_pipeline = _pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pairs.cointegration", fake_module)

    panel = _make_panel(["AAA", "BBB"])
    result = screen_cointegration(
        [Candidate(ticker_a="AAA", ticker_b="BBB")],
        panel,
        formation_window=(panel.index[0], panel.index[-1]),
    )
    assert result.diagnostics.empty


def test_screen_non_dataframe_raises() -> None:
    from pairs._exceptions import InputError

    with pytest.raises(InputError):
        screen_cointegration(
            [],
            np.zeros((10, 2)),  # type: ignore[arg-type]
            formation_window=(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")),
        )


def test_skip_missing_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_pipeline(monkeypatch, {"AAA__BBB": 0.01})
    panel = _make_panel(["AAA", "BBB"])
    candidates = [
        Candidate(ticker_a="AAA", ticker_b="BBB"),
        Candidate(ticker_a="AAA", ticker_b="ZZZ"),  # missing
    ]
    result = screen_cointegration(
        candidates,
        panel,
        formation_window=(panel.index[0], panel.index[-1]),
    )
    assert len(result.diagnostics) == 1
    assert result.diagnostics.iloc[0]["ticker_b"] == "BBB"
