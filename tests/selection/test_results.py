"""Invariant tests for the selection result dataclasses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs.selection.results import RWResult, ScreenResult


def test_screenresult_requires_dataframe() -> None:
    with pytest.raises(TypeError):
        ScreenResult(
            diagnostics=[],  # type: ignore[arg-type]
            coint_results={},
            method="fdr_bh",
            alpha=0.1,
            asof=pd.Timestamp("2024-01-01"),
        )


def test_screenresult_alpha_bounds() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ScreenResult(
            diagnostics=pd.DataFrame(),
            coint_results={},
            method="fdr_bh",
            alpha=0.0,
            asof=pd.Timestamp("2024-01-01"),
        )


def test_screenresult_requires_timestamp() -> None:
    with pytest.raises(TypeError):
        ScreenResult(
            diagnostics=pd.DataFrame(),
            coint_results={},
            method="fdr_bh",
            alpha=0.1,
            asof="2024-01-01",  # type: ignore[arg-type]
        )


def test_screenresult_surviving_pairs_empty() -> None:
    res = ScreenResult(
        diagnostics=pd.DataFrame(),
        coint_results={},
        method="fdr_bh",
        alpha=0.1,
        asof=pd.Timestamp("2024-01-01"),
    )
    assert res.surviving_pairs == []


def test_rwresult_invariants() -> None:
    series = pd.Series([0.01, 0.02], index=["a", "b"])
    res = RWResult(
        surviving_pair_ids=["a"],
        adjusted_pvalues=series,
        null_distribution=np.array([1.0, 2.0]),
        block_length=3,
        n_boot=99,
    )
    assert res.block_length == 3
    assert res.n_boot == 99


def test_rwresult_rejects_bad_block_length() -> None:
    with pytest.raises(ValueError, match="block_length"):
        RWResult(block_length=0, n_boot=10)


def test_rwresult_rejects_bad_n_boot() -> None:
    with pytest.raises(ValueError, match="n_boot"):
        RWResult(block_length=1, n_boot=-1)


def test_rwresult_rejects_non_series() -> None:
    with pytest.raises(TypeError):
        RWResult(adjusted_pvalues=[0.1, 0.2])  # type: ignore[arg-type]


def test_rwresult_rejects_pvalues_out_of_range() -> None:
    with pytest.raises(ValueError, match="adjusted_pvalues"):
        RWResult(
            adjusted_pvalues=pd.Series([0.5, 1.5]),
            block_length=1,
            n_boot=10,
        )
