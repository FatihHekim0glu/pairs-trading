from __future__ import annotations

import numpy as np
import pandas as pd

from pairs.portfolio import correlation_filter, effective_n


def _panel(rng: np.random.Generator, n_days: int, n_pairs: int, rho: float) -> pd.DataFrame:
    common = rng.normal(size=n_days)
    idio = rng.normal(size=(n_days, n_pairs))
    a = float(np.sqrt(max(rho, 0.0)))
    b = float(np.sqrt(max(1.0 - rho, 0.0)))
    data = a * common[:, None] + b * idio
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    cols = [f"P{i}" for i in range(n_pairs)]
    return pd.DataFrame(data, index=idx, columns=cols)


def test_effective_n_perfectly_correlated_is_one() -> None:
    rng = np.random.default_rng(0)
    common = rng.normal(size=500)
    df = pd.DataFrame({f"P{i}": common for i in range(5)})
    assert abs(effective_n(df) - 1.0) < 1e-9


def test_effective_n_orthogonal_is_n() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(2000, 5)), columns=[f"P{i}" for i in range(5)])
    assert abs(effective_n(df) - 5.0) < 0.5


def test_effective_n_intermediate_closed_form() -> None:
    rng = np.random.default_rng(2)
    df = _panel(rng, n_days=5000, n_pairs=4, rho=0.5)
    # 4 / (1 + 3 * 0.5) = 4 / 2.5 = 1.6
    assert abs(effective_n(df) - 1.6) < 0.15


def test_correlation_filter_drops_lower_sharpe() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(size=500)
    # Two near-identical series; P0 has clearly higher mean (higher Sharpe).
    df = pd.DataFrame(
        {
            "P0": base + 0.005,
            "P1": base + 0.0001,
            "P2": rng.normal(size=500),
        }
    )
    survivors = correlation_filter(df, max_pairwise_corr=0.5, min_overlap=60)
    assert "P0" in survivors
    assert "P1" not in survivors
    assert "P2" in survivors


def test_correlation_filter_respects_min_overlap() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(size=500)
    df = pd.DataFrame({"P0": base, "P1": base * 0.999})
    df.loc[df.index[10:], "P1"] = np.nan  # only 10 overlapping rows
    survivors = correlation_filter(df, max_pairwise_corr=0.5, min_overlap=60)
    # Both kept because overlap < min_overlap.
    assert set(survivors) == {"P0", "P1"}


def test_correlation_filter_single_column_passthrough() -> None:
    df = pd.DataFrame({"P0": np.arange(100, dtype=float)})
    assert correlation_filter(df) == ["P0"]


def test_effective_n_handles_single_column() -> None:
    df = pd.DataFrame({"P0": np.arange(100, dtype=float)})
    assert effective_n(df) == 1.0
