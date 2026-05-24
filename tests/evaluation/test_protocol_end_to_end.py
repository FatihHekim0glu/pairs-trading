"""End-to-end smoke test for the evaluation protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pairs.evaluation import EvaluationProtocol, ProtocolReport, TrialLog


@pytest.mark.slow
def test_protocol_synthetic_cointegrated_end_to_end(
    tmp_path: Path,
    synthetic_prices: pd.DataFrame,
    mock_pair_selector,
    mock_pair_backtester,
) -> None:
    log = TrialLog(tmp_path / "trials.json")
    protocol = EvaluationProtocol(
        train_min_years=1.5,
        purge_days=5,
        embargo_pct=0.0,
        n_groups=5,
        k_test=2,
        s_partitions=8,
        trial_log=log,
        rng_seed=42,
        bootstrap_replicates=50,
    )
    rng = np.random.default_rng(0)
    trial_returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(synthetic_prices.shape[0], 10)),
        index=synthetic_prices.index,
        columns=[f"trial_{i}" for i in range(10)],
    )
    benchmark = pd.Series(
        rng.normal(0.0, 0.01, size=synthetic_prices.shape[0]),
        index=synthetic_prices.index,
    )
    report: ProtocolReport = protocol.run(
        synthetic_prices,
        pair_selector=mock_pair_selector,
        pair_backtester=mock_pair_backtester,
        spec_hash="end-to-end-test",
        trial_returns=trial_returns,
        benchmark_returns=benchmark,
    )
    assert report.spec_hash == "end-to-end-test"
    assert report.trial_id == 0
    assert report.walk_forward.fold_count >= 1
    assert report.cpcv.n_combinations >= 1
    assert 0.0 <= report.dsr.dsr <= 1.0
    assert np.isfinite(report.hac_se)
    if report.pbo is not None:
        assert 0.0 <= report.pbo.pbo <= 1.0
    if report.memmel is not None:
        assert 0.0 <= report.memmel.p_value <= 1.0
    if report.spa is not None:
        assert 0.0 <= report.spa.p_value_consistent <= 1.0
