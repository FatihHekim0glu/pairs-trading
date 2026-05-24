"""Tests for the trial-log persistence layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from pairs._exceptions import OOSReuseError, PairsError
from pairs.evaluation import EvaluationProtocol, TrialLog


def test_persists_across_instances(tmp_path: Path) -> None:
    log_path = tmp_path / "trials.json"
    a = TrialLog(log_path)
    trial_id = a.start_trial("hash-A")
    a.record_result(trial_id, {"sharpe": 0.42}, spec_hash="hash-A")
    del a
    b = TrialLog(log_path)
    assert b.count_for_hash("hash-A") == 1


def test_oos_reuse_raises_without_bump(
    tmp_path: Path,
    synthetic_prices,
    mock_pair_selector,
    mock_pair_backtester,
) -> None:
    log = TrialLog(tmp_path / "trials.json")
    log.start_trial("spec-1")
    protocol = EvaluationProtocol(
        train_min_years=1.0,
        purge_days=0,
        embargo_pct=0.0,
        trial_log=log,
        rng_seed=0,
    )
    with pytest.raises(OOSReuseError):
        protocol.run(
            synthetic_prices,
            pair_selector=mock_pair_selector,
            pair_backtester=mock_pair_backtester,
            spec_hash="spec-1",
        )


def test_oos_reuse_allowed_with_new_hash(tmp_path: Path) -> None:
    log = TrialLog(tmp_path / "trials.json")
    log.start_trial("hash-1")
    assert log.count_for_hash("hash-2") == 0
    new_id = log.start_trial("hash-2")
    assert new_id == 0


def test_trial_log_corruption_raises(tmp_path: Path) -> None:
    log_path = tmp_path / "trials.json"
    log_path.write_text("{not json", encoding="utf-8")
    log = TrialLog(log_path)
    with pytest.raises(PairsError):
        log.count_for_hash("hash-x")
