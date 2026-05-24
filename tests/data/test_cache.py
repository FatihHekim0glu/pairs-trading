"""Tests for ``pairs.data.cache`` (atomic Parquet writes + chunked hashing)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pairs.data.cache import _atomic_write_parquet, _read_parquet, _sha256_file


def test_atomic_write_no_tmp_left_on_success(tmp_path: Path) -> None:
    target = tmp_path / "shard.parquet"
    df = pd.DataFrame({"x": np.arange(10)})
    _atomic_write_parquet(df, target)
    assert target.exists()
    assert not target.with_suffix(target.suffix + ".tmp").exists()
    roundtrip = _read_parquet(target)
    pd.testing.assert_frame_equal(roundtrip.reset_index(drop=True), df)


def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "shard.parquet"
    df_old = pd.DataFrame({"x": np.arange(5)})
    df_new = pd.DataFrame({"x": np.arange(20)})
    _atomic_write_parquet(df_old, target)
    first_size = target.stat().st_size
    _atomic_write_parquet(df_new, target)
    second_size = target.stat().st_size
    assert first_size != second_size
    pd.testing.assert_frame_equal(_read_parquet(target).reset_index(drop=True), df_new)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=30)
@given(payload=st.binary(min_size=0, max_size=64 * 1024))
def test_sha256_stable_under_reread(
    tmp_path_factory: pytest.TempPathFactory,
    payload: bytes,
) -> None:
    tmp = tmp_path_factory.mktemp("sha")
    target = tmp / "blob.bin"
    target.write_bytes(payload)
    first = _sha256_file(target)
    second = _sha256_file(target)
    assert first == second
    assert first == hashlib.sha256(payload).hexdigest()
