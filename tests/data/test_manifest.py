"""Tests for ``pairs.data.manifest`` (and the SHA helper it depends on)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from pairs.data.cache import _sha256_file
from pairs.data.manifest import (
    ManifestEntry,
    _atomic_write_json,
    build_entry,
    load_manifest,
    update_manifest_entry,
    verify_manifest,
)


def test_manifest_entry_frozen(frozen_manifest_entry: ManifestEntry) -> None:
    with pytest.raises(ValidationError):
        frozen_manifest_entry.relpath = "different"  # type: ignore[misc]


def test_manifest_roundtrip_json(
    tmp_path: Path,
    frozen_manifest_entry: ManifestEntry,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    updated = update_manifest_entry(manifest_path, frozen_manifest_entry)
    assert updated.entries[frozen_manifest_entry.relpath] == frozen_manifest_entry
    reloaded = load_manifest(manifest_path)
    assert reloaded == updated


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=25)
@given(
    payload=st.binary(min_size=64, max_size=512),
    flip_index=st.integers(min_value=0, max_value=63),
)
def test_verify_manifest_detects_byte_flip(
    tmp_path_factory: pytest.TempPathFactory,
    payload: bytes,
    flip_index: int,
) -> None:
    tmp = tmp_path_factory.mktemp("verify")
    target = tmp / "shard.bin"
    target.write_bytes(payload)
    manifest_path = tmp / "manifest.json"
    entry = ManifestEntry(
        relpath="shard.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        rows=0,
        start="2020-01-01T00:00:00+00:00",
        end="2020-01-02T00:00:00+00:00",
        provider="test",
        written_at="2026-05-23T00:00:00+00:00",
    )
    update_manifest_entry(manifest_path, entry)
    assert verify_manifest(manifest_path) == {"shard.bin": "ok"}
    # Flip one byte: digest must no longer match.
    raw = bytearray(target.read_bytes())
    idx = flip_index % len(raw)
    raw[idx] ^= 0xFF
    target.write_bytes(bytes(raw))
    assert verify_manifest(manifest_path) == {"shard.bin": "mismatch"}
    target.unlink()
    assert verify_manifest(manifest_path) == {"shard.bin": "missing"}


def test_atomic_write_survives_simulated_crash(tmp_path: Path) -> None:
    """If a crash happens between tmp write and replace, original is intact."""
    target = tmp_path / "thing.json"
    # Seed an "existing" version.
    target.write_text(json.dumps({"version": "original"}), encoding="utf-8")
    tmp_path_candidate = target.with_suffix(target.suffix + ".tmp")
    # Simulate the partial state: tmp file written, replace never called.
    tmp_path_candidate.write_text(json.dumps({"version": "new"}), encoding="utf-8")
    # Reader of the canonical path sees the *original*.
    assert json.loads(target.read_text(encoding="utf-8"))["version"] == "original"
    # Cleanup loop (what a recovery step would do) removes the orphan tmp.
    if tmp_path_candidate.exists():
        tmp_path_candidate.unlink()
    # Now perform a real atomic write and confirm it succeeds and leaves no tmp.
    _atomic_write_json({"version": "new"}, target)
    assert json.loads(target.read_text(encoding="utf-8"))["version"] == "new"
    assert not tmp_path_candidate.exists()


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=40)
@given(payload=st.binary(min_size=0, max_size=256 * 1024))
def test_sha256_chunked_matches_oneshot(
    tmp_path_factory: pytest.TempPathFactory,
    payload: bytes,
) -> None:
    tmp = tmp_path_factory.mktemp("sha")
    target = tmp / "blob.bin"
    target.write_bytes(payload)
    assert _sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_build_entry_populates_fields(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(b"hello world")
    entry = build_entry(
        relpath="blob.bin",
        file_path=target,
        rows=1,
        start="2020-01-01",
        end="2020-01-02",
        provider="test",
    )
    assert entry.bytes == os.path.getsize(target)
    assert entry.sha256 == hashlib.sha256(b"hello world").hexdigest()
