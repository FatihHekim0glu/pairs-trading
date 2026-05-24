"""Unit tests for :mod:`pairs._manifest`.

Covers config-hash determinism, build/write/read round-trips, corruption
detection (missing file, malformed JSON, wrong schema, missing key) and the
git / dep-hash helpers in offline mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pairs._exceptions import ManifestError
from pairs._manifest import (
    RunManifest,
    build_manifest,
    compute_config_hash,
    dep_hashes_for,
    git_sha,
    is_git_dirty,
    read_manifest,
    write_manifest,
)


def test_compute_config_hash_is_deterministic() -> None:
    """Repeated calls on the same payload produce the same digest."""
    cfg = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
    assert compute_config_hash(cfg) == compute_config_hash(cfg)


def test_compute_config_hash_is_order_insensitive() -> None:
    """Key ordering does not affect the digest (canonical JSON)."""
    a = compute_config_hash({"x": 1, "y": 2})
    b = compute_config_hash({"y": 2, "x": 1})
    assert a == b


def test_compute_config_hash_distinguishes_payloads() -> None:
    """Semantically different payloads produce different digests."""
    a = compute_config_hash({"x": 1})
    b = compute_config_hash({"x": 2})
    assert a != b


def test_compute_config_hash_handles_non_json_default() -> None:
    """Unknown types fall back to ``str()`` via the JSON default hook."""
    digest = compute_config_hash({"path": Path("/tmp/x")})
    assert isinstance(digest, str) and len(digest) == 32


def test_dep_hashes_for_default_includes_numpy() -> None:
    """The default distribution list at least covers ``numpy``."""
    hashes = dep_hashes_for()
    assert "numpy" in hashes
    assert hashes["numpy"] != "missing"


def test_dep_hashes_for_marks_missing_distributions() -> None:
    """Unknown distributions are recorded as ``"missing"``."""
    hashes = dep_hashes_for(["definitely-not-a-real-package-zzz"])
    assert hashes["definitely-not-a-real-package-zzz"] == "missing"


def test_dep_hashes_for_deduplicates_and_sorts() -> None:
    """Duplicates collapse and keys come back sorted."""
    hashes = dep_hashes_for(["numpy", "numpy"])
    assert list(hashes.keys()) == ["numpy"]


def test_git_sha_on_non_repo_returns_unknown(tmp_path: Path) -> None:
    """Running inside a non-git directory yields ``"unknown"``."""
    sha = git_sha(tmp_path)
    assert sha == "unknown"


def test_is_git_dirty_on_non_repo_is_false(tmp_path: Path) -> None:
    """Non-git directories are not considered dirty."""
    assert is_git_dirty(tmp_path) is False


def test_build_manifest_populates_required_fields(tmp_path: Path) -> None:
    """``build_manifest`` returns a fully populated dataclass."""
    manifest = build_manifest(
        config={"k": "v"},
        seed=7,
        extras={"note": "hello"},
        distributions=["numpy"],
        repo=tmp_path,
    )
    assert isinstance(manifest, RunManifest)
    assert manifest.schema_version == 1
    assert manifest.seed == 7
    assert manifest.extras == {"note": "hello"}
    assert "numpy" in manifest.dep_hashes
    assert manifest.config_hash == compute_config_hash({"k": "v"})


def test_build_manifest_extras_default_is_empty(tmp_path: Path) -> None:
    """When ``extras`` is ``None`` the field becomes an empty dict."""
    manifest = build_manifest(config={}, repo=tmp_path)
    assert manifest.extras == {}


def test_write_then_read_manifest_roundtrips(tmp_path: Path) -> None:
    """A manifest written to disk reads back to an equal dataclass."""
    original = build_manifest(
        config={"a": 1, "b": [1, 2]},
        seed=123,
        distributions=["numpy"],
        repo=tmp_path,
    )
    target = tmp_path / "out" / "manifest.json"
    written = write_manifest(target, original)
    assert written == target
    assert target.is_file()
    loaded = read_manifest(target)
    assert loaded == original


def test_write_manifest_is_canonical_json(tmp_path: Path) -> None:
    """Output uses sorted keys and trailing newline."""
    manifest = build_manifest(config={}, repo=tmp_path)
    target = tmp_path / "manifest.json"
    write_manifest(target, manifest)
    text = target.read_text(encoding="utf-8")
    assert text.endswith("\n")
    payload = json.loads(text)
    assert list(payload.keys()) == sorted(payload.keys())


def test_read_manifest_missing_file_raises(tmp_path: Path) -> None:
    """A nonexistent manifest path raises ``ManifestError``."""
    with pytest.raises(ManifestError, match="manifest not found"):
        read_manifest(tmp_path / "does-not-exist.json")


def test_read_manifest_malformed_json_raises(tmp_path: Path) -> None:
    """Non-JSON content is reported as a read failure."""
    target = tmp_path / "bad.json"
    target.write_text("{not: json", encoding="utf-8")
    with pytest.raises(ManifestError, match="unable to read manifest"):
        read_manifest(target)


def test_read_manifest_unsupported_schema_raises(tmp_path: Path) -> None:
    """A wrong ``schema_version`` is rejected."""
    target = tmp_path / "wrong-schema.json"
    target.write_text(
        json.dumps({"schema_version": 999, "utc_ts": ""}), encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="unsupported manifest schema_version"):
        read_manifest(target)


def test_read_manifest_missing_required_field_raises(tmp_path: Path) -> None:
    """A manifest missing required keys is flagged as malformed."""
    target = tmp_path / "incomplete.json"
    target.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ManifestError, match="malformed manifest"):
        read_manifest(target)


def test_read_manifest_corruption_detected_after_edit(tmp_path: Path) -> None:
    """Tampering with a written manifest is surfaced on reload."""
    manifest = build_manifest(config={"k": "v"}, repo=tmp_path)
    target = tmp_path / "m.json"
    write_manifest(target, manifest)
    # Drop a required key to simulate corruption.
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload.pop("utc_ts")
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError):
        read_manifest(target)
