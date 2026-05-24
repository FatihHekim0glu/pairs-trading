"""Verify that a run can be reproduced from its manifest.

Loads a stored :class:`pairs.RunManifest`, recomputes the configuration hash
from a provided config file (JSON or YAML), and exits non-zero if the digests
disagree. Optional checks compare the git SHA and dependency hashes against the
current environment.

Usage
-----
::

    python scripts/check_reproducibility.py --manifest runs/2026-05-22.json \\
        --config configs/baseline.yaml --check-git --check-deps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from pairs import RunManifest, read_manifest
from pairs._manifest import compute_config_hash, dep_hashes_for, git_sha


def _load_config(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML config file into a dictionary."""
    text: str = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        loaded: Any = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        msg: str = f"config root must be a mapping, got {type(loaded).__name__}"
        raise SystemExit(msg)
    return loaded


def _check_config(manifest: RunManifest, config: dict[str, Any]) -> list[str]:
    """Return a list of mismatch messages for the config-hash check."""
    actual: str = compute_config_hash(config)
    if actual == manifest.config_hash:
        return []
    return [f"config_hash mismatch: manifest={manifest.config_hash} actual={actual}"]


def _check_git(manifest: RunManifest) -> list[str]:
    """Return mismatch messages when the current git SHA differs."""
    current: str = git_sha()
    if current == "unknown":
        return ["git_sha unavailable in current environment"]
    if current != manifest.git_sha:
        return [f"git_sha mismatch: manifest={manifest.git_sha} actual={current}"]
    return []


def _check_deps(manifest: RunManifest) -> list[str]:
    """Return mismatch messages for dependency-hash divergences."""
    current: dict[str, str] = dep_hashes_for(list(manifest.dep_hashes))
    diffs: list[str] = []
    for name, recorded in manifest.dep_hashes.items():
        observed: str = current.get(name, "missing")
        if observed != recorded:
            diffs.append(f"dep {name}: manifest={recorded} actual={observed}")
    return diffs


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Re-verify a pairs-trading run manifest against the current environment.",
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Path to RunManifest JSON.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the configuration file (JSON or YAML) to re-hash.",
    )
    parser.add_argument(
        "--check-git",
        action="store_true",
        help="Also assert that the current git SHA matches the manifest.",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Also assert that installed dependency versions match the manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns ``0`` on success, ``1`` on any mismatch."""
    args: argparse.Namespace = _build_parser().parse_args(argv)
    manifest: RunManifest = read_manifest(args.manifest)
    config: dict[str, Any] = _load_config(args.config)

    failures: list[str] = []
    failures.extend(_check_config(manifest, config))
    if args.check_git:
        failures.extend(_check_git(manifest))
    if args.check_deps:
        failures.extend(_check_deps(manifest))

    if failures:
        for line in failures:
            print(f"FAIL: {line}", file=sys.stderr)
        return 1

    print("OK: manifest reproducible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
