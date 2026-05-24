"""Emit a :class:`pairs.RunManifest` for the current environment.

The manifest captures git revision, dependency versions, Python and platform
metadata, an optional seed, and a hash of the supplied configuration. The
result is written to disk as canonical JSON.

Usage
-----
::

    python scripts/make_manifest.py --config configs/baseline.yaml \\
        --output runs/2026-05-22.json --seed 20260523
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from pairs import write_manifest
from pairs._manifest import build_manifest


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


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Write a RunManifest describing the current environment and config.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the configuration file (JSON or YAML) to hash.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where the manifest JSON will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional top-level random seed to record on the manifest.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Free-form key=value pair appended to the manifest 'extras' map. Repeatable.",
    )
    parser.add_argument(
        "--distribution",
        action="append",
        default=None,
        metavar="NAME",
        help="Extra distribution to record beyond the default scientific stack. Repeatable.",
    )
    return parser


def _parse_extras(pairs_in: list[str]) -> dict[str, Any]:
    """Parse ``KEY=VALUE`` strings into a dictionary."""
    out: dict[str, Any] = {}
    for item in pairs_in:
        if "=" not in item:
            msg: str = f"--extra must be KEY=VALUE, got {item!r}"
            raise SystemExit(msg)
        key, _, value = item.partition("=")
        out[key.strip()] = value.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns ``0`` on success."""
    args: argparse.Namespace = _build_parser().parse_args(argv)
    config: dict[str, Any] = _load_config(args.config)
    manifest = build_manifest(
        config=config,
        seed=args.seed,
        extras=_parse_extras(args.extra),
        distributions=args.distribution,
    )
    out_path: Path = write_manifest(args.output, manifest)
    print(f"wrote manifest to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
