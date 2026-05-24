# Reproducibility

This page is a placeholder. Detailed reproducibility guarantees, manifest
schema, dependency pinning policy, and the canonical workflow for re-running a
historical study are owned by the documentation agent.

## Building blocks

- `pairs.RunManifest` captures git revision, dependency versions, config hash,
  and seed for every run.
- `pairs.default_rng` and `pairs.derive_rng` produce deterministic NumPy
  generators.
- `scripts/check_reproducibility.py` re-hashes a stored configuration and
  asserts it matches the digest recorded in a manifest.
