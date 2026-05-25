# Contributing

This is currently a single-maintainer repository. Pull requests are welcome but expect a slow review cadence.

## Setting up the dev environment

The project uses `uv` for dependency resolution and an editable install for development.

```bash
git clone https://github.com/FatihHekim0glu/pairs-trading && cd pairs-trading
uv sync --all-extras
```

If you prefer `pip`:

```bash
pip install -e ".[dev,app]"
```

## Running tests

```bash
pytest -q
```

Coverage thresholds and slow-test markers are configured in `pyproject.toml`.

## Style

- Formatter and linter: `ruff` (run `ruff check . && ruff format .`)
- Type checker: `mypy --strict src`
- Public APIs need docstrings; private helpers do not.
- No print statements in library code; use the project logger.

## Commit messages

Follow Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`). Keep the subject line under 72 characters.

## Pull request checklist

Before opening a PR, confirm:

- [ ] `pytest -q` passes locally
- [ ] `ruff check .` is clean
- [ ] `mypy --strict src` is clean
- [ ] No vendor-attribution lines, generator stamps, or co-author trailers added by tooling
- [ ] If you changed any number that feeds the money chart, regenerate it via `app/plots/is_vs_oos_bars.py`
- [ ] README stays at or below 300 lines

## Reporting issues

Use the templates in `.github/ISSUE_TEMPLATE/`. Bug reports without a minimal reproduction will be closed.
