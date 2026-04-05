# Contributing

## Local Setup

```bash
uv sync --dev
```

Optional local runtime overrides:

```bash
cp config.example.toml config.toml
```

`config.toml` is gitignored and intended for machine-specific paths or debug settings.

## Common Commands

```bash
# Lint and format
uv run ruff check .
uv run ruff format --check .

# Full test suite
uv run pytest -q

# CI-style core coverage gate
uv run pytest -q \
  --cov=dcs.cli \
  --cov=dcs.router \
  --cov=dcs.types \
  --cov=eval.metrics \
  --cov=eval.runner \
  --cov-report=term-missing \
  --cov-report=xml:coverage-core.xml \
  --cov-fail-under=80

# Full coverage snapshot
uv run pytest -q \
  --cov=dcs \
  --cov=benchmarks \
  --cov=eval \
  --cov-report=term \
  --cov-report=xml:coverage-full.xml

# Build package artifacts
uv build
```

Run the two coverage commands sequentially. Running them in parallel can corrupt the local coverage totals.

## Repo-Local Hooks

Enable the repo-local hooks once:

```bash
git config core.hooksPath .githooks
```

Current hooks:

- `pre-commit`: auto-fix and format staged Python files with Ruff
- `pre-push`: run `uv run pytest -q`

## Project Notes

- Use `uv` for Python commands in this repo.
- Prefer small, accuracy-preserving changes over broad rewrites.
- If you change packaging metadata in `pyproject.toml`, refresh `uv.lock` with `uv lock`.
