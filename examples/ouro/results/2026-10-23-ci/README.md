# CI compatibility follow-up, 2026-10-23

The PR at `59a5b644` failed title validation, Ruff `UP038`, and license validation.
The title now uses allowed `recipe, fsdp, rollout` modules. Three `isinstance`
checks use Python union syntax for CI's pinned Ruff 0.12.2. All 28 Ouro Python
files now include explicit Apache copyright headers; the license checker adds
2026 individual contributors alongside its existing 2025 entry. No hook was
disabled or skipped.

The earlier `2026-10-08` and `2026-10-23` evidence and source manifests remain
historical snapshots of their original sources. This directory records the
post-fix source hashes and fresh CPU validation; it adds no GPU evidence.

From the workspace parent of `verl/`, the fresh validation was:

```bash
PYTHONPATH=engine-current:verl OMP_NUM_THREADS=2 .venv-migration-repro/bin/python -m pytest verl/examples/ouro/tests -q
```

Result: **84 passed, 30 skipped** against engine PR #83 at
`45f0c6737dde1f5be858ffc05fcd35b866474459`. Publication-dependent tests and
CUDA tests remain skipped as documented in the migration evidence.
See [current-tests.log](current-tests.log).

From `verl/`, using a separate Python 3.12.13 CI environment with pre-commit
4.7.0, hydra-core 1.3.2, and this checkout installed with `--no-deps`:

```bash
PATH="$PWD/../.venv-ci/bin:$PATH" ../.venv-ci/bin/pre-commit run --all-files --show-diff-on-failure
```

All 14 hooks passed, including pinned Ruff 0.12.2, mypy, license validation,
config generation, and compilation. See [pre-commit.log](pre-commit.log).
The frozen migration test environment was not modified.
