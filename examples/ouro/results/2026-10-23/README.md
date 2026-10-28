# CPU migration evidence — 2026-10-23

Commands and scope: [MIGRATION.md](../../MIGRATION.md).

| Validation | Result | Evidence |
|---|---|---|
| Adapter suite on #83 `45f0c67` | 84 passed, 30 skipped | `current-suite-final.log` |
| Adapter suite on historical `bdce287` | 92 passed, 22 skipped | `historical-suite-final.log` |
| Actual #78 `0fb16324` publication/rollout | 23 passed | `current-publication-cpu.log` |
| Native #83 rollout CPU tests | 173 passed, 13 GPU cases deselected | `upstream-rollout-cpu.log` |
| Adapter migration/provenance/restore | 11 passed | `migration-green.log` |
| Public backend/trace driver | 39 passed | `backend-trace-green.log` |
| Historical GPU drivers reject different engine before CUDA | 3 passed | `historical-guard-green.log` |
| Original 50 Python engine source hashes | Exact match | `historical-source-check.log` |
| Ruff lint/format | Passed, 28 Python files formatted | `lint-final.log`, `format-final.log` |
| Fresh CPU venv from exact snapshot | 91 compatible packages | `repro-install.log`, `dependency-check.log` |

The two full adapter suites and #78 publication checks used a fresh venv installed
from `requirements-current-cpu.lock`; other focused checks used the original
venv with identical pinned packages. Six expected warnings concern unavailable
optional verl acceleration backends and a Ray deprecation. Current #83 skips
23 publication-dependent cases and 7 CUDA cases. The historical run skips
current-only capabilities and CUDA; these skips are not feature acceptance.

`current-generation.json` comes from actual CPU execution: LLM/StageClient child
processes for full/early generation, native StageEngine speculative target-score
replay, and public StageClient stop/abort/ID-reuse cleanup. Selected-score error
is at most 4.76837158203125e-7 on the tiny FP32 model. It explicitly marks
publication and actual P/D transfer false. No request is assigned a fabricated
committed version.

`*-red.log` files retain the relevant failure-before-fix observations. Log copies
trim trailing whitespace only; the complete original outputs are backed up in
the workspace's `results/2026-10-23-migration` directory. The runtime/source
manifests distinguish all three independent engine checkouts and the tested
adapter sources. No GPU server was started and no N+2 experiment was performed.
