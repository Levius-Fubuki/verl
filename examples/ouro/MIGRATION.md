# Shared engine interface migration — 2026-10-23

This update migrates the framework-side consumers and validates CPU behavior.
It does **not** complete Phase 2A acceptance against a combined new rollout and
weight-publication implementation. The upstream integration order remains open
in [issue #70](https://github.com/InferMatrixLab/vllm-rlt/issues/70).

## Contracts and source pins

| Source | Revision | Actual capability used |
|---|---|---|
| #83 rollout outputs | `45f0c6737dde1f5be858ffc05fcd35b866474459` | `logprobs`, engine-level `raw_logprobs`, effective seeds, stop IDs, speculative selected scores; **no publication/version API** |
| #78 publication reference | `0fb16324ccf02d8bc5fb88fb0a922a4c08b0ca19` | Full transactions and committed versions; retains `log_probs` and per-request mode |
| Current main | `4a350344027960a48384626aa646f1765fda6d4f` | New stage/client and scheduler boundaries underlying #83 |
| Historical training prototype | `bdce2876e3f78ac7efaa6cbf5783c185da2e9d19` | Original October 8 GPU acceptance only |
| verl baseline | `fbb4b3a8bf636f290c9c59fc346f756849e9c241` | Existing contributor-fork training framework |

The heads were inspected independently. No engine PRs were merged, monkeypatched
or copied together to manufacture a combined contract. The native Ouro model
only changed attention-metadata annotations between the historical engine and
#83: physical projection names, RoPE and differentiable replay remain compatible.

## Framework changes

- `backend.py` provides explicit `rollout_outputs` and `prototype` profiles.
  There is no attribute-guessing fallback after an incompatible call. The current
  profile reads `RequestOutput.logprobs`, requests `SamplingParams(logprobs=0)`
  and configures `logprobs_mode="raw_logprobs"` at engine construction.
- `TokenBatchDriver` delegates to a public facade or drives public
  `add_request`/`step`/`has_unfinished_requests`/`abort_request` methods. It preserves
  input order and handles foreign/duplicate outputs and incomplete batches.
  On public request endpoints it cleans up only owned requests after partial
  admission, exceptions or cancellation. Native-facade lifecycle is delegated;
  upstream `LLM.generate` catches `Exception`, so interrupt cleanup is not
  guaranteed there. Callers must close their owned backend in `finally`.
  P/D callers pass a raw public endpoint; `LLM(PDEngine(...))` is not required.
- Semantic `trace_ids` can be reused with raw endpoints while transport request
  IDs remain unique. A facade without trace-ID support is rejected before work;
  pass a public StageClient/PDEngine endpoint for this mode. `OuroRollout` forwards
  optional `prompts.non_tensor_batch["trace_id"]` values to that path.
- `OuroRollout.release/resume` gates the synchronous caller locally. It does not
  call unavailable engine pause/resume, free memory, or control unrelated clients.
  Callers must serialize generation and publication. Actual publication atomicity
  and cache invalidation remain responsibilities of the shared engine contract.
- Versioned rollout/training requires all four public methods:
  `start_weight_update`, `update_weights`, `finish_weight_update`, and
  `get_weight_version`. Missing methods fail before an owned worker, CUDA/FSDP
  initialization or export iteration. No policy version or worker weight is
  synthesized. Native #83 therefore cannot run `train.py` yet.
- Failed publication blocks generation and resume until a complete retry commits.
  Restoring sampler/publication counters requires matching profiles and full
  republication before generation; rejected restore leaves the cursor unchanged.
- PPO consumes current selected scores with explicit engine probability-mode
  provenance, actual seeds, traces and committed versions. Injected backends must
  declare `backend_logprobs_mode="raw_logprobs"`; absent/processed declarations
  are rejected. This is a caller configuration contract, not inferred metadata
  or an inspection of private worker state. Temperature 1 / unfiltered sampling
  restrictions on training remain unchanged.
- `validate_current.py` checks real current generation independently of versioned
  training. It reports `publication_validated=False` and `pd_validated=False`.
  `historical.py` seals the old GPU drivers to all 50 originally qualified engine
  Python-file hashes, preventing their private graph probes from running on an
  unrelated stage/client implementation.

## Reproduce on CPU

Run from this verl fork. The tested environment is macOS arm64, Python 3.12.13,
torch 2.12.1 and transformers 5.12.1. The exact dependency snapshot is
[requirements-current-cpu.lock](requirements-current-cpu.lock); it is separate
from the inherited CUDA image environment used in October 8 experiments.

```bash
uv venv --python 3.12 .venv-ouro-cpu
uv pip install --python .venv-ouro-cpu/bin/python -r examples/ouro/requirements-current-cpu.lock
PY="$PWD/.venv-ouro-cpu/bin/python"
export OMP_NUM_THREADS=2

# Two independent checkouts. Do not merge these PRs locally.
git clone https://github.com/InferMatrixLab/vllm-rlt.git ../engine-current
git -C ../engine-current fetch origin pull/83/head
git -C ../engine-current checkout --detach 45f0c6737dde1f5be858ffc05fcd35b866474459
git clone https://github.com/InferMatrixLab/vllm-rlt.git ../engine-publication
git -C ../engine-publication fetch origin pull/78/head
git -C ../engine-publication checkout --detach 0fb16324ccf02d8bc5fb88fb0a922a4c08b0ca19

export PYTHONPATH=.:../engine-current
$PY -m pytest examples/ouro/tests -q
$PY -m examples.ouro.validate_current --output current-generation.json
$PY -m pytest ../engine-current/tests/test_rollout_outputs.py -q -m 'not gpu'

# Separate CPU publication reference, with explicit legacy probability profile.
PYTHONPATH=.:../engine-publication $PY -m pytest \
  examples/ouro/tests/test_publication.py examples/ouro/tests/test_rollout.py -q
```

For historical regressions, use a third engine checkout pinned to `bdce287` and
`PYTHONPATH=.:../engine-training`. The existing historical commands in README
explicitly select `--engine-contract prototype`. Current-only tests report their
capability skips; CUDA tests are skipped on this CPU machine. These skips are
neither failures nor acceptance of the missing features.

## Evidence and remaining acceptance

Recorded results and source hashes are under
[results/2026-10-23](results/2026-10-23). The current generation report verifies
full depth and early exit through actual LLM/StageClient child processes, plus
speculative target scores through the native StageEngine. Independently replayed
selected-score maximum error is **4.77e-7** on the tiny FP32 CPU model. Returned
entropy seeds reproduce tokens and depths. Stop, two-token abort, request-ID reuse
and zero remaining KV blocks are checked through the public StageClient API.
This does not establish BF16 real-model parity or CUDA scheduling correctness.

Native #78 CPU publication/rollout checks are separate from #83 generation.
Modeled versioned-endpoint tests exercise adapter failure/provenance/restore
behavior only; they do not add publication support to #83. Historical model and
optimizer updates still refer to ROUND3 and their original revision.

After the shared rollout/publication route is chosen and implemented together,
pin that exact revision and rerun full-model BF16 training, FSDP export/recovery,
CUDA Graphs, asynchronous readbacks, speculative scoring and real two-GPU NIXL P/D
publication. No GPU server was started in this migration. Phase 3 N+2 predictor
training and runtime integration remain a separate next effort owned by
Levius-Fubuki; no N+2 implementation or speedup is claimed here.

AI assistance: OpenAI Codex implemented and executed automated tests. The fork
PR remains a draft for human review; no human review or independent human test
attestation is inferred from automated spec/quality review.
