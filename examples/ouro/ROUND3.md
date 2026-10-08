# Phase 2A dual-GPU acceptance — 2026-10-08

This records the tested contributor-fork integration, not production Ray-trainer
acceptance or a model-quality benchmark. Commands are in [README.md](README.md).
The raw JSON and final check logs are in [results/2026-10-08](results/2026-10-08).

## Hardware and provenance

Two RTX 4090 GPUs expose 49,140 MiB each, PCIe PHB, no CUDA peer access. NCCL and
the default NIXL/UCX two-owner transfer both worked without a transport override.
The machine provides 24 CPU cores and 144 GiB cgroup memory. This supports the
local training/rollout coexistence and distinct GPU0-prefill/GPU1-decode jobs.

- verl baseline: `fbb4b3a8bf636f290c9c59fc346f756849e9c241`
- Engine prototype: `bdce2876e3f78ac7efaa6cbf5783c185da2e9d19`
- Model revision: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`
- Model file SHA256: `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`
- Physical parameter count: **1,434,652,673**, one recurrent core, frozen exit gate.
- Runtime: Python 3.12.3, torch 2.12.1+cu130, triton 3.7.1,
  transformers 5.12.1, tensordict 0.10.0, nixl/nixl-cu13 1.4.1.

The exact [runtime imports](results/2026-10-08/runtime-provenance.json),
[source hashes](results/2026-10-08/source-manifest.json) and
[model hash](results/2026-10-08/model-sha256.txt) accompany the evidence. All 50
engine Python files matched the pinned local checkout; all 21 Ouro Python files
matched the tested remote copy. The worker lock snapshot was installed in a fresh
`uv --system-site-packages` venv. The final 53 adapter tests, engine tests and
full-depth run used that new venv. Adaptive/PD runs used the original venv with
the same pinned dependency snapshot. This validates an inherited image runtime,
not verl's upstream torch-2.13 lockfile.

## Results

| Acceptance job | Observed result | Evidence |
|---|---|---|
| Adapter/model/HF/publication/rollout/export/recipe suite | 53 passed; 10 dependency/deprecation warnings | `tests-final.log` |
| Native engine training contract on CUDA | 6 passed | `engine-contract-final.log` |
| Native engine CPU regression | 423 passed, 141 GPU tests skipped | `engine-cpu-final.log` |
| Ruff check / format | All checks passed; 21 Python files formatted | `lint-final.log`, `format-final.log` |
| True two-rank shared-layer gradient oracle | Exact parameter equality with averaged-gradient reference | `distributed-oracle.json` |
| Per-unit physical FSDP export, two ranks | 4 tests passed on each rank, including owned tensors and bounded GPU peak | `export-two-rank.log` |
| Full-depth Ouro1.4B RL | 3 fresh rollout/update/publication iterations; finite nonzero gradients; FP32 master/Adam | `train-full-depth.json` |
| Adaptive-exit Ouro1.4B RL | 3 fresh iterations; same gates, checkpoint at step 2 | `train-streaming.json` |
| New-process two-rank resume | Exact next-step equality across 8 fields | `train-streaming-resumed.json`, `streaming-resume-comparison.json` |
| Local serving publication matrix | 8 modes match fresh engines; selected log-probability error 0 | `adapter-matrix.json` |
| NIXL P/D eager + graphs | Both owners version 1, cache blocks released, selected score error 0 | `adapter-pd.json` |

The per-unit export peak regression failed with the old bulk snapshot:
7,920,128 bytes of additional CUDA allocations exceeded its 3,411,968-byte tiny
model budget. The replacement passed for world sizes 1 and 2. The budget includes
the active physical unit and owned output tensors; it does not claim constant
memory independent of the largest physical parameter or publisher bucket.
The regression guard also prevents `state_dict()` export and requires original
parameters; manual parameter offload is rejected.

### Training and BF16 probability envelope

Both modes use four real text prompts and two samples per prompt per iteration,
standard verl GRPO advantages/PPO loss, lr=1e-6, response limit 32 and microbatch 1.
Raw generated text, tokens, rewards, exit depths, seeds and state hashes are saved.
The documented first-integer proximity reward is an integration task; these
three iterations do not measure convergence or held-out arithmetic accuracy.

| Mode | Gradient norms (steps 1/2/3) | Largest raw selected log-probability difference | Observed behavior-ratio range |
|---|---|---|---|
| Full depth (`exit_threshold=1`) | 14.39316 / 6.88354 / 6.95866 | 0.340715 | [0.711262, 1.196085] |
| Adaptive exit (`exit_threshold=0`) | 12.10639 / 9.22244 / 6.74488 | 0.228540 | [0.833278, 1.256764] |

Every valid token passed the declared smoke gate: finite, absolute difference
≤0.5 and ratio in [0.5,2]. The ratio is exp(recomputed old-policy log-probability
minus raw rollout log-probability). The gate runs **before** token TIS correction,
which is capped at 2. These discrepancies do not meet a strict 1e-3 parity target.
This run does not erase the earlier larger-sample mismatch in ROUND2.md.

Rank-0 allocated GPU peak was 14.502 GiB after the first iteration and 19.208 GiB
by later iterations; the peak is cumulative and includes framework lifetimes.
The earlier full-snapshot run's first-step peak was 17.174 GiB. This is a paired
engineering observation, not a controlled throughput or statistical speedup
benchmark. The tiny export regression supplies the explicit bounded-export check.

Synchronous full-weight publication elapsed **0.451–0.464 seconds** after each
update across these six iterations. `publish_seconds` includes per-unit gathering,
bucket publication, state broadcast and barrier while generation is idle. It
measures the recipe's publication pause, not isolated engine-copy latency or an
asynchronous zero-downtime guarantee.

### Recovery

The restarted job reconstructs both workers and the rollout engine from the
step-2 checkpoint and performs step 3. Tokens, exit depths, actual seeds, texts,
rewards, rank-0 losses, gradient norm and both shard/Adam/scheduler hashes equal
the uninterrupted step exactly. The restore republishes weights at a newer
monotonic version: rollout version 4 versus uninterrupted version 3. Version
identity is intentionally excluded from the equality test.

Only same-world-size recovery with compatible configuration is accepted.
Resharding and distributed fault-tolerant retry are not established by this test.

### Publication compatibility

The local matrix covers full depth, early exit, prefix cache, async scheduling,
CUDA Graphs, graphs+async, speculation and graphs+speculation. Cache/graphs are
warmed **before** weights change. Local target parameter addresses remain stable.
Existing captured graph objects survive and replay after publication:
28 replays (graphs), 21 (graphs+async), 38 (graphs+speculation). Speculation also
captures one legitimate previously unseen verification shape after the update.

For P/D, prefill runs on GPU0 and decode on GPU1. Each eager/graphs job transfers
12,599,296 bytes in 6 actual NIXL transfers over warmup plus post-update requests;
these are KV transfers, not a measurement of weight-publication bandwidth.
Both owners finish at weight version 1 with zero used cache blocks. Graph mode
retains one capture and reports 84 decode replays versus 42 in an independently
measured identical warmup-only job. The post-publication decode counters therefore
show 42 additional replays. Updated outputs/scores match a freshly loaded local
engine in the same execution mode with the same sampler cursor.

These checks establish publication/fresh-engine equality **within each mode**,
not independent eager-versus-feature semantic parity. P/D speculation is not
supported by the pinned engine. No public engine API or production serving file
was changed by this framework integration.

## Delivery boundary

This provides the experimental Phase 2A training/publication/recovery modules
for a contributor-fork draft PR. Remaining work includes migrating to the final
shared Phase 1 contract, production Ray orchestration and its locked runtime,
BF16 numerical analysis, native packed/sequence-parallel kernels and broader
quality/performance experiments. N+2 predictor training and runtime policy are
a separate Phase 3 effort. This evidence makes no N+2 result or speedup claim.

Development and automated tests used OpenAI Codex assistance. Human line-by-line
review and independent testing remain required before an upstream verl PR.
