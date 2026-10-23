# Ouro BF16 verl/FSDP integration (Phase 2A experiment)

This contributor-fork implementation addresses the Ouro training integration in
[vllm-rlt issue #70](https://github.com/ThinkFlowLab/vllm-rlt/issues/70).
`train.py` runs synchronous generation → GRPO advantages → verl PPO/FSDP update →
full-weight publication on one or two GPUs. It uses real verl `BaseRollout`,
`DataProto`, `FSDPEngine`, loss and checkpoint APIs through a direct torchrun
recipe. It does not implement the production Ray AgentLoop/RolloutReplica path.

Two-GPU Ouro-1.4B acceptance covers full-depth and adaptive-exit training,
checkpoint recovery, per-unit sharded export, and separate-device NIXL P/D
publication. See [ROUND3.md](ROUND3.md) for measured results and raw evidence;
[ROUND2.md](ROUND2.md) and [VALIDATION.md](VALIDATION.md) retain earlier evidence.

## Current interface migration (2026-10-23)

The adapter now defaults to the shared rollout-output contract: `logprobs`,
engine-level `raw_logprobs`, and public request/step/abort generation for raw
StageClient/PDEngine endpoints. See [MIGRATION.md](MIGRATION.md) for CPU evidence,
exact pins and commands. The existing October 8 GPU results below are historical.

**Current #83 has no public weight publication or committed policy version.**
Versioned `OuroRollout` and `train.py` reject that missing capability before
starting a worker or initializing CUDA/FSDP. `validate_current.py` separately
qualifies actual generation, seeds, trace replay, stop/abort and speculative
selected scores; it does not claim a training update or real P/D validation.
The shared publication interface must be integrated before a new full training
or CUDA/P/D acceptance run. No upstream PRs were combined locally.

`contract="prototype"` (CLI `--engine-contract prototype`) is only for deliberate
reproduction against the pinned old contract. It is never auto-selected as a
fallback. Historical GPU drivers verify all engine Python sources against the
original hash manifest before allocating CUDA; they reject newer checkouts.

## Modules

| File | Responsibility |
|---|---|
| `model.py`, `hf.py` | Checkpoint-compatible differentiable recurrent replay; local HF registration |
| `worker.py` | Trace-aware `OuroFSDPEngine`, response-only PPO batch conversion, per-unit FSDP1 export |
| `rollout.py` | In-process verl `BaseRollout`, actual request seeds/depths/log probabilities/versions |
| `publication.py` | Physical named-tensor buckets, monotonic transactions, retry/restart counters |
| `recipe_utils.py`, `train.py` | Deterministic arithmetic task, rewards, probability gate, synchronous RL and resume |
| `validate_distributed.py`, `validate_adapter.py` | Two-rank gradient oracle and serving-feature publication acceptance |

The training model retains one physical 24-layer Ouro core reused across four
loops. Inactive inputs carry their last executed layer K/V into deeper loop
planes, including gradients through this reuse. Layer checkpointing is
non-reentrant, and every rank executes the same layer/loop call count despite
unequal local traces. The existing exit gate is frozen: replay conditions on the
supplied discrete trace and does not train an N+2 predictor.

The engine registers under model type `ouro` without replacing verl's global
`language_model` backend. Packed inputs are padded into independent rows and
flattened back after execution; this is a correctness bridge, not fused varlen
attention. Sequence parallelism and fused loss kernels are rejected.

FSDP1 export requires `use_orig_params=True`. Every rank gathers physical units
in the same order; only rank 0 publishes. Export yields owned BF16 tensors from
one unit at a time without a full-model `state_dict()` snapshot. The publisher
uses a soft bucket cap: a single oversized parameter is sent alone, and the
producer may hold the active gathered unit and one lookahead tensor. LoRA and
manual parameter offload are unsupported; FSDP2 delegates to verl and is unverified.

## Historical training contract and environment (2026-10-08)

| Component | Validated revision/version |
|---|---|
| verl baseline | `fbb4b3a8bf636f290c9c59fc346f756849e9c241` |
| Usable engine prototype, [0z5a/vllm-rlt PR #4](https://github.com/0z5a/vllm-rlt/pull/4) | `bdce2876e3f78ac7efaa6cbf5783c185da2e9d19` |
| ByteDance/Ouro-1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| Model safetensors SHA256 | `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af` |
| Image runtime | Python 3.12.3, torch 2.12.1+cu130, triton 3.7.1 |
| Adapter dependencies | transformers 5.12.1, tensordict 0.10.0, NIXL and nixl-cu13 1.4.1 |
| Acceptance hardware | 2 × RTX 4090, 49,140 MiB each; 24 CPU cores, 144 GiB RAM; PCIe PHB |

The validated engine exposes `RequestOutput.log_probs`, request-level
`logprobs_mode="raw"`, `start_weight_update(version=...)`, `update_weights(...)`,
parameterless `finish_weight_update()` and `get_weight_version()`. Do not run this
adapter against a different proposed contract without adapting and retesting it.
As checked on 2026-10-08, [#83](https://github.com/ThinkFlowLab/vllm-rlt/pull/83)
uses `logprobs` and engine-level `raw_logprobs`, while the shared weight-update
work is still a design. This PR develops the framework integration against a
working prototype; it does not duplicate the engine API PRs.

A successful publication starts at version `global_steps + 1`. Retry/restore
uses a strictly newer version when the previous attempt or backend already
observed that number. Update failure keeps admission paused; a partial P/D
commit is not rolled back. Recover/retry before resuming generation. `release()`
pauses admission only; it does not free model/cache memory or implement sleep/wake.

The worker dependency snapshot is [requirements-worker.lock](requirements-worker.lock).
It inherits the image's matching torch/CUDA/triton binaries and was installed in
a fresh venv before the final tests and full-depth GPU run. This is an
image-dependent reproducer, not validation of verl's torch-2.13 lockfile.

```bash
# From this fork's repository root; sibling engine checkout is intentional.
git clone https://github.com/0z5a/vllm-rlt.git ../engine-training
git -C ../engine-training checkout bdce2876e3f78ac7efaa6cbf5783c185da2e9d19
uv venv --python /path/to/image/python --system-site-packages .venv-ouro
uv pip install --python .venv-ouro/bin/python --no-deps \
  -r examples/ouro/requirements-worker.lock
export PYTHONPATH=.:../engine-training
export OMP_NUM_THREADS=4
PY="$PWD/.venv-ouro/bin/python"
$PY -c 'import torch, transformers, tensordict, nixl; print(torch.__version__)'
```

Use the pinned model's `config.json`, `model.safetensors` and tokenizer files.
No model-repository Python is loaded (`trust_remote_code=False`). Revision
labels in the training JSON describe the required pins; independently verify
checkout and model hashes, as in the checked-in source/model manifests.
Allow roughly 55 GB of persistent storage for the environment, model and one
full model/Adam checkpoint; retain more space if keeping multiple checkpoints.
The observed run fits this 48 GB-per-card machine without NVLink or GPU peer access.

## Reproduce historical training acceptance

```bash
mkdir -p results
$PY -m pytest examples/ouro/tests -q
$PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m examples.ouro.validate_distributed --min-world-size=2 --output results/distributed.json
$PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m pytest examples/ouro/tests/test_export.py -q

# Native engine regression against the same pinned prototype.
(cd ../engine-training && CONTRACT_DEVICE=cuda OMP_NUM_THREADS=4 \
  "$PY" -m pytest tests/test_training_contract.py -q)
(cd ../engine-training && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 \
  "$PY" -m pytest tests -q)

# Full-depth BF16 baseline: three fresh RL iterations.
$PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m examples.ouro.train --engine-contract prototype --model /path/to/Ouro-1.4B --steps 3 \
  --response-length 32 --prompts-per-step 4 --samples-per-prompt 2 \
  --micro-batch 1 --exit-threshold 1 --output results/full-depth.json

# Adaptive-exit run: save step 2, then continue to step 3.
$PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m examples.ouro.train --engine-contract prototype --model /path/to/Ouro-1.4B --steps 3 \
  --response-length 32 --prompts-per-step 4 --samples-per-prompt 2 \
  --micro-batch 1 --exit-threshold 0 --checkpoint-dir checkpoint \
  --checkpoint-at 2 --output results/uninterrupted.json

# New processes restore both shards, Adam, scheduler, RNG and sampler cursor.
$PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m examples.ouro.train --engine-contract prototype --model /path/to/Ouro-1.4B --steps 3 \
  --response-length 32 --prompts-per-step 4 --samples-per-prompt 2 \
  --micro-batch 1 --exit-threshold 0 --resume checkpoint --output results/resumed.json

$PY - <<'PY'
import json
left = json.load(open("results/uninterrupted.json"))["iterations"][-1]
right = json.load(open("results/resumed.json"))["iterations"][0]
keys = ("token_ids", "exit_depths", "sampling_seeds", "texts", "rewards",
        "losses_rank0", "grad_norm", "shard_state_sha256")
for key in keys:
    assert left[key] == right[key], key
print("Exact next-iteration resume:", keys)
# Policy version intentionally advances on republishing restored weights.
PY

# Run separately from training: these jobs own the GPUs.
$PY -m examples.ouro.validate_adapter --model /path/to/Ouro-1.4B \
  --output results/adapter-matrix.json
$PY -m examples.ouro.validate_adapter --model /path/to/Ouro-1.4B \
  --pd --output results/adapter-pd.json
```

Each iteration uses four scheduled addition prompts from the 64 pairs
`a,b ∈ {1,...,8}`, with two independently seeded samples per prompt. Reward is
`1 / (1 + abs(first_integer - answer))`, or zero when no integer is produced.
This intentionally small integration task supports nonzero GRPO advantages;
it does not establish arithmetic quality or convergence. The recipe rejects
all-zero advantages, nonfinite/zero gradients and non-FP32 master/Adam state.

The PPO fields are response-only. Model logits predict the next input token;
the first response target comes from the last full-depth prompt input. The
final input and prompt targets are masked out. `input_execution_depths()`
performs this shift and rejects inconsistent first-output depth. Actual response
lengths, including short EOS microbatches, determine padding and loss alignment.

Raw selected rollout probabilities and recomputed old-policy probabilities
remain distinct. Training currently requires temperature=1, top_p=1, top_k=-1;
filtered/greedy behavior requires a matching policy processing implementation.
Before correction, valid response tokens must be finite and satisfy max absolute
log-probability difference ≤0.5 and likelihood ratio in [0.5,2]. This is an
explicit BF16 smoke envelope, not 1e-3 parity. The recipe then applies verl token
TIS capped at 2 using recomputed old-policy probabilities for PPO. Changing the
bounds changes acceptance; raw discrepancies are recorded in every iteration.

## Acceptance scope and follow-up

The feature matrix compares a warmed, updated engine with a fresh engine in the
same mode, with identical sampler state. It covers full depth, early exit,
prefix cache, async scheduling, CUDA Graphs and speculation, plus supported
P/D eager/graphs. It verifies actual post-publication replay of existing graph
captures and real NIXL transfers, not just enabled flags. It does not establish
independent eager-versus-feature semantic parity. P/D speculation is unsupported
by the pinned engine and is not included.

The checkpoint test restores the same world size and compatible configuration.
World-size-changing resharding, fault-tolerant distributed retry, production Ray
trainer orchestration, the upstream locked runtime, native packed attention and
sequence parallelism are later work. Dense masked replay still computes inactive
rows; no early-exit training speedup, held-out quality gain or N+2 speedup is claimed.

Phase 3 is a separate follow-up: train/evaluate a loop-specific N+2 exit predictor
and integrate its runtime policy. This PR supplies replay, training/publication
and reproducible baseline modules for that work. Migration to the shared Phase 1
API follows its final implementation.

## Attribution and review

The physical layers reuse vllm-rlt (Apache-2.0), which attributes ByteDance/Ouro
and Qwen/Hugging Face. The differentiable replay was cross-checked against
[0z5a/vime PR #5](https://github.com/0z5a/vime/pull/5), revision
`0a88bf7c1c9a94a3a730f09e32d84476bc60aa77`. Batched replay is tested against the
engine's independent chronological KV implementation. Development and experiment
execution used OpenAI Codex assistance. The contributor-fork draft is for human
review; the submitter must review every changed line and independently run the
relevant tests before any upstream verl contribution under `AGENTS.md`.
