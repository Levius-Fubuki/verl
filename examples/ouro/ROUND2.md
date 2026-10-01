# Single-GPU integration evidence — 2026-10-01

Status: experimental single-rank integration passed; Phase 2A is not complete.
No public PR/comment was posted. No N+2 predictor was trained.

## Provenance

- Model: ByteDance/Ouro-1.4B, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
- Engine: **0z5a/vllm-rlt PR #4 prototype**, commit
  `bdce2876e3f78ac7efaa6cbf5783c185da2e9d19`; not upstream main.
- verl baseline: `fbb4b3a8bf636f290c9c59fc346f756849e9c241`.
- One device reported as RTX 4090, 49,140 MiB; 72 GiB RAM cgroup limit.
- Python 3.12.3, torch 2.12.1+cu130, transformers 5.12.1, tensordict 0.10.0.
- The uv environment inherits the image's torch. An initial unconstrained
  dependency install shadowed it with torch 2.14.1 and broke torchvision; that
  shadowing torch/triton was removed before all worker results below. Recorded
  imports and package metadata confirm the original torch 2.12.1 was used.
- `uv pip check` does not count this inherited torch and reports four missing
  torch dependencies. Thus this is not a standalone clean lockfile validation.

## Results

| Experiment | Result | Peak allocated / reserved GiB |
|---|---|---|
| Native oracle, HF, trace/mask, packed isolation/gradient, CUDA tests | 23 passed | — |
| Prototype engine training-contract tests, CUDA | 6 passed | — |
| Real model, root FSDP, 2×16 response tokens, two PPO steps | Finite gradients, changed weights, versioned updates match fresh engine | 35.16 / 46.45 |
| Layer FSDP, early exit, 2×32 response tokens | Two steps and fresh-engine equivalence pass | 27.13 / 38.97 |
| Layer FSDP, early exit, 2×128 response tokens, save/resume | Exact next loss and SHA-256 of all weights, Adam state and scheduler state | 35.15 / 46.28 |
| Re-padded packed bridge, 2×128 tokens, optional verl token TIS | Two steps, finite gradients, versioned updates match fresh engine | 27.13 / 38.73 |
| Torchrun gradient oracle, world size 1 | Parameter error 0 against the unwrapped reference | — |

These runs differ in checkpoint/fresh-model lifetime and sequence shape. Their
memory figures are capacity observations, not a controlled performance comparison.
All FSDP runs used **NO_SHARD** because world size was one. No multi-GPU
communication, optimizer sharding, distributed recovery or PD result is claimed.

The resume comparison's next-step loss was `-0.02887723036110401`; combined state
SHA-256 was `99aaa50e4ea65e1187d923e586dfe4ede3bfa4891c20175f79021151003eb69c`.

## Serving paths

Real-model native generation, trace replay and full-weight update were exercised
for full depth, early exit, prefix cache, async delayed exit, CUDA Graphs,
graphs+async, speculative decoding, and graphs+speculative decoding. All eight
matched a fresh engine loaded with the updated weights, including selected-token
log probabilities and exit depths. Stop-token alignment passed. Graph modes
performed 56, 42 and 100 graph replays respectively; they were not merely enabled.
The prototype's CUDA tests additionally cover abort and failed-update recovery.

## Numerical boundary

The first divergent stage with identical BF16 batched prefill is attention:
first-layer Q/K/V projections match exactly; attention output projection differs.
The error grows across the four shared-core loops. This localizes the first
observed divergence but does not prove that every later error has one cause.

- 32 sampled tokens: maximum selected log-probability difference 0.1593;
  probability ratios 0.8527–1.0954.
- 256 sampled early-exit tokens: maximum difference 0.4344;
  ratios 0.6477–1.1642; 1/256 outside [0.8, 1.2].
- Optional `--rollout-correction` uses verl's detached token TIS with cap 2.
  It retains both behavior and recomputed probabilities. This biased,
  low-variance correction is an experiment, not a numerical parity fix.
- Earlier FP32 controls had maximum all-vocabulary log-probability errors below
  1e-4 against native chronological execution. The conservative BF16 1e-3 gate
  remains unmet; it is not a maintainer-defined acceptance threshold.

Prompts were short synthetic token-ID sequences and advantages alternated ±1.
The results verify software plumbing, not task reward, convergence, long-context
quality, speedup, or N+2 training.

## Next acceptance gates

1. Two real GPUs: run `validate_distributed.py --min-world-size=2` to test
   shared-layer FSDP against the averaged-gradient oracle with unequal local
   sequence counts and exit depths. Then repeat with real model/Adam/checkpoints.
2. Two separate device roles: verify PD rollout and weight publication with no
   mixed versions, stale prefix cache or missing token likelihoods.
3. Software work remains: upstream Phase 1 contract migration, a full Ray trainer
   recipe and reward dataset, broader numerical validation, and later N+2 work.
   These are not all hardware blockers; single-GPU experiments above do not
   constitute completion of Phase 2A.

Raw JSON/logs are archived in the workspace's
`results/2026-10-01-round2/results-round2/`. Full checkpoints remain on the
server's persistent data disk at `/root/autodl-tmp/ouro-v1/checkpoint-round2`.
