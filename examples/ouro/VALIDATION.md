# V1 model validation — 2026-10-01

Status: **model component implemented; full-model BF16 probability consistency
not accepted; Phase 2A remains incomplete.** No public PR was created.

## Environment and provenance

- One CUDA device reported as RTX 4090 with 49,140 MiB; compute capability 8.9.
- Ubuntu 22.04, Python 3.12.3, torch 2.12.1+cu130, transformers 5.12.1.
- RAM cgroup limit: 77,309,411,328 bytes. Data volume: 50 GB.
- Model revision: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
- Model safetensors SHA-256:
  `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
- Native engine and verl checkout revisions are listed in README.md.
- Weight download used the public hf-mirror endpoint because the server's direct
  Hugging Face connection timed out. No remote model code was executed.

## Passed checks

`PYTHONPATH=.:../engine OMP_NUM_THREADS=4 python -m pytest examples/ouro/tests -q`

Result: **18 passed**. Ruff check and format check passed.

Tests include full/variable depths, GQA/MHA, padding, causal isolation, checkpointed
gradient equivalence, finite differences through a shallow token's reused KV,
exact tiny-model Adam save/resume, CUDA BF16 with torch/Triton native oracles,
HF loading/save/reload, and reloading that saved checkpoint into the native engine.

A regression exposed uninitialized nonpersistent RoPE buffers in HF's meta
loading path. The initializer now rebuilds them explicitly. A second regression
ensures HF-saved architecture metadata remains readable by the native engine.

## Actual Ouro-1.4B single-rank FSDP smoke

- 1,434,652,673 physical parameters; 269 state-dict tensors.
- Existing exit gate frozen; all other parameters trainable.
- BF16 compute; FP32 master weights and AdamW moments; FP32 RoPE buffers.
- Root FSDP, world size 1 (`NO_SHARD`), non-reentrant activation checkpointing.
- One synthetic sequence of **8 tokens**, with input depths `[4,4,4,1,3,2,4,1]`.
- Two optimizer steps produced finite gradients and changed weights.
- Synthetic cross-entropy: `20.821184` then `17.135157`.
- Peak allocated GPU memory: **24.0825 GiB**; peak reserved: **32.5313 GiB**.

These memory figures do not predict long-sequence or concurrent rollout capacity.
The loss decrease is a smoke result, not evidence of RL learning, quality, or
convergence. Full-model optimizer resume and multi-rank FSDP were not tested.

## Probability mismatch: retained as an explicit blocker

The native reference processes the supplied token sequence chronologically and
finalizes each token's LAST_EXITED KV. The training model processes a dense
sequence at each recurrence. The initial BF16 SDPA/Triton comparison reports:

| Trace | Max absolute logit error | Max absolute all-vocabulary logprob error | Greedy equality |
|---|---:|---:|---|
| Full 4 loops | 0.625 | 0.657654 | No |
| Variable depths | 0.380859 | 0.414968 | Yes |

The conservative diagnostic gate is max logprob error <= 0.001 plus equal
argmax; **it failed**. This threshold is not a maintainer-approved RL tolerance.

FP32 with TF32 disabled, default SDPA versus the Triton native reference, reduces
the maximum all-vocabulary logprob differences to **9.73e-5 (full)** and
**7.06e-5 (variable)**, with equal argmax on all eight inputs. FP32 math SDPA
reduces them to roughly 2.29e-5. For the synthetic next-token labels, default
FP32 selected-score errors are <= 2.29e-5.

Further controls compared torch/Triton attention, math/default SDPA, a batched
native prefill, and explicit FP32 attention accumulation over BF16 Q/K/V. The
BF16 discrepancy persists: merely changing the softmax implementation does not
resolve it. The evidence supports a precision/execution-path investigation;
it does not yet isolate one kernel or prove a root cause.

## Next implementation slice

1. Capture per-layer/per-loop BF16 states on matched prompt/decode shapes to
   locate the first divergence; preserve the failed case as the acceptance case.
2. Establish train/rollout probability tolerance and any required correction
   with the Phase 1 contract, using actual rollout tokens and meaningful lengths.
3. Wire the local HF model into verl's locked worker environment and forward
   execution depths through batching. Do not silently fall back to full depth.
4. Connect rollout output/publication APIs and run multi-iteration RL, full
   optimizer resume, then true multi-GPU FSDP and existing inference-feature tests.

Raw logs and JSON were copied to the parent workspace's
`results/2026-10-01-v1/results/`, including environment versions, checkpoint hash,
the initial failing regressions, final tests, FSDP smoke and parity diagnostics.
