# Ouro differentiable replay and verl worker (Phase 2A experiment)

This experiment adds the trainable model needed by ThinkFlowLab/vllm-rlt issue
[#70](https://github.com/ThinkFlowLab/vllm-rlt/issues/70). It is not a complete
production verl Ray-trainer integration. The single-GPU driver now exercises a
real verl FSDPEngine, PPO loss, native rollout, full-weight publication and resume.
Run from this fork's repository root with the sibling engine on `PYTHONPATH`.

**Acceptance boundary:** BF16 kernels differ numerically. On 256 generated tokens,
the largest selected log-probability discrepancy was 0.4344 and one probability
ratio fell outside [0.8, 1.2]. This is measured, not hidden by changing tolerances.
Optional verl token-level TIS is an experimental loss correction, not a parity
fix or convergence claim. See [ROUND2.md](ROUND2.md) and the earlier [VALIDATION.md](VALIDATION.md).

## What is implemented

- A single physical recurrent core, with standard Ouro checkpoint parameter
  names and full-depth prefill semantics.
- Differentiable SDPA and explicit per-input-token execution depths. Inactive
  tokens carry their final executed layer K/V into all deeper loop planes,
  preserving gradients through `last_exited` cache reuse.
- Dense batches with independent traces and contiguous left/right padding.
- Non-reentrant layer checkpointing; fixed loop/layer call counts.
- Local HF `AutoConfig` / `AutoModelForCausalLM` registration and save/load,
  using `trust_remote_code=False`. No Hugging Face Python code is downloaded.
- CPU engine parity, finite-difference gradient checks, exact tiny-model Adam
  save/resume, CUDA BF16/Triton parity, and a single-rank FSDP smoke driver.
- `OuroFSDPEngine` forwards mandatory exit traces from jagged TensorDict inputs.
  The adapter preserves response masks, actual seeds and weight versions.
- Packed input is re-padded into independent rows and gathered back afterwards.
  It is a correctness bridge, not a fused variable-length attention optimization.
- Physical replay layers expose a training forward for per-layer FSDP wrapping,
  with fixed invocation counts regardless of local traces or sequence count.
- Real-model two-step PPO, full-weight update/fresh-engine comparison, and exact
  next-step resume of weights, Adam and scheduler states on one GPU.

The existing gate is frozen. Gradients condition on a supplied execution trace;
this does not differentiate the discrete exit decision or train an N+2 head.
The dense implementation still computes masked rows. Do not claim training
speedups from early exit in this first correctness implementation.

## Versions and environment

| Component | Revision / validation version |
|---|---|
| verl checkout | `fbb4b3a8bf636f290c9c59fc346f756849e9c241` |
| vllm-rlt | `ecb1f8b505b7e831815b40aec3b4598619cca23a` |
| Round 2 engine prototype, 0z5a/vllm-rlt PR #4 | `bdce2876e3f78ac7efaa6cbf5783c185da2e9d19` |
| ByteDance/Ouro-1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| Model-level test environment | Python 3.12, torch 2.12.1+cu130, transformers 5.12.1 |

Both rounds reuse the rental image's torch 2.12.1. The worker was tested with
tensordict 0.10.0 and transformers 5.12.1; this is not validation of verl's full
torch-2.13 lockfile environment. The physical model classes require vllm-rlt; they
do not require installation of upstream vLLM or Megatron.

```bash
uv venv --python 3.12 --system-site-packages .venv-ouro
uv pip install --python .venv-ouro/bin/python -r examples/ouro/requirements-validation.txt
PYTHONPATH=.:../engine OMP_NUM_THREADS=4 .venv-ouro/bin/python -m pytest examples/ouro/tests/test_replay.py examples/ouro/tests/test_hf.py -q
PYTHONPATH=.:../engine OMP_NUM_THREADS=4 .venv-ouro/bin/python -m examples.ouro.validate_gpu --output tiny-fsdp.json
PYTHONPATH=.:../engine OMP_NUM_THREADS=4 .venv-ouro/bin/python -m examples.ouro.validate_gpu --model /path/to/pinned/Ouro-1.4B --output full-fsdp.json
```

PyTorch/CUDA must already be available in the selected interpreter. Dependencies
listed here cover this model slice, not the entire verl worker stack. The GPU
smoke uses synthetic token IDs to validate gradients/updates, not reward quality.

## Model API and trace alignment

```python
from examples.ouro.model import OuroTrainingModel, input_execution_depths
import torch

model = OuroTrainingModel.from_pretrained("/path/to/Ouro-1.4B", device="cuda")
model.gradient_checkpointing_enable()
depths = input_execution_depths(prompt_length=3, output_depths=[4, 1, 3], max_loops=4)
# [4, 4, 4, 1, 3, 1], aligned to prompt + response inputs.
ids = torch.tensor([[2, 3, 5, 7, 11, 13]], device="cuda")
logits = model(ids, execution_depths=torch.tensor([depths], device="cuda")).logits
```

The first output token comes from the last full-depth prompt input. Subsequent
output depths describe execution of the preceding response input. The final
input has no response target and **must be masked out of the external loss**.
Prompt targets must also be masked for a response-only RL loss. The helper
rejects non-full-depth first outputs rather than silently shifting a wrong trace.

`forward` returns unprocessed model logits. The initial worker experiment uses
temperature=1, top_p=1, top_k=-1 and the prototype's `logprobs_mode="raw"`.
Top-k/top-p and other temperatures require corresponding training-policy
processing; the experimental adapter rejects them instead of silently mixing
probability conventions. This does not restrict the shared serving engine.

For the HF load path, import `examples.ouro.hf` before `AutoConfig` loading.
The intended verl hook is `model.external_lib=examples.ouro.hf` with
`trust_remote_code=False`; this is a registration hook, not a runnable RL recipe.

## Remaining integration work

1. **Needs multiple GPUs:** sharded FSDP gradients and collectives under unequal
   local traces, cross-rank checkpoint recovery, and separate-device PD updates.
   A torchrun gradient oracle is prepared in `validate_distributed.py`; its
   world-size-1 run is only a harness check, not distributed acceptance.
2. Integrate the forthcoming upstream Phase 1 schema. This prototype uses
   `log_probs` and request-level `raw`; the proposed upstream API uses `logprobs`
   and engine-level `raw_logprobs`. Do not conflate the two contracts.
3. Register the worker/rollout in a full Ray trainer recipe, validate verl's
   locked environment, and test meaningful rewards/convergence on a dataset.
4. Continue BF16 numerical analysis and assess correction on broader samples.
   Native packed kernels, sequence parallelism, filtered sampling and N+2-head
   training remain later development; no N+2 speedup is claimed here.

## Worker experiments

Install the verl worker dependencies before running these commands. The raw
round-2 environment inventory is stored with the experiment results; the small
`requirements-validation.txt` alone does not install the worker stack. Keep the
image's matching torch/torchvision pair; unconstrained dependency resolution can
upgrade torch and break the inherited torchvision binary.

```bash
PYTHONPATH=.:../engine-training python -m pytest examples/ouro/tests -q
PYTHONPATH=.:../engine-training python -m examples.ouro.validate_worker \
  --model /path/to/Ouro-1.4B --tokens 128 --layer-wrap --early-exit \
  --checkpoint /path/to/checkpoint --output worker-resume.json
PYTHONPATH=.:../engine-training python -m examples.ouro.validate_worker \
  --model /path/to/Ouro-1.4B --tokens 128 --packed --layer-wrap --early-exit \
  --rollout-correction --output worker-corrected.json
PYTHONPATH=.:../engine-training python -m examples.ouro.validate_runtime \
  --model /path/to/Ouro-1.4B --output runtime-matrix.json
# Next machine: requires TWO real visible GPUs on one host.
PYTHONPATH=.:../engine-training torchrun --standalone --nproc_per_node=2 \
  -m examples.ouro.validate_distributed --min-world-size=2 --output distributed.json
```

## Attribution

The architecture and physical layers are reused from vllm-rlt (Apache-2.0),
which attributes ByteDance/Ouro and the Qwen/Hugging Face implementations.
The differentiable depth-replay approach was cross-checked against 0z5a's
[VIME PR #5](https://github.com/0z5a/vime/pull/5), revision
`0a88bf7c1c9a94a3a730f09e32d84476bc60aa77`. Our batched masked replay is checked
against the engine's independent chronological KV implementation. This code was
developed with AI assistance and must receive human review before submission.
