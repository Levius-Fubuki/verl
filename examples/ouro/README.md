# Ouro differentiable replay (Phase 2A, V1 model component)

This experiment adds the trainable model needed by ThinkFlowLab/vllm-rlt issue
[#70](https://github.com/ThinkFlowLab/vllm-rlt/issues/70). It is not a complete
verl RL rollout/update integration. Run from this fork's repository root with
the sibling engine on `PYTHONPATH`.

**Current blocker:** the full Ouro BF16 replay does not yet pass the probability
consistency gate. Small-model tests and full-model optimizer updates pass, but
they must not be presented as full-model BF16 probability parity. See the saved
validation report. FP32 diagnostics distinguish this from a gross recurrence/KV
semantic error; the BF16 numerical path still needs work before RL acceptance.

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

The existing gate is frozen. Gradients condition on a supplied execution trace;
this does not differentiate the discrete exit decision or train an N+2 head.
The dense implementation still computes masked rows. Do not claim training
speedups from early exit in this first correctness implementation.

## Versions and environment

| Component | Revision / validation version |
|---|---|
| verl checkout | `fbb4b3a8bf636f290c9c59fc346f756849e9c241` |
| vllm-rlt | `ecb1f8b505b7e831815b40aec3b4598619cca23a` |
| ByteDance/Ouro-1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| Model-level test environment | Python 3.12, torch 2.12.1+cu130, transformers 5.12.1 |

The model-level environment reuses the rental image's torch. The full verl
worker environment pins torch 2.13, and has not been validated by these tests.
Keep those claims separate. The physical model classes require vllm-rlt; they
do not require installation of upstream vLLM or Megatron.

```bash
uv venv --python 3.12 --system-site-packages .venv-ouro
uv pip install --python .venv-ouro/bin/python -r examples/ouro/requirements-validation.txt
PYTHONPATH=.:../engine OMP_NUM_THREADS=4 .venv-ouro/bin/python -m pytest examples/ouro/tests -q
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

`forward` returns unprocessed model logits. The verl actor currently divides
logits by temperature; for a first unfiltered sampling integration, request
engine `processed_logprobs` at the same positive temperature. Top-k/top-p,
repetition processors, and depth-dependent temperature require corresponding
training-policy handling and must not be treated as equivalent automatically.

For the HF load path, import `examples.ouro.hf` before `AutoConfig` loading.
The intended verl hook is `model.external_lib=examples.ouro.hf` with
`trust_remote_code=False`; this is a registration hook, not a runnable RL recipe.

## Remaining integration work

1. Thread `execution_depths` through the verl worker's batch preparation, with
   padding/response masks and explicit failure if an adaptive trajectory loses
   its trace. Native packed/remove-padding and sequence-parallel inputs need a
   separate implementation; this model API currently accepts dense sequences.
2. Validate the full worker under verl's locked environment. Root FSDP is the
   smoke configuration. Native decoder-layer auto-wrapping is unsafe because
   this training forward calls their submodules rather than serving forwards.
   Keep RoPE buffers FP32 under mixed precision.
3. Connect the Phase 1 shared rollout output API and full-weight publication,
   including committed model versions and lifecycle handling.
4. Verify multi-iteration RL updates, full-model optimizer/checkpoint recovery,
   and true multi-GPU FSDP. Single-rank FSDP uses `NO_SHARD` and proves none of
   the multi-rank collective behavior.
5. Validate existing early-exit, async, speculative, CUDA Graphs, and PD paths
   through the adapter. Model-level trace replay is not their integration proof.

## Attribution

The architecture and physical layers are reused from vllm-rlt (Apache-2.0),
which attributes ByteDance/Ouro and the Qwen/Hugging Face implementations.
The differentiable depth-replay approach was cross-checked against 0z5a's
[VIME PR #5](https://github.com/0z5a/vime/pull/5), revision
`0a88bf7c1c9a94a3a730f09e32d84476bc60aa77`. Our batched masked replay is checked
against the engine's independent chronological KV implementation. This code was
developed with AI assistance and must receive human review before submission.
