# Copyright 2026 Individual Contributor: Levius-Fubuki
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
"""Model-level native parity and FSDP/Adam smoke, not an RL training recipe."""

import argparse
import gc
import json
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

from examples.ouro.model import OuroTrainingModel
from examples.ouro.tests.test_replay import engine_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Pinned local Ouro checkpoint; omit for tiny model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if args.model:
        native = OuroForCausalLM.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
    else:
        native = OuroForCausalLM(
            OuroConfig(
                vocab_size=37,
                hidden_size=32,
                intermediate_size=48,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
            )
        ).to("cuda", torch.bfloat16)
    ids = torch.tensor([[2, 3, 5, 7, 11, 13, 17, 19]], device="cuda")
    traces = {"full": [4] * 8, "adaptive": [4, 4, 4, 1, 3, 2, 4, 1]}
    references = {name: engine_logits(native, ids[0], depths, backend="triton") for name, depths in traces.items()}
    model = OuroTrainingModel(native)
    model.gradient_checkpointing_enable()
    report = {
        "model": args.model or "tiny",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "visible_gpus": torch.cuda.device_count(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "physical_tensors": len(model.state_dict()),
        "world_size": 1,
        "scope": "FSDP single-rank NO_SHARD, not multi-GPU sharding or verl RL",
        "native_backend": "triton",
        "parity": {},
    }
    with torch.no_grad():
        for name, values in traces.items():
            actual = model(ids, execution_depths=torch.tensor([values], device="cuda")).logits[0]
            expected = references[name]
            delta = (actual.float() - expected.float()).abs()
            lp_delta = (actual.float().log_softmax(-1) - expected.float().log_softmax(-1)).abs()
            report["parity"][name] = {
                "max_abs_logit": delta.max().item(),
                "mean_abs_logit": delta.mean().item(),
                "max_abs_logprob": lp_delta.max().item(),
                "argmax_equal": bool((actual.argmax(-1) == expected.argmax(-1)).all()),
            }
            # Independent BF16 kernels are not expected to be bitwise identical.
            # Record the entire error envelope; do not label this policy-ratio acceptance.
            assert torch.isfinite(actual).all()
    report["bf16_parity_gate"] = {
        "max_abs_logprob_limit": 1e-3,
        "require_same_argmax": True,
        "passed": all(v["max_abs_logprob"] <= 1e-3 and v["argmax_equal"] for v in report["parity"].values()),
        "note": "Conservative model-slice gate, not a complete RL acceptance criterion.",
    }
    del native, references, actual, expected, delta, lp_delta
    gc.collect()
    torch.cuda.empty_cache()
    # FP32 master weights/Adam state, BF16 compute. The pretrained BF16 values are
    # preserved exactly when promoted; this is full-parameter, not LoRA training.
    model.float()
    depths = torch.tensor([traces["adaptive"]], device="cuda")
    with tempfile.TemporaryDirectory() as temp:
        dist.init_process_group("nccl", init_method="file://" + temp + "/init", rank=0, world_size=1)
        try:
            wrapped = FSDP(
                model,
                device_id=0,
                use_orig_params=True,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                ),
            )
            optimizer = torch.optim.AdamW(
                (p for p in wrapped.parameters() if p.requires_grad),
                lr=1e-5,
                foreach=False,
            )
            losses = []
            watched = model.model.layers[0].self_attn.q_proj.weight
            before = watched.detach().float().clone()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                logits = wrapped(ids, execution_depths=depths).logits
                # A synthetic teacher-forcing loss validates the model/optimizer,
                # not reward quality, convergence, or a complete policy update.
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    ids[:, 1:].reshape(-1),
                )
                assert torch.isfinite(loss)
                loss.backward()
                assert all(
                    p.grad is not None and torch.isfinite(p.grad).all() for p in wrapped.parameters() if p.requires_grad
                )
                optimizer.step()
                losses.append(loss.item())
            report["losses"] = losses
            report["fp32_master_and_adam"] = bool(
                all(
                    p.dtype == torch.float32 and optimizer.state[p]["exp_avg"].dtype == torch.float32
                    for p in wrapped.parameters()
                    if p.requires_grad
                )
            )
            report["weight_changed"] = not torch.equal(before, watched.detach().float())
            assert report["fp32_master_and_adam"] and report["weight_changed"]
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            report["seconds"] = time.monotonic() - started
        finally:
            dist.destroy_process_group()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
