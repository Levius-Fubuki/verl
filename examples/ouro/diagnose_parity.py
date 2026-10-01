# SPDX-License-Identifier: Apache-2.0
"""Separate precision, attention backend, and native prefill batching effects."""

import argparse
import json

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.ouro import OuroForCausalLM

from examples.ouro.model import OuroTrainingModel
from examples.ouro.tests.test_replay import engine_logits


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    ids = torch.tensor([[2, 3, 5, 7, 11, 13, 17, 19]], device="cuda")
    report = {}

    def compare(a, b):
        delta = (a.float() - b.float()).abs()
        lp_a, lp_b = a.float().log_softmax(-1), b.float().log_softmax(-1)
        targets = ids[:, 1:].T
        selected = (lp_a[:-1].gather(1, targets) - lp_b[:-1].gather(1, targets)).abs()
        return {
            "max_abs_logit": delta.max().item(),
            "mean_abs_logit": delta.mean().item(),
            "max_abs_logprob": (lp_a - lp_b).abs().max().item(),
            "max_abs_selected_logprob": selected.max().item(),
            "argmax_equal": bool((a.argmax(-1) == b.argmax(-1)).all()),
        }

    for dtype in (torch.float32, torch.bfloat16):
        native = OuroForCausalLM.from_pretrained(args.model, device="cuda", dtype=dtype)
        train = OuroTrainingModel(native).eval()
        for tag, depths in (("full", [4] * 8), ("adaptive", [4, 4, 4, 1, 3, 2, 4, 1])):
            depth_tensor = torch.tensor([depths], device="cuda")
            with sdpa_kernel(SDPBackend.MATH):
                math = train(ids, execution_depths=depth_tensor).logits[0]
            default = train(ids, execution_depths=depth_tensor).logits[0]
            for backend in ("torch", "triton"):
                expected = engine_logits(native, ids[0], depths, backend=backend)
                report[f"{dtype}/{tag}/{backend}/math"] = compare(math, expected)
                report[f"{dtype}/{tag}/{backend}/default"] = compare(default, expected)
            if tag == "full":
                cfg = native.config
                cache = KVCacheManager(
                    cfg.num_hidden_layers,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    num_blocks=4,
                    block_size=16,
                    max_loops=4,
                    device="cuda",
                    dtype=dtype,
                    backend="torch",
                )
                assert cache.allocate("prompt", 8)
                hidden = native.prelude(ids[0])
                for depth in range(4):
                    hidden, _ = native.recurrent(hidden, ["prompt"] * 8, [depth] * 8, list(range(8)), cache)
                report[f"{dtype}/full/batched_prefill/math"] = compare(math, native.coda(hidden))
                del cache, hidden
            print(json.dumps({k: v for k, v in report.items() if k.startswith(f"{dtype}/{tag}")}), flush=True)
        del native, train, math, default, expected
        torch.cuda.empty_cache()
    with open(args.output, "w") as out:
        json.dump(report, out, indent=2)


if __name__ == "__main__":
    main()
