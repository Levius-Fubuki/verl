# SPDX-License-Identifier: Apache-2.0
"""Locate the first native/replay divergence with identical batched prefill."""

import argparse
import json

import torch
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.ouro import OuroForCausalLM

from examples.ouro.model import OuroTrainingModel


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    native = OuroForCausalLM.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
    replay = OuroTrainingModel(native).eval()
    ids = torch.tensor([[2, 3, 5, 7, 11, 13, 17, 19]], device="cuda")
    snapshots = {}
    phase = "native"
    counts = {}
    rows = []

    def hook(name):
        def capture(module, inputs, output):
            count = counts.get(name, 0)
            counts[name] = count + 1
            key = f"{count}/{name}"
            value = output.reshape(-1, output.shape[-1]).float()
            if phase == "native":
                snapshots[key] = value.cpu()
            else:
                expected = snapshots[key].to(value.device)
                delta = (value - expected).abs()
                rows.append({"stage": key, "max_abs": delta.max().item(), "mean_abs": delta.mean().item()})

        return capture

    handles = [
        module.register_forward_hook(hook(name))
        for name, module in native.named_modules()
        if name.startswith("model.layers.0.") and not list(module.children())
    ]
    cfg = native.config
    cache = KVCacheManager(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        num_blocks=4,
        block_size=16,
        max_loops=4,
        device="cuda",
        dtype=torch.bfloat16,
        backend="torch",
    )
    assert cache.allocate("prompt", 8)
    hidden = native.prelude(ids[0])
    for depth in range(4):
        hidden, _ = native.recurrent(hidden, ["prompt"] * 8, [depth] * 8, list(range(8)), cache)
    phase = "replay"
    counts.clear()
    replay(ids)
    for handle in handles:
        handle.remove()
    with open(args.output, "w") as out:
        json.dump(rows, out, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
