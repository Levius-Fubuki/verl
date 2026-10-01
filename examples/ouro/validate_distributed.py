# SPDX-License-Identifier: Apache-2.0
"""Shared-layer FSDP gradient oracle; torchrun on one or multiple real GPUs.

Single-rank execution validates the harness only. At least two GPUs are needed
to exercise sharded all-gathers, reduce-scatters and unequal local trace shapes.
"""

import argparse
import copy
import functools
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from examples.ouro.model import OuroReplayDecoderLayer, OuroTrainingModel
from examples.ouro.tests.test_replay import native_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-world-size", type=int, default=1)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if world < args.min_world_size or torch.cuda.device_count() < world:
        raise ValueError("run with the requested number of distinct visible GPUs")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        model = OuroTrainingModel(native_model(device=device))
        expected = copy.deepcopy(model)
        model.gradient_checkpointing_enable()
        # Distinct sequence counts and exit traces across ranks must not change
        # the fixed number of calls to a physical shared layer.
        ids = torch.arange(2, 8, device=device)[None].repeat(rank + 1, 1)
        depths = torch.tensor([[4, 1 + rank % 3, 3, 2, 4, 1]], device=device).repeat(rank + 1, 1)
        reference_loss = expected(ids, execution_depths=depths).logits.float().square().mean()
        reference_loss.backward()
        for parameter in expected.parameters():
            if parameter.requires_grad:
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(world)
        torch.optim.SGD(expected.parameters(), lr=0.01).step()
        wrapped = FSDP(
            model,
            device_id=device,
            use_orig_params=True,
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls={OuroReplayDecoderLayer}
            ),
        )
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
        loss = wrapped(ids, execution_depths=depths).logits.float().square().mean()
        loss.backward()
        optimizer.step()
        with FSDP.state_dict_type(
            wrapped, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
        ):
            actual = wrapped.state_dict()
        max_error = 0.0
        for name, tensor in expected.state_dict().items():
            torch.testing.assert_close(actual[name], tensor, atol=2e-6, rtol=3e-5)
            max_error = max(max_error, (actual[name] - tensor).abs().max().item())
        error = torch.tensor(max_error, device=device)
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        if rank == 0:
            report = {
                "world_size": world,
                "multi_gpu_validated": world > 1,
                "max_parameter_error_vs_averaged_gradient_oracle": error.item(),
                "shared_layer_wrapping": True,
                "activation_checkpointing": True,
            }
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
