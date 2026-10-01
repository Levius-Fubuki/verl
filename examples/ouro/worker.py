# SPDX-License-Identifier: Apache-2.0
"""Ouro trace-aware verl engine and pinned training-contract rollout adapter.

The prototype uses RequestOutput.log_probs and raw/processed mode spellings.
Keep this boundary explicit until the upstream Phase 1 contract is merged.
"""

import math

import torch
from tensordict import TensorDict

from examples.ouro.model import input_execution_depths
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def rollout_to_tensordict(outputs, *, expected_version, max_loops, packed=False, device="cpu"):
    if not outputs:
        raise ValueError("rollout batch must be nonempty")
    rows = {key: [] for key in ("input_ids", "position_ids", "execution_depths", "response_mask", "rollout_log_probs")}
    for output in outputs:
        if not output.finished or output.weight_version != expected_version:
            raise ValueError("rollout must be finished and match the requested weight version")
        params = output.sampling_params
        # Raw probabilities equal the behavior policy only for this sampler.
        # Refuse to silently train on greedy/top-p samples as on-policy PPO.
        if params is None or params.temperature != 1 or params.top_p != 1 or params.top_k != -1:
            raise ValueError("this on-policy adapter requires temperature=1, top_p=1, top_k=-1")
        if params.logprobs_mode != "raw" or params.seed is None:
            raise ValueError("raw logprobs and the actual rollout seed are required")
        prompt, response = output.prompt_token_ids, output.token_ids
        if not response or output.log_probs is None:
            raise ValueError("nonempty responses with selected-token log_probs are required")
        if len(response) != len(output.exit_depths) or len(response) != len(output.log_probs):
            raise ValueError("response, exit trace, and log_probs lengths must match")
        if not all(math.isfinite(value) and value <= 0 for value in output.log_probs):
            raise ValueError("rollout log_probs must be finite and nonpositive")
        depths = input_execution_depths(len(prompt), output.exit_depths, max_loops=max_loops)
        ids = prompt + response
        rows["input_ids"].append(torch.tensor(ids, dtype=torch.long, device=device))
        rows["position_ids"].append(torch.arange(len(ids), device=device))
        rows["execution_depths"].append(torch.tensor(depths, device=device))
        # verl output log_probs are aligned with the INPUT predicting each target.
        rows["response_mask"].append(
            torch.tensor([False] * (len(prompt) - 1) + [True] * len(response) + [False], device=device)
        )
        rows["rollout_log_probs"].append(
            torch.tensor([0.0] * (len(prompt) - 1) + output.log_probs + [0.0], dtype=torch.float32, device=device)
        )
    offsets = torch.tensor([0] + [len(row) for row in rows["input_ids"]], device=device).cumsum(0)
    batch = TensorDict(
        {key: torch.nested.nested_tensor_from_jagged(torch.cat(values), offsets) for key, values in rows.items()},
        batch_size=[len(outputs)],
    )
    batch["temperature"] = torch.ones(len(outputs), device=device)
    batch["weight_version"] = torch.tensor([output.weight_version for output in outputs], device=device)
    batch["sampling_seed"] = torch.tensor([output.sampling_params.seed for output in outputs], device=device)
    tu.assign_non_tensor_data(batch, "use_remove_padding", packed)
    return batch


class OuroFSDPEngine(FSDPEngineWithLMHead):
    """Use verl's loading, FSDP, optimizer and loss path, preserving exit traces."""

    def prepare_model_inputs(self, micro_batch):
        depths = micro_batch.get("execution_depths")
        ids = micro_batch["input_ids"]
        if depths is None:
            raise ValueError("Ouro rollout replay requires execution_depths")
        if not depths.is_nested or not torch.equal(depths.offsets(), ids.offsets()):
            raise ValueError("execution_depths offsets must match input_ids")
        if self.use_ulysses_sp:
            raise NotImplementedError("Ouro sequence-parallel attention requires a distributed implementation")
        if tu.get_non_tensor_data(micro_batch, "use_fused_kernels", False):
            raise NotImplementedError("Ouro fused loss kernels are not implemented")
        inputs, output_args = super().prepare_model_inputs(micro_batch)
        if tu.get_non_tensor_data(micro_batch, "use_remove_padding", True):
            values = depths.values()
            pad = output_args["pad_size"]
            if pad:
                values = torch.nn.functional.pad(values, (0, pad), value=1)
            inputs["execution_depths"] = values.unsqueeze(0)
        else:
            inputs["execution_depths"] = torch.nested.to_padded_tensor(
                depths, 1, output_size=tuple(inputs["input_ids"].shape)
            )
        return inputs, output_args
