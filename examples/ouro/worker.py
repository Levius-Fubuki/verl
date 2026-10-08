# SPDX-License-Identifier: Apache-2.0
"""Ouro trace-aware verl engine and pinned training-contract rollout adapter.

The prototype uses RequestOutput.log_probs and raw/processed mode spellings.
Keep this boundary explicit until the upstream Phase 1 contract is merged.
"""

import math

import torch
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from examples.ouro.model import input_execution_depths
from verl.utils import tensordict_utils as tu
from verl.utils.fsdp_utils import fsdp_version
from verl.workers.engine.base import EngineRegistry
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


@EngineRegistry.register(model_type="ouro", backend=["fsdp", "fsdp2"], device="cuda")
class OuroFSDPEngine(FSDPEngineWithLMHead):
    """Use verl's loading, FSDP, optimizer and loss path, preserving exit traces."""

    def get_per_tensor_param(self, layered_summon=False, base_sync_done=False, **kwargs):
        """Stream physical Ouro weights without a full-model FSDP1 snapshot.

        All ranks consume the units in the same module order. Each yielded
        tensor owns its BF16 storage, so a publisher bucket can retain it after
        the unit's temporary unsharded storage has been released.
        """
        if hasattr(getattr(self.module, "_fsdp_wrapped_module", self.module), "peft_config"):
            raise NotImplementedError("Ouro weight export supports full weights only, without LoRA")
        if fsdp_version(self.module) != 1:
            return super().get_per_tensor_param(layered_summon=layered_summon, base_sync_done=base_sync_done, **kwargs)
        if self._is_offload_param:
            raise NotImplementedError("Ouro FSDP1 weight export does not support manual parameter offload")
        units = list(self._export_fsdp_units(self.module))
        if any(not unit._use_orig_params for _, unit in units):
            raise ValueError("Ouro FSDP1 weight export requires use_orig_params=True on every unit")
        return self._export_fsdp_weights(units), None

    @staticmethod
    def _export_fsdp_units(module, prefix=""):
        if isinstance(module, FSDP):
            yield prefix, module
            module = module._fsdp_wrapped_module
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from OuroFSDPEngine._export_fsdp_units(child, child_prefix)

    @staticmethod
    def _export_unit_parameters(module, prefix):
        # Nested FSDP units own their parameters and must be gathered separately.
        if isinstance(module, FSDP):
            return
        for name, param in module.named_parameters(recurse=False):
            if name != "_flat_param":
                yield f"{prefix}.{name}" if prefix else name, param
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from OuroFSDPEngine._export_unit_parameters(child, child_prefix)

    @staticmethod
    def _export_fsdp_weights(units):
        seen = set()
        for prefix, unit in units:
            with FSDP.summon_full_params(unit, recurse=False, writeback=False, rank0_only=False):
                for name, param in OuroFSDPEngine._export_unit_parameters(unit._fsdp_wrapped_module, prefix):
                    if id(param) in seen:
                        continue
                    seen.add(id(param))
                    yield name, param.detach().to(dtype=torch.bfloat16, copy=True)

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


def training_batch_from_rollout(
    rollout,
    *,
    advantages,
    expected_version,
    world_size,
    global_batch_size,
    global_token_count,
    device="cpu",
    packed=False,
):
    """Convert native output to verl's RESPONSE-aligned standard PPO batch.

    Full-sequence model log_probs are sliced and shifted exactly once by verl's
    no_padding_2_padding utility. Preserve the older model-slice adapter above
    for its standalone validation scripts, but do not use it directly in PPO.
    """
    outputs = list(rollout.non_tensor_batch["rlt_outputs"])
    if any(output.finish_reason in ("abort", "error") for output in outputs):
        raise ValueError("aborted/error outputs cannot enter ordinary PPO training")
    batch = rollout_to_tensordict(outputs, expected_version=expected_version, max_loops=4, packed=packed, device=device)
    response_lengths = [len(output.token_ids) for output in outputs]
    offsets = torch.tensor([0] + response_lengths, device=device).cumsum(0)

    def response_nested(rows):
        return torch.nested.nested_tensor_from_jagged(torch.cat(rows), offsets)

    batch["responses"] = response_nested([torch.tensor(out.token_ids, device=device) for out in outputs])
    batch["response_mask"] = response_nested(
        [torch.ones(size, dtype=torch.bool, device=device) for size in response_lengths]
    )
    batch["rollout_log_probs"] = response_nested(
        [torch.tensor(out.log_probs, dtype=torch.float32, device=device) for out in outputs]
    )
    batch["old_log_probs"] = batch["rollout_log_probs"].detach().clone()
    batch["advantages"] = response_nested(
        [advantages[index, :size].to(device) for index, size in enumerate(response_lengths)]
    )
    prompt_lengths = [len(out.prompt_token_ids) for out in outputs]
    prompt_offsets = torch.tensor([0] + prompt_lengths, device=device).cumsum(0)
    batch["prompts"] = torch.nested.nested_tensor_from_jagged(
        torch.tensor([token for out in outputs for token in out.prompt_token_ids], device=device), prompt_offsets
    )
    loss_rows = [
        torch.tensor([False] * len(out.prompt_token_ids) + [True] * len(out.token_ids), device=device)
        for out in outputs
    ]
    batch["loss_mask"] = torch.nested.nested_tensor_from_jagged(torch.cat(loss_rows), batch["input_ids"].offsets())
    for key, value in (
        ("dp_size", world_size),
        ("batch_num_tokens", global_token_count),
        ("global_batch_size", global_batch_size),
    ):
        tu.assign_non_tensor_data(batch, key, value)
    return batch
