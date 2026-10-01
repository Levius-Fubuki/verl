# SPDX-License-Identifier: Apache-2.0
"""Differentiable, shared-weight Ouro replay for the verl/FSDP integration.

Physical layers and checkpoint names come from vllm-rlt. No serving cache or
inference-mode forward is used during training. The input trace is fixed data:
this computes a token-policy gradient conditional on the trace, not a gradient
of the discrete exit policy. See README.md for the current integration boundary.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from vllm_rlt.models.ouro import OuroForCausalLM


def input_execution_depths(prompt_length: int, output_depths: list[int], *, max_loops: int) -> list[int]:
    """Shift RequestOutput.exit_depths onto prompt+response input positions.

    The first response is predicted by the final full-depth prompt input. The
    last response input has no response target and MUST be loss-masked by the
    caller; it is assigned depth 1. No implicit off-by-one correction elsewhere.
    """
    if type(prompt_length) is not int or prompt_length < 1:
        raise ValueError("prompt_length must be a positive integer")
    if type(max_loops) is not int or max_loops < 1:
        raise ValueError("max_loops must be a positive integer")
    if any(type(depth) is not int or not 1 <= depth <= max_loops for depth in output_depths):
        raise ValueError("output depths must be integers within the supported range")
    if output_depths and output_depths[0] != max_loops:
        raise ValueError("the first output depth must match full-depth prefill")
    return [max_loops] * prompt_length + (list(output_depths[1:]) + [1] if output_depths else [])


@dataclass
class OuroTrainingOutput:
    logits: torch.Tensor


def _replay_layer(layer, hidden, cos, sin, active, causal_mask, previous_k, previous_v):
    """Batch all token positions at one depth; retain differentiable copied KV."""
    attention = layer.self_attn
    cfg = attention.config
    batch, length, _ = hidden.shape
    value = layer.input_layernorm(hidden)
    q, k, v = (
        projection(value).view(batch, length, -1, cfg.head_dim).transpose(1, 2)
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    )

    def rotate(tensor):
        first, second = tensor.chunk(2, dim=-1)
        return tensor * cos + torch.cat((-second, first), dim=-1) * sin

    q, k = rotate(q), rotate(k)
    if previous_k is not None:
        # A token that exited at d must supply its depth-d KV at every d'>d.
        # Using detach/in-place inference-cache writes here loses its gradient.
        kv_active = active[:, None, :, None]
        k = torch.where(kv_active, k, previous_k)
        v = torch.where(kv_active, v, previous_v)
    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, dropout_p=0.0, enable_gqa=True)
    attended = attended.transpose(1, 2).reshape(batch, length, -1)
    updated = hidden + layer.input_layernorm_2(attention.o_proj(attended))
    updated = updated + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(updated)))
    return torch.where(active[:, :, None], updated, hidden), k, v


class OuroTrainingModel(nn.Module):
    """Trainable model component; takes ownership of the supplied native modules.

    This is not yet the verl worker/AutoModel registration or rollout adapter.
    Native state_dict names are unchanged, allowing strict checkpoint loading
    and FSDP wrapping without expanding physical weights for the four loops.
    """

    def __init__(self, native: OuroForCausalLM):
        super().__init__()
        self.config = native.config
        self.model = native.model
        self.lm_head = native.lm_head
        self.requires_grad_(True)
        self.model.early_exit_gate.requires_grad_(False)
        self.gradient_checkpointing = False

    @classmethod
    def from_pretrained(cls, path_or_repo, *, revision=None, device="cpu", dtype=torch.bfloat16):
        return cls(OuroForCausalLM.from_pretrained(path_or_repo, revision=revision, device=device, dtype=dtype))

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids,
        *,
        attention_mask=None,
        execution_depths=None,
        position_ids=None,
    ):
        if input_ids.ndim != 2 or min(input_ids.shape) < 1 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be a nonempty integer [batch, sequence] tensor")
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask
        if valid.shape != input_ids.shape or valid.device != input_ids.device:
            raise ValueError("attention mask must match input shape and device")
        if not bool(((valid == 0) | (valid == 1)).all()):
            raise ValueError("attention mask must contain only zero or one")
        valid = valid.bool()
        if not bool(valid.any(dim=1).all()):
            raise ValueError("attention mask must contain a token in every row")
        # A row is one sequence, optionally left/right padded, never packed samples.
        positions = valid.long().cumsum(dim=1) - 1
        width = valid.shape[1]
        indices = torch.arange(width, device=input_ids.device)[None, :]
        first = torch.where(valid, indices, width).amin(dim=1)
        last = torch.where(valid, indices, -1).amax(dim=1)
        if not bool((last - first + 1 == valid.sum(dim=1)).all()):
            raise ValueError("attention mask must describe a contiguous sequence per row")
        positions = positions.clamp_min(0)
        if position_ids is not None:
            if position_ids.shape != input_ids.shape or position_ids.device != input_ids.device:
                raise ValueError("position_ids must match input shape and device")
            if not torch.equal(position_ids[valid], positions[valid]):
                raise ValueError("position_ids must start at zero for each unpadded sequence")
        if execution_depths is None:
            execution_depths = torch.full_like(input_ids, self.config.total_ut_steps)
        if (
            execution_depths.shape != input_ids.shape
            or execution_depths.device != input_ids.device
            or execution_depths.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("execution depths must be integer tensors matching input shape and device")
        if not bool(((execution_depths[valid] >= 1) & (execution_depths[valid] <= self.config.total_ut_steps)).all()):
            raise ValueError("execution depths are outside the model's supported range")
        hidden = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(hidden.reshape(-1, hidden.shape[-1]), positions.reshape(-1))
        cos = cos.reshape(*input_ids.shape, -1).unsqueeze(1)
        sin = sin.reshape(*input_ids.shape, -1).unsqueeze(1)
        causal = indices[:, :, None] >= indices[:, None, :]
        causal_mask = (causal & valid[:, None, :]).unsqueeze(1)
        previous = [(None, None)] * len(self.model.layers)
        # Execute a fixed number of calls regardless of per-rank traces, so future
        # FSDP collectives follow the same order on ranks with different depths.
        for depth in range(self.config.total_ut_steps):
            active = valid & (execution_depths > depth)
            for index, layer in enumerate(self.model.layers):
                args = (layer, hidden, cos, sin, active, causal_mask, *previous[index])
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    hidden, k, v = checkpoint(_replay_layer, *args, use_reentrant=False)
                else:
                    hidden, k, v = _replay_layer(*args)
                previous[index] = (k, v)
            hidden = torch.where(active[:, :, None], self.model.norm(hidden), hidden)
        return OuroTrainingOutput(logits=self.lm_head(hidden))
