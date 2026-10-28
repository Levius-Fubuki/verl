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
"""Opt-in serial online N=1 -> N+2=3 experiment using public native operations.

This is a correctness/quality reference, not the engine's batched scheduler.
Prompt inputs always run all loops. Every shallow decode finalizes last-exited
KV before the next token; logits come from the actual selected depth.
"""

import math
from dataclasses import dataclass

import torch
from vllm_rlt.core.kv_cache_manager import KVCacheManager


@dataclass
class LookaheadHead:
    mean: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor

    @classmethod
    def load(cls, path, *, model_revision, hidden_size, max_loops, device="cpu"):
        data = torch.load(path, map_location="cpu", weights_only=True)
        required = {
            "schema",
            "source_depth",
            "target_depth",
            "max_loops",
            "model_revision",
            "mean",
            "scale",
            "weight",
            "bias",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("invalid predictor artifact fields")
        if (data["schema"], data["source_depth"], data["target_depth"], data["max_loops"], data["model_revision"]) != (
            1,
            1,
            3,
            max_loops,
            model_revision,
        ) or max_loops != 4:
            raise ValueError("predictor model/depth metadata mismatch")
        shapes = {"mean": (hidden_size,), "scale": (hidden_size,), "weight": (1, hidden_size), "bias": (1,)}
        tensors = {}
        for key, shape in shapes.items():
            value = data[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid predictor tensor: {key}")
            tensors[key] = value.to(device=device, dtype=torch.float32)
        if not (tensors["scale"] > 0).all():
            raise ValueError("predictor scale must be positive")
        return cls(**tensors)

    def predict(self, hidden):
        normalized = (hidden.float() - self.mean) / self.scale
        return (normalized @ self.weight.T + self.bias).sigmoid().squeeze(-1)


@torch.inference_mode()
def generate(model, prompt, *, max_tokens, policy="full", head=None, threshold=0.5, backend="torch", eos_token_id=None):
    if policy not in ("full", "fixed3", "gate", "nplus2"):
        raise ValueError("unknown exit policy")
    if policy == "nplus2" and head is None:
        raise ValueError("nplus2 requires a loaded head")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    if not prompt or type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("nonempty prompt and positive max_tokens required")
    cfg = model.config
    if cfg.total_ut_steps != 4:
        raise ValueError("experiment requires four-loop Ouro")
    parameter = next(model.parameters())
    cache = KVCacheManager(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        num_blocks=4 * math.ceil((len(prompt) + max_tokens) / 16),
        block_size=16,
        max_loops=4,
        device=parameter.device,
        dtype=parameter.dtype,
        backend=backend,
    )
    if not cache.allocate("online", len(prompt) + max_tokens):
        raise RuntimeError("KV allocation failed")
    tokens, scores, depths, selected, predictions, inputs = [], [], [], [], [], []
    try:
        # Batched causal prefill. Its last row supplies the first response.
        ids = torch.tensor(prompt, device=parameter.device)
        hidden = model.prelude(ids)
        for depth in range(4):
            hidden, _ = model.recurrent(
                hidden, ["online"] * len(prompt), [depth] * len(prompt), list(range(len(prompt))), cache
            )
        inputs.extend([4] * len(prompt))
        logits = model.coda(hidden[-1:]).float()[0]
        actual_depth, chosen_depth, prediction_depth = 4, 4, None
        for output_index in range(max_tokens):
            token = int(logits.argmax())
            tokens.append(token)
            scores.append(float(logits.log_softmax(-1)[token]))
            depths.append(actual_depth)
            selected.append(chosen_depth)
            predictions.append(prediction_depth)
            if token == eos_token_id or output_index + 1 == max_tokens:
                break
            position = len(prompt) + output_index
            hidden = model.prelude(torch.tensor([token], device=parameter.device))
            survival = 1.0
            chosen_depth, prediction_depth = 4, None
            for depth in range(4):
                hidden, gate = model.recurrent(hidden, ["online"], [depth], [position], cache)
                actual_depth = depth + 1
                if policy == "nplus2" and actual_depth == 1:
                    chosen_depth = 3 if float(head.predict(hidden)[0]) >= threshold else 4
                    prediction_depth = 1
                elif policy == "fixed3":
                    chosen_depth = 3
                elif policy == "gate":
                    survival *= 1.0 - float(gate.float().sigmoid()[0])
                    if actual_depth >= 2 and threshold < 1 and 1.0 - survival >= threshold:
                        chosen_depth = actual_depth
                if actual_depth >= chosen_depth:
                    break
            cache.finalize_token("online", position, actual_depth - 1)
            inputs.append(actual_depth)
            logits = model.coda(hidden).float()[0]
    finally:
        cache.free("online")
    return {
        "token_ids": tokens,
        "log_probs": scores,
        "exit_depths": depths,
        "selected_depths": selected,
        "prediction_depths": predictions,
        "input_depths": inputs,
        "used_blocks_after_free": cache.num_used_blocks,
    }
