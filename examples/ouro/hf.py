# SPDX-License-Identifier: Apache-2.0
"""Local HF registration for verl; import this module through external_lib.

Loads the standard Ouro physical weights without executing Hub Python code.
The worker still needs to forward execution_depths (tracked in the next slice).
"""

from dataclasses import asdict, fields

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from vllm_rlt.layers import RotaryEmbedding
from vllm_rlt.models.ouro import OuroConfig as NativeConfig
from vllm_rlt.models.ouro import OuroForCausalLM as NativeModel

from examples.ouro.model import OuroTrainingModel


class OuroHFConfig(PretrainedConfig):
    model_type = "ouro"

    def __init__(self, **kwargs):
        native_values = {field.name: kwargs.pop(field.name) for field in fields(NativeConfig) if field.name in kwargs}
        native = NativeConfig(**native_values)
        super().__init__(
            bos_token_id=native.bos_token_id,
            eos_token_id=native.eos_token_id,
            pad_token_id=native.pad_token_id,
            tie_word_embeddings=False,
            **kwargs,
        )
        for key, value in asdict(native).items():
            setattr(self, key, value)
        self.architectures = ["OuroForCausalLM"]
        self.use_cache = False

    def native_config(self):
        return NativeConfig(**{field.name: getattr(self, field.name) for field in fields(NativeConfig)})

    def to_dict(self):
        result = super().to_dict()
        # Transformers 5 introduces an unused null field. The engine deliberately
        # rejects unknown architectural fields; preserve its native config schema.
        if result.get("rope_parameters") is None:
            result.pop("rope_parameters", None)
        return result


class OuroForCausalLM(PreTrainedModel):
    config_class = OuroHFConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # This slice validates root FSDP only. Do not auto-wrap native decoder layers:
    # replay uses their projections explicitly rather than their serving forward.
    _no_split_modules = ["OuroForCausalLM"]

    def __init__(self, config):
        super().__init__(config)
        native = NativeModel(config.native_config())
        self.model = native.model
        self.lm_head = native.lm_head
        self.gradient_checkpointing = False
        self.requires_grad_(True)
        self.model.early_exit_gate.requires_grad_(False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, RotaryEmbedding):
            # HF materializes nonpersistent buffers from meta with to_empty.
            # They are absent from safetensors and must be rebuilt, not left
            # uninitialized or restored by a checkpoint weight-key comparison.
            module.inv_freq = module.frequencies(device=module.inv_freq.device)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs not in (None, {}, {"use_reentrant": False}):
            raise ValueError("Ouro replay uses non-reentrant activation checkpointing")
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        execution_depths=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        if use_cache or kwargs:
            raise ValueError("Ouro training replay accepts full sequences, masks and execution_depths only")
        output = OuroTrainingModel.forward(
            self, input_ids, attention_mask=attention_mask, position_ids=position_ids, execution_depths=execution_depths
        )
        return CausalLMOutput(logits=output.logits) if return_dict else (output.logits,)


def register_ouro():
    AutoConfig.register("ouro", OuroHFConfig, exist_ok=True)
    AutoModelForCausalLM.register(OuroHFConfig, OuroForCausalLM, exist_ok=True)


register_ouro()
