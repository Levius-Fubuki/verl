# SPDX-License-Identifier: Apache-2.0
"""The model must be loadable through the AutoModel entry point used by verl."""

import importlib.util

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM
from vllm_rlt.models.ouro import OuroForCausalLM as NativeModel

from examples.ouro.tests.test_replay import engine_logits, native_model


def test_hf_native_checkpoint_load_save_and_backward(tmp_path):
    assert importlib.util.find_spec("examples.ouro.hf") is not None, "Ouro HF registration is missing"
    from examples.ouro.hf import register_ouro

    register_ouro()
    native = native_model()
    import json

    (tmp_path / "config.json").write_text(json.dumps(native.config.to_dict()))
    save_file(native.state_dict(), tmp_path / "model.safetensors")
    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(tmp_path, config=config, trust_remote_code=False)
    torch.testing.assert_close(model.model.rotary_emb.inv_freq, native.model.rotary_emb.inv_freq, atol=0, rtol=0)
    assert set(model.state_dict()) == set(native.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, native.state_dict()[name], atol=0, rtol=0)
    ids = torch.tensor([[2, 3, 5, 7]])
    depths = torch.tensor([[4, 1, 3, 4]])
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    output = model(input_ids=ids, execution_depths=depths, use_cache=False, return_dict=True)
    torch.testing.assert_close(output.logits[0], engine_logits(native, ids[0], depths[0]), atol=3e-6, rtol=3e-5)
    output.logits.square().mean().backward()
    assert model.model.layers[0].self_attn.k_proj.weight.grad is not None
    saved = tmp_path / "saved"
    model.save_pretrained(saved)
    reloaded = AutoModelForCausalLM.from_pretrained(saved, trust_remote_code=False)
    torch.testing.assert_close(reloaded.model.rotary_emb.inv_freq, native.model.rotary_emb.inv_freq, atol=0, rtol=0)
    with torch.no_grad():
        expected = reloaded(input_ids=ids, execution_depths=depths, use_cache=False).logits
    torch.testing.assert_close(expected, output.logits.detach(), atol=0, rtol=0)
    # The saved physical checkpoint must also remain consumable by the engine.
    serving = NativeModel.from_pretrained(saved, dtype=torch.float32)
    torch.testing.assert_close(engine_logits(serving, ids[0], depths[0]), expected[0], atol=3e-6, rtol=3e-5)
