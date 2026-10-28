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
"""Check training replay against chronological inference, not a second replay."""

import copy
import importlib.util

import pytest
import torch
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM


def training_api():
    assert importlib.util.find_spec("examples.ouro.model") is not None, "Ouro training replay is missing"
    from examples.ouro.model import OuroTrainingModel, input_execution_depths

    return OuroTrainingModel, input_execution_depths


def native_model(device="cpu", dtype=torch.float32, kv_heads=2):
    torch.manual_seed(123)
    config = OuroConfig(
        vocab_size=37,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        head_dim=8,
        total_ut_steps=4,
        max_position_embeddings=128,
    )
    return OuroForCausalLM(config).to(device=device, dtype=dtype)


@torch.no_grad()
def engine_logits(native, tokens, depths, backend="torch"):
    cfg = native.config
    cache = KVCacheManager(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        num_blocks=cfg.total_ut_steps * ((len(tokens) + 15) // 16),
        block_size=16,
        max_loops=cfg.total_ut_steps,
        device=tokens.device,
        dtype=next(native.parameters()).dtype,
        backend=backend,
    )
    assert cache.allocate("test", len(tokens))
    outputs = []
    for position, token in enumerate(tokens):
        hidden = native.prelude(token.reshape(1))
        for depth in range(int(depths[position])):
            hidden, _ = native.recurrent(hidden, ["test"], [depth], [position], cache)
        cache.finalize_token("test", position, int(depths[position]) - 1)
        outputs.append(native.coda(hidden)[0])
    return torch.stack(outputs)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("depths", [[4] * 6, [4, 4, 1, 3, 2, 4]])
def test_matches_native_last_exited(kv_heads, depths):
    cls, _ = training_api()
    native = native_model(kv_heads=kv_heads)
    tokens = torch.tensor([2, 3, 5, 7, 11, 13])
    expected = engine_logits(native, tokens, depths)
    model = cls(copy.deepcopy(native))
    actual = model(tokens[None], execution_depths=torch.tensor([depths])).logits[0]
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


def test_batched_left_and_right_padding_and_causality():
    cls, _ = training_api()
    native = native_model()
    model = cls(copy.deepcopy(native))
    ids = torch.tensor([[0, 0, 2, 3, 5, 7], [11, 13, 17, 0, 0, 0]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    depths = torch.tensor([[0, 0, 4, 4, 1, 3], [4, 1, 4, 0, 0, 0]])
    actual = model(ids, attention_mask=mask, execution_depths=depths).logits
    for row in range(2):
        valid = mask[row].bool()
        expected = engine_logits(native, ids[row, valid], depths[row, valid])
        torch.testing.assert_close(actual[row, valid], expected, atol=3e-6, rtol=3e-5)
    changed = ids.clone()
    changed[0, -1] = 19
    other = model(changed, attention_mask=mask, execution_depths=depths).logits
    torch.testing.assert_close(actual[0, 2:-1], other[0, 2:-1], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], other[1], atol=0, rtol=0)


def test_gradient_checkpointing_and_physical_weight_sharing():
    cls, _ = training_api()
    native = native_model()
    first, second = cls(copy.deepcopy(native)), cls(copy.deepcopy(native))
    second.gradient_checkpointing_enable()
    ids = torch.tensor([[2, 3, 5, 7, 11, 13]])
    depths = torch.tensor([[4, 4, 1, 3, 2, 4]])
    assert set(first.state_dict()) == set(native.state_dict())
    assert sum(p.numel() for p in first.parameters()) == sum(p.numel() for p in native.parameters())
    for model in (first, second):
        # A loss only on the last token exercises gradients through earlier copied KV.
        model(ids, execution_depths=depths).logits[0, -1].square().sum().backward()
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters(), strict=True):
        if "early_exit_gate" in name:
            assert not a.requires_grad and a.grad is None
        else:
            assert a.grad is not None and torch.isfinite(a.grad).all(), name
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=3e-5)
    param = first.model.embed_tokens.weight
    index = (5, 3)  # token 5 exited shallowly; later tokens attend to its copied KV
    analytical = param.grad[index].item()
    epsilon = 0.002
    with torch.no_grad():
        original = param[index].item()
        param[index] = original + epsilon
        plus = first(ids, execution_depths=depths).logits[0, -1].square().sum().item()
        param[index] = original - epsilon
        minus = first(ids, execution_depths=depths).logits[0, -1].square().sum().item()
        param[index] = original
    assert abs(analytical) > 1e-5
    assert analytical == pytest.approx((plus - minus) / (2 * epsilon), abs=2e-4, rel=0.03)


def test_shifted_output_trace_and_validation():
    _, convert = training_api()
    assert convert(3, [4, 1, 3], max_loops=4) == [4, 4, 4, 1, 3, 1]
    assert convert(3, [], max_loops=4) == [4, 4, 4]
    for trace in ([1, 2], [4, 0], [4, 5], [4, 1.5], [4, True]):
        with pytest.raises(ValueError):
            convert(3, trace, max_loops=4)


@pytest.mark.parametrize("bad_depths", [[[4, 0, 1]], [[4, 5, 1]], [[4.0, 1.0, 1.0]], [[4, 1]]])
def test_invalid_depths_rejected(bad_depths):
    cls, _ = training_api()
    with pytest.raises(ValueError, match="depth"):
        cls(native_model())(torch.tensor([[2, 3, 5]]), execution_depths=torch.tensor(bad_depths))


def test_invalid_mask_rejected():
    cls, _ = training_api()
    for mask in ([[0, 0, 0]], [[1, 0, 1]], [[1, 2, 1]]):
        with pytest.raises(ValueError, match="mask"):
            cls(native_model())(torch.tensor([[2, 3, 5]]), attention_mask=torch.tensor(mask))


def test_optimizer_checkpoint_resume(tmp_path):
    cls, _ = training_api()
    first = cls(native_model())
    ids = torch.tensor([[2, 3, 5, 7]])
    depths = torch.tensor([[4, 1, 3, 4]])
    optimizer = torch.optim.AdamW((p for p in first.parameters() if p.requires_grad), lr=1e-3)

    def step(model, optim):
        optim.zero_grad(set_to_none=True)
        loss = model(ids, execution_depths=depths).logits.square().mean()
        loss.backward()
        optim.step()

    original = first.model.layers[0].self_attn.q_proj.weight.detach().clone()
    step(first, optimizer)
    assert not torch.equal(original, first.model.layers[0].self_attn.q_proj.weight)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": first.state_dict(), "optimizer": optimizer.state_dict()}, path)
    second = cls(native_model())
    restored = torch.optim.AdamW((p for p in second.parameters() if p.requires_grad), lr=1e-3)
    checkpoint = torch.load(path, weights_only=True)
    second.load_state_dict(checkpoint["model"], strict=True)
    restored.load_state_dict(checkpoint["optimizer"])
    step(first, optimizer)
    step(second, restored)
    for name, param in first.state_dict().items():
        torch.testing.assert_close(param, second.state_dict()[name], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_cuda_bfloat16_replay_backward(backend):
    cls, _ = training_api()
    native = native_model(device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[2, 3, 5, 7, 11, 13]], device="cuda")
    depths = torch.tensor([[4, 4, 1, 3, 2, 4]], device="cuda")
    expected = engine_logits(native, ids[0], depths[0], backend=backend)
    model = cls(copy.deepcopy(native))
    model.gradient_checkpointing_enable()
    output = model(ids, execution_depths=depths).logits[0]
    torch.testing.assert_close(output, expected, atol=0.008, rtol=0.05)
    output.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
