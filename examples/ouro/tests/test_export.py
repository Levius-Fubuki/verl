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
"""Real CUDA FSDP export; also runnable with torchrun -m pytest on two ranks."""

import gc
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from examples.ouro.hf import OuroForCausalLM, OuroHFConfig
from examples.ouro.worker import OuroFSDPEngine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FSDP is required")


@pytest.fixture(scope="module")
def cuda_group(tmp_path_factory):
    device = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device)
    owns_group = not dist.is_initialized()
    if owns_group:
        if "RANK" in os.environ:
            dist.init_process_group("nccl")
        else:
            rendezvous = tmp_path_factory.mktemp("ouro-export") / "rendezvous"
            dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    yield device
    if owns_group:
        dist.destroy_process_group()


def wrapped_engine(device, *, layers=2, hidden_size=32, use_orig_params=True, include_probe=True):
    torch.manual_seed(123)
    model = OuroForCausalLM(
        OuroHFConfig(
            vocab_size=37,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 3 // 2,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=hidden_size // 4,
            total_ut_steps=4,
            max_position_embeddings=128,
        )
    ).to(device=device, dtype=torch.bfloat16)
    # Persistent buffers must not accidentally become rollout weights.
    if include_probe:
        model.register_buffer("export_probe", torch.tensor([17], device=device))
    expected = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
    assert any("early_exit_gate" in name for name in expected)
    assert all(not param.requires_grad for param in model.model.early_exit_gate.parameters())
    for index, layer in enumerate(model.model.layers):
        model.model.layers[index] = FSDP(layer, device_id=device, use_orig_params=use_orig_params)
    # FSDP without original params cannot mix the frozen gate with trainable params.
    # Wrap that gate separately to exercise rejection rather than constructor failure.
    if not use_orig_params:
        model.model.early_exit_gate = FSDP(model.model.early_exit_gate, device_id=device, use_orig_params=False)
    module = FSDP(model, device_id=device, use_orig_params=use_orig_params)
    engine = object.__new__(OuroFSDPEngine)
    engine.module = module
    engine._is_offload_param = False
    engine._uses_fsdp2_cpu_offload_policy = False
    engine._qat_enabled = False
    engine.model_config = SimpleNamespace(lora={}, hf_config=model.config)
    return engine, expected


def test_exports_owned_physical_checkpoint_weights_one_unit_at_a_time(cuda_group, monkeypatch):
    engine, expected = wrapped_engine(cuda_group)
    units = FSDP.fsdp_modules(engine.module)
    real_summon = FSDP.summon_full_params
    entered, active = [], []

    @contextmanager
    def tracked_summon(module, **kwargs):
        assert kwargs.get("recurse") is False, "export must not unshard the whole model"
        assert kwargs.get("writeback") is False
        assert kwargs.get("rank0_only", False) is False
        assert not active, "only one physical FSDP unit may be unsharded"
        entered.append(module)
        with real_summon(module, **kwargs):
            active.append(module)
            try:
                yield
            finally:
                active.pop()

    def forbidden_state_dict(*args, **kwargs):
        pytest.fail("Ouro export must not materialize a bulk FSDP state_dict")

    monkeypatch.setattr(FSDP, "summon_full_params", staticmethod(tracked_summon))
    monkeypatch.setattr(engine.module, "state_dict", forbidden_state_dict)
    iterator, metadata = engine.get_per_tensor_param(layered_summon=True)
    assert metadata is None
    assert entered == [], "creating the iterator must not gather any weights"
    exported = list(iterator)
    assert not active
    assert len(entered) == len(units) and {id(unit) for unit in entered} == {id(unit) for unit in units}
    names = [name for name, _ in exported]
    assert len(names) == len(set(names)) == len(expected)
    assert set(names) == set(expected)
    assert sum(tensor.numel() for _, tensor in exported) == sum(tensor.numel() for tensor in expected.values())
    # Read retained tensors only after every summon context has freed its storage.
    for name, tensor in exported:
        assert tensor.dtype == torch.bfloat16 and tensor.is_cuda
        assert not tensor.requires_grad
        torch.testing.assert_close(tensor.cpu(), expected[name], atol=0, rtol=0)
    # The publisher may mutate/reuse its payload after the iterator has finished.
    for _, tensor in exported:
        tensor.zero_()
    with real_summon(engine.module, writeback=False):
        actual = {name: param.detach().cpu().clone() for name, param in engine.module.named_parameters()}
    assert set(actual) == set(expected)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


def test_streaming_export_gpu_peak_is_bounded_by_largest_physical_unit(cuda_group):
    engine, expected = wrapped_engine(cuda_group, layers=8, hidden_size=256, include_probe=False)
    # The CPU reference does not inflate the CUDA peak. A consumer immediately
    # copies out each payload, as a bounded publisher bucket does in production.
    unit_bytes = [
        sum(
            tensor.numel() * tensor.element_size()
            for name, tensor in expected.items()
            if name.startswith(f"model.layers.{index}.")
        )
        for index in range(8)
    ]
    largest_unit_bytes = max(unit_bytes)
    largest_tensor_bytes = max(tensor.numel() * tensor.element_size() for tensor in expected.values())
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    iterator, _ = engine.get_per_tensor_param()
    seen = set()
    for name, tensor in iterator:
        torch.testing.assert_close(tensor.cpu(), expected[name], atol=0, rtol=0)
        seen.add(name)
        del tensor
    torch.cuda.synchronize()
    extra_peak = torch.cuda.max_memory_allocated() - baseline
    # Allow two unit-sized workspaces, one owned tensor, and allocator overhead.
    # An eager full-model export exceeds this budget even with world_size=1.
    budget = 2 * largest_unit_bytes + 2 * largest_tensor_bytes + 1024 * 1024
    assert extra_peak <= budget, f"export staged {extra_peak} bytes, unit budget is {budget}"
    assert seen == set(expected)


def test_fsdp_export_requires_original_parameters(cuda_group):
    engine, _ = wrapped_engine(cuda_group, use_orig_params=False)
    with pytest.raises(ValueError, match="use_orig_params=True"):
        iterator, _ = engine.get_per_tensor_param()
        list(iterator)


def test_fsdp_export_rejects_manual_parameter_offload(cuda_group):
    engine, _ = wrapped_engine(cuda_group)
    engine._is_offload_param = True
    with pytest.raises(NotImplementedError, match="offload"):
        iterator, _ = engine.get_per_tensor_param()
        list(iterator)
