# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
import torch
from vllm_rlt.request import RequestOutput
from vllm_rlt.sampling_params import SamplingParams

from examples.ouro.tests.test_replay import native_model
from examples.ouro.worker import OuroFSDPEngine, rollout_to_tensordict


def sample():
    return RequestOutput(
        "x",
        [2, 3],
        [5, 7, 11],
        [4, 1, 3],
        True,
        log_probs=[-1.0, -2.0, -3.0],
        weight_version=2,
        sampling_params=SamplingParams(temperature=1.0, logprobs=0),
    )


def test_rollout_trace_mask_and_version():
    batch = rollout_to_tensordict([sample()], expected_version=2, max_loops=4)
    assert batch["execution_depths"].values().tolist() == [4, 4, 1, 3, 1]
    assert batch["response_mask"].values().tolist() == [False, True, True, True, False]
    assert batch["rollout_log_probs"].values().tolist() == [0.0, -1.0, -2.0, -3.0, 0.0]
    with pytest.raises(ValueError, match="version"):
        rollout_to_tensordict([sample()], expected_version=3, max_loops=4)


def test_worker_preserves_depths_and_rejects_missing_trace():
    engine = object.__new__(OuroFSDPEngine)
    engine.pad_to_length = False
    engine.use_ulysses_sp = False
    engine.pass_packed_cu_seqlens = False
    batch = rollout_to_tensordict([sample()], expected_version=2, max_loops=4)
    inputs, _ = engine.prepare_model_inputs(batch)
    assert inputs["execution_depths"].tolist() == [[4, 4, 1, 3, 1]]
    del batch["execution_depths"]
    with pytest.raises(ValueError, match="execution_depths"):
        engine.prepare_model_inputs(batch)


def test_packed_replay_has_independent_sequences_and_gradients():
    from examples.ouro.hf import OuroForCausalLM, OuroHFConfig

    native = native_model()
    model = OuroForCausalLM(OuroHFConfig(**native.config.to_dict()))
    model.load_state_dict(native.state_dict())
    other = copy.deepcopy(model)
    ids = torch.tensor([[2, 3, 5, 7, 11]])
    depths = torch.tensor([[4, 1, 3, 4, 2]])
    calls = []
    handle = model.model.layers[0].register_forward_hook(lambda *args: calls.append(1))
    packed = model(
        ids,
        execution_depths=depths,
        position_ids=torch.tensor([[0, 1, 2, 0, 1]]),
        cu_seqlens=torch.tensor([0, 3, 5]),
        cu_seqlens_cpu=torch.tensor([0, 3, 5]),
    ).logits
    handle.remove()
    assert len(calls) == 4
    separate = torch.cat(
        [
            other(ids[:, :3], execution_depths=depths[:, :3]).logits,
            other(ids[:, 3:], execution_depths=depths[:, 3:]).logits,
        ],
        dim=1,
    )
    torch.testing.assert_close(packed, separate)
    packed.square().sum().backward()
    separate.square().sum().backward()
    for a, b in zip(model.parameters(), other.parameters(), strict=True):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=3e-5)


def test_misaligned_trace_rejected():
    engine = object.__new__(OuroFSDPEngine)
    engine.pad_to_length = False
    engine.use_ulysses_sp = False
    batch = rollout_to_tensordict([sample()], expected_version=2, max_loops=4)
    batch["execution_depths"] = torch.nested.nested_tensor([torch.tensor([4, 1])], layout=torch.jagged)
    with pytest.raises(ValueError, match="offsets"):
        engine.prepare_model_inputs(batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_jagged_dense_padding_matches_ids():
    from dataclasses import replace

    engine = object.__new__(OuroFSDPEngine)
    engine.pad_to_length = False
    engine.use_ulysses_sp = False
    shorter = replace(sample(), prompt_token_ids=[2])
    batch = rollout_to_tensordict([sample(), shorter], expected_version=2, max_loops=4, device="cuda")
    inputs, _ = engine.prepare_model_inputs(batch)
    assert inputs["execution_depths"].shape == inputs["input_ids"].shape == (2, 5)
