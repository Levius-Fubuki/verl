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
"""Publisher checks against tiny native CPU engines, without GPU/NIXL claims."""

import copy
import importlib.util

import pytest
import torch
from vllm_rlt import CacheConfig, SamplingParams
from vllm_rlt.engine.llm_engine import LLMEngine

from examples.ouro.tests.test_replay import native_model


def publisher_api():
    assert importlib.util.find_spec("examples.ouro.publication") is not None, "Ouro weight publisher is missing"
    from examples.ouro.publication import OuroWeightPublisher

    return OuroWeightPublisher


@pytest.fixture
def native_pair():
    if not hasattr(LLMEngine, "start_weight_update"):
        pytest.skip("#83 is rollout-only; native publication is checked separately against the pinned prototype")
    target = native_model()
    source = copy.deepcopy(target)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(0.01)
    engine = LLMEngine(target, cache_config=CacheConfig(num_blocks=32, block_size=2))
    yield engine, source
    engine.close()


def manifest(model):
    return {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}


def test_streamed_physical_bf16_publication_preserves_native_addresses(native_pair, monkeypatch):
    engine, source = native_pair
    cls = publisher_api()
    expected = manifest(source)
    pointers = {name: parameter.data_ptr() for name, parameter in engine.model.named_parameters()}
    calls, yielded = [], []
    original = engine.update_weights
    cap = 512

    def record(weights):
        assert isinstance(weights, list)
        calls.append([(name, tensor.numel() * tensor.element_size()) for name, tensor in weights])
        for _, tensor in weights:
            assert type(tensor) is torch.Tensor
            assert not tensor.requires_grad and tensor.grad_fn is None
            assert tensor.dtype == torch.bfloat16
        original(weights)

    monkeypatch.setattr(engine, "update_weights", record)

    def weights():
        for name, parameter in source.named_parameters():
            # A list of the entire generator would violate this bounded lookahead.
            if yielded:
                assert calls, "the generator was materialized before the first transfer"
            yielded.append(name)
            yield name, parameter

    publisher = cls(engine, bucket_bytes=cap, use_bf16=True, expected_manifest=expected)
    assert publisher.publish(weights(), global_step=0) == 1
    assert yielded == list(expected)
    assert [name for call in calls for name, _ in call] == list(expected)
    assert all(sum(size for _, size in call) <= cap or len(call) == 1 for call in calls)
    assert any(call[0][1] > cap for call in calls)
    for name, parameter in engine.model.named_parameters():
        torch.testing.assert_close(parameter, dict(source.named_parameters())[name].to(torch.bfloat16).float())
        assert parameter.data_ptr() == pointers[name]
    assert publisher.publish(source.named_parameters(), global_step=1) == 2
    engine.add_request("version", [2, 3], SamplingParams(max_tokens=1))
    outputs = []
    while engine.has_unfinished_requests():
        outputs.extend(engine.step())
    assert outputs[-1].weight_version == 2


@pytest.mark.parametrize("step", [-1, True, 1.5, "1"])
def test_invalid_global_step_does_not_start_a_transaction(native_pair, step):
    engine, source = native_pair
    publisher = publisher_api()(engine)
    with pytest.raises(ValueError, match="global_step"):
        publisher.publish(source.named_parameters(), global_step=step)
    assert engine.get_weight_version() == 0
    engine.add_request("still available", [2], SamplingParams(max_tokens=1))
    engine.abort_request("still available")


@pytest.mark.parametrize("fault", ["duplicate", "missing", "unknown", "shape", "generator"])
def test_invalid_full_stream_keeps_native_generation_fenced_until_retry(native_pair, fault):
    engine, source = native_pair
    publisher = publisher_api()(engine, bucket_bytes=1, expected_manifest=manifest(source))

    def broken():
        for index, (name, parameter) in enumerate(source.named_parameters()):
            if index == 1:
                if fault == "missing":
                    continue
                if fault == "unknown":
                    name = "persistent_buffer"
                if fault == "shape":
                    parameter = parameter.flatten()[:1]
                if fault == "generator":
                    raise RuntimeError("generator failed")
            yield name, parameter
            if index == 0 and fault == "duplicate":
                yield name, parameter

    with pytest.raises((ValueError, RuntimeError)):
        publisher.publish(broken(), global_step=0)
    assert publisher.last_attempted_version == 1
    assert engine.get_weight_version() == 0
    with pytest.raises(RuntimeError, match="paused"):
        engine.add_request("blocked", [2], SamplingParams(max_tokens=1))
    assert publisher.publish(source.named_parameters(), global_step=0) == 2
    engine.add_request("recovered", [2], SamplingParams(max_tokens=1))
    engine.abort_request("recovered")


def test_checkpoint_counters_allow_fresh_backend_and_reject_stale_steps(native_pair):
    engine, source = native_pair
    cls = publisher_api()
    publisher = cls(engine, use_bf16=False)
    assert publisher.publish(source.named_parameters(), global_step=3) == 4
    state = publisher.state_dict()
    fresh = LLMEngine(native_model(), cache_config=CacheConfig(num_blocks=32, block_size=2))
    try:
        restored = cls(fresh, use_bf16=False)
        restored.load_state_dict(state)
        assert restored.state_dict() == state
        assert restored.publish(source.named_parameters(), global_step=3) == 5
        assert restored.global_step == 3
        assert restored.committed_version == 5
        with pytest.raises(ValueError, match="global_step"):
            restored.publish(source.named_parameters(), global_step=2)
        for name, parameter in fresh.model.named_parameters():
            torch.testing.assert_close(parameter, dict(source.named_parameters())[name], rtol=0, atol=0)
    finally:
        fresh.close()


def test_loading_older_checkpoint_cannot_reuse_a_failed_attempt(native_pair):
    engine, source = native_pair
    publisher = publisher_api()(engine)
    old_state = publisher.state_dict()

    def broken():
        yield next(source.named_parameters())
        raise RuntimeError("producer failed")

    with pytest.raises(RuntimeError, match="producer failed"):
        publisher.publish(broken(), global_step=0)
    publisher.load_state_dict(old_state)
    assert publisher.last_attempted_version == 1
    assert publisher.publish(source.named_parameters(), global_step=0) == 2


def test_native_rejection_retains_failed_transfer_fence_without_manifest(native_pair):
    engine, source = native_pair
    publisher = publisher_api()(engine, bucket_bytes=1)
    weights = iter(source.named_parameters())
    first, second = next(weights), next(weights)
    with pytest.raises(ValueError, match="shape"):
        publisher.publish(iter([first, (second[0], torch.zeros(1))]), global_step=0)
    with pytest.raises(RuntimeError, match="successful weight update"):
        engine.finish_weight_update()
    with pytest.raises(RuntimeError, match="paused"):
        engine.add_request("blocked", [2])
    assert publisher.publish(source.named_parameters(), global_step=0) == 2


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64])
def test_nonfloating_physical_weight_is_rejected_before_conversion(native_pair, dtype):
    engine, source = native_pair
    publisher = publisher_api()(engine, bucket_bytes=1, use_bf16=True)
    name, parameter = next(source.named_parameters())
    before = dict(engine.model.named_parameters())[name].detach().clone()
    invalid = parameter.detach().to(dtype=dtype)

    def weights():
        for current_name, current_parameter in source.named_parameters():
            yield current_name, invalid if current_name == name else current_parameter

    with pytest.raises(ValueError, match="floating"):
        publisher.publish(weights(), global_step=0)
    torch.testing.assert_close(dict(engine.model.named_parameters())[name], before, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="paused"):
        engine.add_request("blocked", [2])
    assert publisher.publish(source.named_parameters(), global_step=0) == 2


def test_commit_confirmation_does_not_acknowledge_wrong_backend_version(native_pair, monkeypatch):
    engine, source = native_pair
    publisher = publisher_api()(engine)
    original = engine.get_weight_version
    calls = 0

    def stale_reply():
        nonlocal calls
        calls += 1
        return original() if calls == 1 else 0

    monkeypatch.setattr(engine, "get_weight_version", stale_reply)
    with pytest.raises(RuntimeError, match="committed weight version"):
        publisher.publish(source.named_parameters(), global_step=0)
    assert publisher.global_step is None and publisher.committed_version == 0
    assert publisher.last_attempted_version == 1
    monkeypatch.setattr(engine, "get_weight_version", original)
    assert publisher.publish(source.named_parameters(), global_step=0) == 2


def test_partial_peer_commit_retry_uses_newer_attempt_and_checkpoint(native_pair):
    """Only failure delivery is injected; both peers execute native update APIs."""
    engine, source = native_pair
    other = LLMEngine(native_model(), cache_config=CacheConfig(num_blocks=32, block_size=2))

    class FailingTransport:
        fail = True

        def get_weight_version(self):
            return engine.get_weight_version() if not self.fail else 0

        def start_weight_update(self, version):
            engine.start_weight_update(version)
            other.start_weight_update(version)

        def update_weights(self, weights):
            engine.update_weights(weights)
            other.update_weights(weights)

        def finish_weight_update(self):
            engine.finish_weight_update()
            if self.fail:
                raise RuntimeError("transport failed after first peer committed")
            other.finish_weight_update()

    try:
        transport = FailingTransport()
        publisher = publisher_api()(transport, bucket_bytes=512)
        with pytest.raises(RuntimeError, match="transport failed"):
            publisher.publish(source.named_parameters(), global_step=0)
        assert engine.get_weight_version() == 1
        assert other.get_weight_version() == 0
        with pytest.raises(RuntimeError):
            other.add_request("blocked", [2])
        restored = publisher_api()(transport)
        restored.load_state_dict(publisher.state_dict())
        transport.fail = False
        assert restored.publish(source.named_parameters(), global_step=0) == 2
        assert engine.get_weight_version() == other.get_weight_version() == 2
    finally:
        other.close()


def test_partial_native_start_failure_reserves_version_before_transport_error(native_pair):
    """The first peer really starts; injected delivery failure skips the second."""
    engine, source = native_pair
    other = LLMEngine(native_model(), cache_config=CacheConfig(num_blocks=32, block_size=2))

    class FailingStartTransport:
        fail = True

        def get_weight_version(self):
            return engine.get_weight_version()

        def start_weight_update(self, version):
            assert publisher.last_attempted_version == version
            attempts.append(version)
            engine.start_weight_update(version)
            if self.fail:
                raise RuntimeError("transport failed after first peer started")
            other.start_weight_update(version)

        def update_weights(self, weights):
            engine.update_weights(weights)
            other.update_weights(weights)

        def finish_weight_update(self):
            engine.finish_weight_update()
            other.finish_weight_update()

    try:
        transport, attempts = FailingStartTransport(), []
        publisher = publisher_api()(transport, bucket_bytes=512)
        with pytest.raises(RuntimeError, match="first peer started"):
            publisher.publish(source.named_parameters(), global_step=0)
        assert publisher.last_attempted_version == 1
        assert publisher.committed_version == engine.get_weight_version() == other.get_weight_version() == 0
        assert publisher.global_step is None
        with pytest.raises(RuntimeError, match="paused"):
            engine.add_request("blocked", [2])
        transport.fail = False
        assert publisher.publish(source.named_parameters(), global_step=0) == 2
        assert attempts == [1, 2]
        assert engine.get_weight_version() == other.get_weight_version() == 2
    finally:
        other.close()
