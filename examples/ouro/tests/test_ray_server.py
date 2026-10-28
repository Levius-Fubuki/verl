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
"""Exercise asynchronous ownership with real CPU stage engines."""

import asyncio
import importlib.util

import pytest
from vllm_rlt import CacheConfig
from vllm_rlt.engine.stage_engine import StageEngine

from examples.ouro.tests.test_replay import native_model


def api():
    assert importlib.util.find_spec("examples.ouro.ray_server") is not None, "async server is missing"
    from examples.ouro.ray_server import OuroTokenServer

    return OuroTokenServer


def server():
    if not hasattr(StageEngine, "start_weight_update"):
        pytest.skip("requires actual #78 publication implementation")
    cls = api()
    model = native_model()
    backend = StageEngine(model, cache_config=CacheConfig(128, 4))
    return cls(backend=backend), model


def test_concurrent_requests_abort_publication_fence_and_retry():
    async def run():
        owner, model = server()
        try:
            with pytest.raises(RuntimeError, match="publication"):
                await owner.generate("before", prompt_ids=[2, 3], sampling_params={"max_tokens": 2})
            assert await owner.publish(list(model.named_parameters()), global_step=0) == 1
            task = asyncio.create_task(owner.generate("long", prompt_ids=[2, 3], sampling_params={"max_tokens": 80}))
            # Wait until admitted to the actual native owner, then cancel in flight.
            for _ in range(100):
                if (await owner.stats())["active"]:
                    break
                await asyncio.sleep(0)
            assert await owner.abort_request("long")
            with pytest.raises(RuntimeError, match="abort"):
                await task
            a, b = await asyncio.gather(
                owner.generate("a", prompt_ids=[2, 3], sampling_params={"max_tokens": 3, "seed": 7}),
                owner.generate("b", prompt_ids=[5, 7], sampling_params={"max_tokens": 3, "seed": 8}),
            )
            assert a.extra_fields["weight_version"] == b.extra_fields["weight_version"] == 1
            assert a.extra_fields["global_steps"] == 0
            assert a.extra_fields["sampling_params"]["seed"] == 7
            assert len(a.log_probs) == len(a.token_ids) == len(a.extra_fields["exit_depths"])
            with pytest.raises(ValueError, match="duplicate"):
                first = next(model.named_parameters())
                await owner.publish([first, first], global_step=1)
            with pytest.raises(RuntimeError, match="publication"):
                await owner.generate("bad", prompt_ids=[2, 3], sampling_params={"max_tokens": 2})
            assert await owner.publish(list(model.named_parameters()), global_step=1) == 3
            out = await owner.generate("reused", prompt_ids=[2, 3], sampling_params={"max_tokens": 2})
            assert out.extra_fields["weight_version"] == 3
            assert out.extra_fields["global_steps"] == 1
            assert (await owner.stats())["active"] == 0
        finally:
            await owner.close()

    asyncio.run(run())


def test_coroutine_cancellation_retires_owned_request_and_stale_abort_is_ignored():
    async def run():
        owner, model = server()
        try:
            await owner.publish(list(model.named_parameters()), global_step=0)
            task = asyncio.create_task(owner.generate("same", prompt_ids=[2, 3], sampling_params={"max_tokens": 80}))
            for _ in range(100):
                if (await owner.stats())["active"]:
                    break
            old_generation = (await owner.stats())["generations"]["same"]
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            newer = asyncio.create_task(owner.generate("same", prompt_ids=[2, 3], sampling_params={"max_tokens": 20}))
            for _ in range(100):
                if (await owner.stats())["active"]:
                    break
            assert not await owner.abort_request("same", generation=old_generation)
            out = await newer
            assert out.extra_fields["request_generation"] != old_generation
        finally:
            await owner.close()

    asyncio.run(run())


def test_registry_resolves_standalone_and_rejects_pd():
    api()
    from verl.workers.rollout.replica import get_rollout_replica_class

    assert get_rollout_replica_class("vllm_rlt").__name__ == "OuroReplica"
    with pytest.raises(NotImplementedError):
        get_rollout_replica_class("vllm_rlt", disaggregation_enabled=True)


def test_immediate_abort_is_ordered_after_queued_admission():
    async def run():
        owner, model = server()
        try:
            await owner.publish(list(model.named_parameters()), global_step=0)
            for _ in range(3):
                await asyncio.sleep(0)
            generation = asyncio.create_task(
                owner.generate("fifo", prompt_ids=[2, 3], sampling_params={"max_tokens": 20})
            )
            abort = asyncio.create_task(owner.abort_request("fifo"))
            assert await abort
            with pytest.raises(RuntimeError, match="abort"):
                await generation
        finally:
            await owner.close()

    asyncio.run(run())


def test_fresh_owner_restore_requires_republication_and_advances_real_version():
    async def run():
        owner, model = server()
        restored = None
        try:
            await owner.publish(list(model.named_parameters()), global_step=0)
            old = await owner.generate("old", prompt_ids=[2, 3], sampling_params={"max_tokens": 2})
            assert hasattr(owner, "state_dict"), "server publisher restore is missing"
            snapshot = await owner.state_dict()
            await owner.close()
            restored, new_model = server()
            await restored.load_state_dict(snapshot)
            with pytest.raises(RuntimeError, match="publication"):
                await restored.generate("new", prompt_ids=[2, 3], sampling_params={"max_tokens": 2})
            assert await restored.publish(list(new_model.named_parameters()), global_step=0) == 2
            new = await restored.generate("new", prompt_ids=[2, 3], sampling_params={"max_tokens": 2})
            assert old.token_ids == new.token_ids
            assert old.extra_fields["server_epoch"] != new.extra_fields["server_epoch"]
        finally:
            await owner.close()
            if restored is not None:
                await restored.close()

    asyncio.run(run())


def test_resume_cannot_reopen_admission_while_publication_drains_requests():
    async def run():
        owner, model = server()
        try:
            await owner.publish(list(model.named_parameters()), global_step=0)
            generation = asyncio.create_task(
                owner.generate("drain", prompt_ids=[2, 3], sampling_params={"max_tokens": 40})
            )
            while not (await owner.stats())["active"]:
                await asyncio.sleep(0)
            update = asyncio.create_task(owner.publish(list(model.named_parameters()), global_step=1))
            while not (await owner.stats())["paused"]:
                await asyncio.sleep(0)
            with pytest.raises(RuntimeError, match="publication"):
                await owner.resume_generation()
            assert (await generation).extra_fields["weight_version"] == 1
            assert await update == 2
        finally:
            await owner.close()

    asyncio.run(run())
