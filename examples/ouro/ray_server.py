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
"""Standalone Ray token service with a single serialized public engine owner.

Engine calls run on one actor event-loop thread, one step at a time. Yielding
between steps lets abort/control RPCs progress without sharing a pipe or CUDA
owner across threads. Hybrid memory switching and distributed publication are
explicitly outside this first standalone increment.
"""

import asyncio
from dataclasses import asdict
from uuid import uuid4

import ray
import torch
from vllm_rlt import LLM, CacheConfig, SamplingParams, SchedulerConfig

from examples.ouro.backend import require_publication
from examples.ouro.publication import OuroWeightPublisher
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, RolloutReplicaRegistry, TokenOutput


class OuroTokenServer:
    def __init__(self, model_path=None, *, backend=None, device="cuda:0", cache_blocks=256):
        require_publication(LLM if backend is None else backend)
        if backend is None:
            facade = LLM(
                model_path,
                device=device,
                dtype=torch.bfloat16,
                attention_backend="triton",
                cache_config=CacheConfig(cache_blocks, 16),
                scheduler_config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=256),
            )
            backend = facade.engine
        self.backend = backend
        self.publisher = OuroWeightPublisher(backend, use_bf16=device.startswith("cuda") and model_path is not None)
        self.epoch = uuid4().hex
        self._commands = asyncio.Queue()
        self._active = {}
        self._paused = True
        self._failed = False
        self._updating = False
        self._closed = False
        self._idle = asyncio.Event()
        self._idle.set()
        self._publication_lock = asyncio.Lock()
        self._pump = None

    def _ensure_pump(self):
        if self._closed:
            raise RuntimeError("server is closed")
        if self._pump is None:
            self._pump = asyncio.create_task(self._run())

    async def _control(self, operation, *args):
        self._ensure_pump()
        future = asyncio.get_running_loop().create_future()
        self._commands.put_nowait((operation, args, future))
        return await asyncio.shield(future)

    async def ready(self):
        return await self.stats()

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None, **kwargs):
        if image_data is not None or video_data is not None or kwargs:
            raise ValueError("Ouro token server supports text-only requests without extra options")
        if not isinstance(request_id, str) or not request_id or not prompt_ids:
            raise ValueError("nonempty request ID and token prompt required")
        options = dict(sampling_params)
        if options.get("logprobs_mode", "raw") != "raw":
            raise ValueError("only raw selected probability mode is supported")
        options.update(logprobs=0, logprobs_mode="raw")
        params = SamplingParams(**options)
        self._ensure_pump()
        future = asyncio.get_running_loop().create_future()
        generation = f"{self.epoch}-{uuid4().hex}"
        self._commands.put_nowait(("submit", (request_id, generation, list(prompt_ids), params), future))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.cancel()
            await self.abort_request(request_id, generation=generation)
            raise

    async def abort_request(self, request_id, generation=None):
        return await self._control("abort", request_id, generation)

    async def abort_all_requests(self, reject_request=False):
        return await self._control("abort_all")

    async def resume_generation(self):
        return await self._control("resume")

    async def publish(self, weights, *, global_step):
        async with self._publication_lock:
            await self._control("pause")
            await self._idle.wait()
            return await self._control("publish", weights, global_step)

    async def state_dict(self):
        return await self._control("snapshot")

    async def load_state_dict(self, state):
        return await self._control("restore", state)

    async def clear_kv_cache(self):
        return await self._control("clear")

    async def stats(self):
        return await self._control("stats")

    async def close(self):
        if self._closed:
            return
        await self._control("close")
        await self._pump

    async def sleep(self):
        raise NotImplementedError("standalone server does not implement memory switching")

    async def wake_up(self):
        raise NotImplementedError("standalone server does not implement memory switching")

    def _retire(self, request_id, reason):
        item = self._active.pop(request_id, None)
        if item is None:
            return False
        internal, _, _, future = item
        try:
            self.backend.abort_request(internal)
        finally:
            if not future.done():
                future.set_exception(RuntimeError(reason))
            if not self._active:
                self._idle.set()
        return True

    def _command(self, operation, args, future):
        if operation == "submit":
            request_id, internal, prompt, params = args
            if future.cancelled():
                return
            if self._paused or self._failed:
                raise RuntimeError("generation requires successful full publication and open admission")
            if request_id in self._active:
                raise ValueError("duplicate active external request ID")
            if len(self._active) >= 64:
                raise RuntimeError("pending request limit reached")
            try:
                self.backend.add_request(internal, prompt, params)
            except BaseException:
                if self.backend.has_request(internal):
                    self.backend.abort_request(internal)
                raise
            self._active[request_id] = (internal, prompt, params, future)
            self._idle.clear()
            return
        if operation == "abort":
            request_id, generation = args
            item = self._active.get(request_id)
            value = (
                False
                if item is None or generation is not None and item[0] != generation
                else self._retire(request_id, "request aborted")
            )
        elif operation in ("abort_all", "close"):
            self._paused = True
            for request_id in list(self._active):
                self._retire(request_id, "request aborted by server control")
            if operation == "close":
                self.backend.close()
                self._closed = True
            value = None
        elif operation == "pause":
            self._paused = True
            self._updating = True
            value = None
        elif operation == "publish":
            weights, step = args
            self._failed = True
            value = self.publisher.publish(weights, global_step=step)
            self._failed = False
            self._updating = False
            self._paused = False
        elif operation == "resume":
            if self._failed or self._updating or self.publisher.global_step is None:
                raise RuntimeError("generation requires successful full publication")
            self._paused = False
            value = None
        elif operation == "snapshot":
            if self._active:
                raise RuntimeError("checkpoint counters require idle owner")
            value = {"engine_contract": "publication_outputs", "publisher": self.publisher.state_dict()}
        elif operation == "restore":
            (state,) = args
            if self._active or self.publisher.global_step is not None:
                raise RuntimeError("restore requires a fresh idle owner")
            if (
                not isinstance(state, dict)
                or set(state) != {"engine_contract", "publisher"}
                or state["engine_contract"] != "publication_outputs"
            ):
                raise ValueError("server checkpoint profile mismatch")
            self.publisher.load_state_dict(state["publisher"])
            self._paused = True
            self._failed = True
            value = None
        elif operation == "clear":
            if self._active:
                raise RuntimeError("cache reset requires idle owner")
            value = self.backend.reset_prefix_cache()
        elif operation == "stats":
            value = {
                "active": len(self._active),
                "generations": {k: v[0] for k, v in self._active.items()},
                "epoch": self.epoch,
                "paused": self._paused,
                "failed": self._failed,
                "updating": self._updating,
                "publisher": self.publisher.state_dict(),
                "engine": self.backend.stats(),
            }
        else:
            raise ValueError("unknown owner operation")
        if not future.done():
            future.set_result(value)

    def _deliver(self, output):
        found = [(key, value) for key, value in self._active.items() if value[0] == output.request_id]
        if len(found) != 1:
            raise RuntimeError("foreign or stale engine output")
        key, (_, prompt, _, future) = found[0]
        if not output.finished:
            return
        if output.prompt_token_ids != prompt or output.weight_version != self.publisher.committed_version:
            raise RuntimeError("output prompt/version mismatch")
        if (
            output.log_probs is None
            or len(output.log_probs) != len(output.token_ids)
            or len(output.exit_depths) != len(output.token_ids)
        ):
            raise RuntimeError("misaligned selected scores/depths")
        params = output.sampling_params
        if params is None or params.logprobs_mode != "raw" or params.seed is None:
            raise RuntimeError("missing actual raw probability/seed provenance")
        result = TokenOutput(
            token_ids=output.token_ids,
            log_probs=output.log_probs,
            stop_reason="aborted" if output.finish_reason == "abort" else "completed",
            extra_fields={
                "exit_depths": output.exit_depths,
                "sampling_params": asdict(params),
                "weight_version": output.weight_version,
                "global_steps": self.publisher.global_step,
                "logprobs_mode": "raw_logprobs",
                "engine_contract": "publication_outputs",
                "request_generation": output.request_id,
                "server_epoch": self.epoch,
            },
        )
        self._active.pop(key)
        if not self._active:
            self._idle.set()
        if not future.done():
            future.set_result(result)

    async def _run(self):
        while not self._closed:
            first = None
            if not self._active and self._commands.empty():
                first = await self._commands.get()
            while first is not None or not self._commands.empty():
                operation, args, future = first if first is not None else self._commands.get_nowait()
                first = None
                try:
                    self._command(operation, args, future)
                except Exception as exc:
                    if not future.done():
                        future.set_exception(exc)
                if self._closed:
                    break
            if self._active and not self._closed:
                try:
                    for output in self.backend.step():
                        self._deliver(output)
                except Exception as exc:
                    self._paused = self._failed = True
                    for request_id in list(self._active):
                        self._retire(request_id, f"engine execution failed: {exc}")
            await asyncio.sleep(0)
        while not self._commands.empty():
            _, _, future = self._commands.get_nowait()
            if not future.done():
                future.set_exception(RuntimeError("server closed"))


class OuroReplica(RolloutReplica):
    """One standalone token server per one-GPU replica; no HTTP serving."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.world_size != 1 or self.config.disaggregation.enabled:
            raise NotImplementedError("Ouro standalone replica requires one GPU and no PD")

    async def init_hybrid(self, *args, **kwargs):
        raise NotImplementedError("hybrid memory/execution coordination is not implemented")

    async def init_hybrid_colocated(self, *args, **kwargs):
        raise NotImplementedError("hybrid memory/execution coordination is not implemented")

    async def init_colocated(self, *args, **kwargs):
        raise NotImplementedError("colocated memory switching is not implemented")

    async def init_standalone(self, resource_pool=None):
        self.rollout_mode = RolloutMode.STANDALONE
        self.resource_pool = resource_pool
        await self.launch_servers()

    async def launch_servers(self):
        options = {"num_gpus": 1, "num_cpus": 1}
        if self.resource_pool is not None:
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

            bundles = [(pg, i) for pg in self.resource_pool.get_placement_groups() for i in range(pg.bundle_count)]
            pg, index = bundles[self.replica_rank]
            options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=index, placement_group_capture_child_tasks=True
            )
        server = ray.remote(OuroTokenServer).options(**options).remote(model_path=self.model_config.local_path)
        self.servers = [server]
        self._server_handle = server
        try:
            await server.ready.remote()
        except BaseException:
            ray.kill(server)
            self.servers = []
            self._server_handle = None
            raise

    async def close(self):
        for server in self.servers:
            try:
                await server.close.remote()
            finally:
                ray.kill(server)
        self.servers = []
        self._server_handle = None


RolloutReplicaRegistry.register("vllm_rlt", lambda: OuroReplica)
