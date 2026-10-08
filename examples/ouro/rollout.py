# SPDX-License-Identifier: Apache-2.0
"""Native vllm-rlt implementation of verl's in-process BaseRollout contract.

Training iterations are synchronous. The inherited config uses mode=async as
required by this verl revision; this adapter is invoked directly by the Ouro
torchrun recipe, rather than a Ray RolloutReplica/AgentLoop server.
"""

import copy
from dataclasses import asdict

import numpy as np
import torch
from vllm_rlt import LLM, CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig, SpeculativeConfig

from examples.ouro.model import input_execution_depths
from verl import DataProto
from verl.workers.rollout.base import _ROLLOUT_REGISTRY, BaseRollout


class OuroRollout(BaseRollout):
    def __init__(self, config, model_config, device_mesh, *, backend=None, device="cuda", attention_backend="triton"):
        super().__init__(config, model_config, device_mesh)
        from examples.ouro.publication import OuroWeightPublisher

        options = dict(self.config.engine_kwargs.get("vllm_rlt", {}))
        self._released = False
        self._closed = False
        self._request_counter = 0
        self.backend = backend or LLM(
            self.model_config.local_path,
            device=device,
            dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
            attention_backend=attention_backend,
            cache_config=CacheConfig(**options.get("cache", {"num_blocks": 256})),
            scheduler_config=SchedulerConfig(**options.get("scheduler", {"max_num_seqs": self.config.max_num_seqs})),
            exit_config=ExitConfig(**options.get("exit", {})),
            execution_config=ExecutionConfig(**options.get("execution", {})),
            speculative_config=SpeculativeConfig(**options["speculative"]) if options.get("speculative") else None,
        )
        self.publisher = OuroWeightPublisher(
            self.backend, bucket_bytes=options.get("bucket_bytes", 64 << 20), use_bf16=str(device).startswith("cuda")
        )

    async def resume(self, tags):
        if self._closed:
            raise RuntimeError("rollout is closed")
        if any(tag not in ("weights", "kv_cache") for tag in tags):
            raise ValueError("unknown rollout resume tag")
        if self._released:
            self.backend.resume_generation()
        self._released = False

    async def release(self):
        # Native sleep/wake and memory switching are a separate engine extension.
        # Release here closes admission between synchronous phases.
        if not self._released:
            self.backend.pause_generation()
        self._released = True

    async def update_weights(self, weights, wire_format="named_tensors", **kwargs):
        if wire_format != "named_tensors":
            raise ValueError("Ouro rollout supports wire_format='named_tensors' full updates only")
        if "global_steps" not in kwargs:
            raise ValueError("global_steps is required for versioned publication")
        self.backend.pause_generation()
        version = self.publisher.publish(weights, global_step=kwargs["global_steps"])
        if not self._released:
            self.backend.resume_generation()
        return version

    def generate_sequences(self, prompts):
        if self._closed or self._released:
            raise RuntimeError("rollout is closed or released")
        ids, mask = prompts.batch["input_ids"], prompts.batch["attention_mask"]
        if ids.ndim != 2 or ids.shape != mask.shape or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("prompt IDs and binary mask must have matching two-dimensional shapes")
        prompt_lists = []
        for row, valid in zip(ids, mask.bool(), strict=True):
            indices = valid.nonzero().flatten()
            if not indices.numel() or int(indices[-1] - indices[0] + 1) != indices.numel():
                raise ValueError("each prompt must contain a nonempty contiguous sequence")
            prompt_lists.append(row[valid].cpu().tolist())
        options = dict(self.config.engine_kwargs.get("vllm_rlt", {}))
        common = dict(
            max_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            logprobs=0,
            logprobs_mode="raw",
            ignore_eos=self.config.ignore_eos,
            **options.get("sampling", {}),
        )
        samples = [
            SamplingParams(**common, seed=(self.config.seed + self._request_counter + i) % (2**63))
            for i in range(len(prompt_lists))
        ]
        outputs = self.backend.generate(prompt_lists, samples)
        self._request_counter += len(prompt_lists)
        if len(outputs) != len(prompt_lists):
            raise RuntimeError("engine returned an incomplete rollout batch")
        expected = self.backend.get_weight_version()
        length, width = len(outputs), ids.shape[1]
        response_width = self.config.response_length
        device = ids.device
        response = torch.full(
            (length, response_width), self.model_config.hf_config.pad_token_id or 0, dtype=torch.long, device=device
        )
        response_mask = torch.zeros_like(response, dtype=torch.bool)
        scores = torch.zeros_like(response, dtype=torch.float32)
        depths = torch.ones((length, width + response_width), dtype=torch.long, device=device)
        positions = torch.zeros_like(depths)
        for index, output in enumerate(outputs):
            if not output.finished or output.finish_reason in ("abort", "error") or output.weight_version != expected:
                raise ValueError("uncommitted, aborted or failed rollout cannot enter the training batch")
            count = len(output.token_ids)
            if count > response_width or output.log_probs is None or len(output.log_probs) != count:
                raise ValueError("response and selected probabilities must match the configured length")
            if len(output.exit_depths) != count or output.prompt_token_ids != prompt_lists[index]:
                raise ValueError("engine trace/prompt metadata is misaligned")
            trace = input_execution_depths(
                len(prompt_lists[index]), output.exit_depths, max_loops=self.model_config.hf_config.total_ut_steps
            )
            response[index, :count] = torch.tensor(output.token_ids, device=device)
            response_mask[index, :count] = True
            scores[index, :count] = torch.tensor(output.log_probs, device=device)
            valid = mask[index].bool()
            depths[index, :width][valid] = torch.tensor(trace[: len(prompt_lists[index])], device=device)
            depths[index, width : width + count] = torch.tensor(trace[len(prompt_lists[index]) :], device=device)
            positions[index, :width] = mask[index].long().cumsum(-1).sub(1).clamp_min(0)
            positions[index, width : width + count] = torch.arange(
                len(prompt_lists[index]), len(prompt_lists[index]) + count, device=device
            )
        full_ids = torch.cat((ids, response), dim=1)
        full_mask = torch.cat((mask.bool(), response_mask), dim=1)
        tensors = {
            "prompts": ids,
            "responses": response,
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "position_ids": positions,
            "execution_depths": depths,
            "response_mask": response_mask,
            "rollout_log_probs": scores,
            "weight_version": torch.tensor([out.weight_version for out in outputs], device=device),
            "sampling_seed": torch.tensor([out.sampling_params.seed for out in outputs], device=device),
        }
        non_tensors = dict(prompts.non_tensor_batch)
        non_tensors.update(
            {
                "rlt_outputs": np.array(outputs, dtype=object),
                "finish_reason": np.array([out.finish_reason for out in outputs], dtype=object),
                "sampling_params": np.array([asdict(out.sampling_params) for out in outputs], dtype=object),
            }
        )
        return DataProto.from_dict(
            tensors=tensors, non_tensors=non_tensors, meta_info={**prompts.meta_info, "rlt_logprobs_mode": "raw"}
        )

    def state_dict(self):
        return {"request_counter": self._request_counter, "publisher": self.publisher.state_dict()}

    def load_state_dict(self, state):
        counter = state["request_counter"]
        if type(counter) is not int or counter < 0:
            raise ValueError("invalid rollout request counter")
        self._request_counter = counter
        self.publisher.load_state_dict(copy.deepcopy(state["publisher"]))

    def close(self):
        if not self._closed:
            self.backend.close()
        self._closed = True


_ROLLOUT_REGISTRY[("vllm_rlt", "async")] = "examples.ouro.rollout.OuroRollout"
