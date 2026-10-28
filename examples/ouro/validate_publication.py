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
"""Current #78 publication qualification through public engine operations.

This is separate from the historical guarded drivers. Select a real checkpoint
and a mode explicitly. Public profiling captures can establish graph execution;
enabling a flag alone is not a replay measurement.
"""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import torch
import vllm_rlt
from vllm_rlt import LLM, CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig, SpeculativeConfig
from vllm_rlt.models.ouro import OuroForCausalLM
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.engine import PDEngine

from examples.ouro.rollout import OuroRollout
from verl import DataProto
from verl.workers.config import HFModelConfig, RolloutConfig


def source_manifest(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }


def prompts():
    ids = torch.tensor([[3, 4, 5, 6], [3, 4, 5, 7]])
    return DataProto.from_dict(tensors={"input_ids": ids, "attention_mask": torch.ones_like(ids)})


def compare(actual, expected):
    for key in ("responses", "response_mask", "execution_depths", "sampling_seed"):
        torch.testing.assert_close(actual.batch[key], expected.batch[key], atol=0, rtol=0)
    delta = (actual.batch["rollout_log_probs"] - expected.batch["rollout_log_probs"]).abs()
    torch.testing.assert_close(
        actual.batch["rollout_log_probs"], expected.batch["rollout_log_probs"], atol=2e-5, rtol=2e-5
    )
    return float(delta.max())


def run_case(model_path, mode):
    pd = mode.startswith("pd_")
    graphs = "graphs" in mode
    asynchronous = "async" in mode or pd
    speculative = "speculative" in mode
    cache = {"num_blocks": 128, "block_size": 16, "enable_prefix_caching": mode == "prefix" or pd}
    scheduler = {"max_num_seqs": 1 if pd else 2, "max_num_batched_tokens": 64}
    execution = {
        "cuda_graphs": graphs,
        "async_scheduling": asynchronous,
        "cuda_graph_max_batch_size": 8,
        "cuda_graph_max_graphs": 8,
    }
    exit_config = {"mode": "ouro_delayed" if asynchronous else "ouro"}
    options = {
        "contract": "publication_outputs",
        "cache": cache,
        "scheduler": scheduler,
        "execution": execution,
        "exit": exit_config,
        "bucket_bytes": 32 << 20,
        "sampling": {"exit_threshold": 0.0 if mode == "early_exit" else 1.0},
    }
    if speculative:
        options["speculative"] = {"num_speculative_tokens": 3}
    config = RolloutConfig(
        name="vllm_rlt",
        mode="async",
        tensor_model_parallel_size=1,
        response_length=8,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        ignore_eos=True,
        seed=123,
        engine_kwargs={"vllm_rlt": options},
    )
    model_config = HFModelConfig(
        path=model_path,
        load_tokenizer=False,
        external_lib="examples.ouro.hf",
        override_config={"attn_implementation": "eager"},
    )

    def make_rollout():
        if pd:
            endpoint = PDEngine(
                model_path,
                dtype="bfloat16",
                attention_backend="triton",
                pd_config=PDConfig(startup_timeout=180, request_timeout=120, shutdown_timeout=30),
                prefill_cache_config=CacheConfig(**cache),
                decode_cache_config=CacheConfig(**cache),
                prefill_scheduler_config=SchedulerConfig(**scheduler),
                decode_scheduler_config=SchedulerConfig(**scheduler),
                execution_config=ExecutionConfig(**execution),
                exit_config=ExitConfig(**exit_config),
            )
        else:
            endpoint = LLM(
                model_path,
                device="cuda:0",
                dtype=torch.bfloat16,
                attention_backend="triton",
                cache_config=CacheConfig(**cache),
                scheduler_config=SchedulerConfig(**scheduler),
                execution_config=ExecutionConfig(**execution),
                exit_config=ExitConfig(**exit_config),
                speculative_config=SpeculativeConfig(3) if speculative else None,
            )
        return OuroRollout(
            config,
            model_config,
            None,
            backend=endpoint,
            backend_logprobs_mode="raw_logprobs",
            device="cuda:0",
        )

    source = OuroForCausalLM.from_pretrained(model_path, device="cpu", dtype=torch.bfloat16)
    with torch.no_grad():
        source.lm_head.weight.mul_(0.98)
        source.model.layers[0].self_attn.q_proj.weight.mul_(0.98)
    rollout = make_rollout()
    fresh = None
    try:
        rollout.generate_sequences(prompts())
        started = time.monotonic()
        assert asyncio.run(rollout.update_weights(source.named_parameters(), global_steps=0)) == 1
        publication_seconds = time.monotonic() - started
        state = rollout.state_dict()
        actual = rollout.generate_sequences(prompts())
        # Trigger a real native duplicate-name rejection, then require full retry.
        first = next(source.named_parameters())
        try:
            asyncio.run(rollout.update_weights([first, first], global_steps=1))
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("duplicate publication unexpectedly committed")
        try:
            rollout.generate_sequences(prompts())
        except RuntimeError:
            pass
        else:
            raise AssertionError("failed publication did not fence generation")
        assert asyncio.run(rollout.update_weights(source.named_parameters(), global_steps=1)) == 3
        # Close old owners before reconstructing P/D on the same physical devices.
        endpoint = rollout.backend
        rollout.close()
        cleanup = endpoint.stats() if pd else endpoint.engine.stats()
        metrics = endpoint.worker_metrics if pd else None
        fresh = make_rollout()
        fresh.load_state_dict(state)
        assert asyncio.run(fresh.update_weights(source.named_parameters(), global_steps=0)) == 2
        expected = fresh.generate_sequences(prompts())
        error = compare(actual, expected)
        result = {
            "mode": mode,
            "publication_version": 1,
            "recovery_version": 3,
            "fresh_version": 2,
            "publication_seconds": publication_seconds,
            "selected_score_max_error": error,
            "token_ids": actual.batch["responses"].tolist(),
            "exit_depths": actual.batch["execution_depths"].tolist(),
            "cleanup": cleanup,
            "worker_metrics": metrics,
            "graphs_enabled": graphs,
            "graph_replay_measured": False,
            "pd_transport_requested": pd,
        }
        if pd:
            assert metrics and all(value["used_blocks"] == 0 for value in metrics.values())
        else:
            assert cleanup["used_blocks"] == 0
        return result
    finally:
        rollout.close()
        if fresh is not None:
            fresh.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "full",
            "early_exit",
            "prefix",
            "async",
            "graphs",
            "graphs_async",
            "speculative",
            "graphs_speculative",
            "pd_eager",
            "pd_graphs",
        ),
        required=True,
    )
    args = parser.parse_args()
    result = run_case(args.model, args.mode)
    result["engine_source"] = source_manifest(Path(vllm_rlt.__file__).resolve().parent)
    result["torch"] = torch.__version__
    result["contract"] = "publication_outputs"
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "engine_source"}, indent=2))


if __name__ == "__main__":
    main()
