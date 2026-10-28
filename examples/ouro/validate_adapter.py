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
"""Exercise serving features and PD through the verl BaseRollout adapter."""

import argparse
import asyncio
import gc
import json
from pathlib import Path

import torch
from vllm_rlt import LLM, CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.engine import PDEngine

from examples.ouro.rollout import OuroRollout
from verl import DataProto
from verl.workers.config import HFModelConfig, RolloutConfig


def prompts():
    ids = torch.tensor([[3, 4, 5, 6], [3, 4, 5, 7]])
    return DataProto.from_dict(tensors={"input_ids": ids, "attention_mask": torch.ones_like(ids)})


def compare_outputs(left, right):
    for key in ("responses", "response_mask", "execution_depths", "sampling_seed"):
        torch.testing.assert_close(left.batch[key], right.batch[key], atol=0, rtol=0)
    torch.testing.assert_close(left.batch["rollout_log_probs"], right.batch["rollout_log_probs"], atol=2e-5, rtol=2e-5)
    return float((left.batch["rollout_log_probs"] - right.batch["rollout_log_probs"]).abs().max())


class GraphReplayProbe:
    """Validation-only wrapper around an existing captured CUDA graph."""

    def __init__(self, graph):
        self.graph, self.replays = graph, 0

    def replay(self):
        self.replays += 1
        self.graph.replay()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pd", action="store_true")
    args = parser.parse_args()
    from examples.ouro.historical import require_historical_prototype

    require_historical_prototype()
    model_config = HFModelConfig(
        path=args.model,
        load_tokenizer=False,
        external_lib="examples.ouro.hf",
        override_config={"attn_implementation": "eager"},
    )
    report = {}
    cases = (
        ["pd_eager", "pd_graphs"]
        if args.pd
        else ["full", "early_exit", "prefix", "async", "graphs", "graphs_async", "speculative", "graphs_speculative"]
    )
    for mode in cases:
        graphs = "graphs" in mode
        asynchronous = args.pd or "async" in mode
        cache = {"num_blocks": 128, "block_size": 2, "enable_prefix_caching": mode == "prefix" or args.pd}
        # One slot makes PD/local BF16 prefill batching identical for parity.
        scheduler = {"max_num_seqs": 1 if args.pd else 2, "max_num_batched_tokens": 64}
        execution = {
            "cuda_graphs": graphs,
            "async_scheduling": asynchronous,
            "cuda_graph_max_batch_size": 8,
            "cuda_graph_max_graphs": 8,
        }
        exit_config = {"mode": "ouro_delayed" if asynchronous else "ouro"}
        options = {
            "contract": "prototype",
            "cache": cache,
            "scheduler": scheduler,
            "execution": execution,
            "exit": exit_config,
            "sampling": {"exit_threshold": 0.0 if mode == "early_exit" or asynchronous else 1.0},
            "bucket_bytes": 32 << 20,
        }
        if "speculative" in mode:
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
        backend = None
        pd_baseline_metrics = None
        if args.pd:

            def make_pd(cache=cache, scheduler=scheduler, execution=execution, exit_config=exit_config):
                return PDEngine(
                    args.model,
                    dtype="bfloat16",
                    attention_backend="triton",
                    pd_config=PDConfig(startup_timeout=180.0, request_timeout=120.0, shutdown_timeout=30.0),
                    prefill_cache_config=CacheConfig(**cache),
                    decode_cache_config=CacheConfig(**cache),
                    prefill_scheduler_config=SchedulerConfig(**scheduler),
                    decode_scheduler_config=SchedulerConfig(**scheduler),
                    execution_config=ExecutionConfig(**execution),
                    exit_config=ExitConfig(**exit_config),
                )

            if graphs:
                # PD reports counters only at close. Measure an identical
                # warmup-only owner pair first, then require additional replay
                # in the warmup+publication pair without adding a private API.
                baseline_pd = make_pd()
                baseline = OuroRollout(
                    config,
                    model_config,
                    None,
                    backend=LLM(baseline_pd),
                    backend_logprobs_mode="raw_logprobs",
                    device="cuda:0",
                )
                try:
                    baseline.generate_sequences(prompts())
                finally:
                    baseline.close()
                pd_baseline_metrics = baseline_pd.worker_metrics
            backend = LLM(make_pd())
        rollout = OuroRollout(
            config,
            model_config,
            None,
            backend=backend,
            backend_logprobs_mode="raw_logprobs" if backend is not None else None,
            device="cuda:0",
        )
        fresh = None
        try:
            # Warm real cache/graph state BEFORE publication, then compare a
            # freshly constructed engine with identical sampler counter/state.
            rollout.generate_sequences(prompts())
            graph_before = None
            probes = {}
            if graphs and not args.pd:
                runner = (
                    rollout.backend.engine.speculative_runner
                    if "speculative" in mode
                    else rollout.backend.engine.model_runner
                )
                graph_before = (runner.graphs.replays, runner.graphs.captures)
                for key, entry in runner.graphs.entries.items():
                    probes[key] = GraphReplayProbe(entry.graph)
                    entry.graph = probes[key]
            from vllm_rlt.models.ouro import OuroForCausalLM as NativeModel

            if args.pd:
                # CPU weights are sufficient for bucketed PD publication.
                source = NativeModel.from_pretrained(args.model, device="cpu", dtype=torch.bfloat16)
            else:
                source = rollout.backend.engine.model
            pointers = [p.data_ptr() for p in source.parameters()]
            with torch.no_grad():
                source.lm_head.weight.mul_(0.98)
                source.model.layers[0].self_attn.q_proj.weight.mul_(0.98)
            asyncio.run(rollout.update_weights(source.named_parameters(), global_steps=0))
            if not args.pd:
                assert pointers == [p.data_ptr() for p in source.parameters()]
            generation_state = rollout.state_dict()
            output = rollout.generate_sequences(prompts())
            assert output.batch["weight_version"].tolist() == [1, 1]
            # Fresh local engine starts with the same sampler request counter.
            fresh = OuroRollout(config, model_config, None, device="cuda:0")
            fresh.load_state_dict(generation_state)
            asyncio.run(fresh.update_weights(source.named_parameters(), global_steps=0))
            expected = fresh.generate_sequences(prompts())
            error = compare_outputs(output, expected)
            row = {
                "weight_version": 1,
                "max_abs_selected_logprob": error,
                "exit_depths": [out.exit_depths for out in output.non_tensor_batch["rlt_outputs"]],
                "fresh_engine_equal": True,
                "local_target_addresses_preserved": True if not args.pd else None,
                "scope": "publication/fresh-engine equality for each mode, not eager-vs-mode runtime equivalence",
            }
            if args.pd:
                # Metrics are returned with the normal stop acknowledgement.
                # Do not infer actual NIXL traffic/graph replay from flags alone.
                rollout.close()
                row["worker_metrics"] = rollout.backend.engine.worker_metrics
                row["request_metrics"] = list(rollout.backend.engine.metrics)
                assert len(row["worker_metrics"]) == 2
                assert all(
                    metric["used_blocks"] == 0 and metric["weight_version"] == 1
                    for metric in row["worker_metrics"].values()
                )
                assert sum(metric["bytes_sent"] for metric in row["worker_metrics"].values()) > 0
                if graphs:
                    before = next(metric for name, metric in pd_baseline_metrics.items() if "decode" in name)
                    after = next(metric for name, metric in row["worker_metrics"].items() if "decode" in name)
                    assert before["graph_replays"] > 0
                    assert after["graph_replays"] > before["graph_replays"]
                    assert after["graph_captures"] == before["graph_captures"]
                    row["warmup_only_worker_metrics"] = pd_baseline_metrics
                    row["decode_replays_beyond_warmup"] = after["graph_replays"] - before["graph_replays"]
            elif graphs:
                engine = rollout.backend.engine
                bank = engine.speculative_runner.graphs if "speculative" in mode else engine.model_runner.graphs
                row["graph_replays"] = bank.replays
                row["graph_replays_after_publication"] = bank.replays - graph_before[0]
                assert row["graph_replays_after_publication"] > 0
                # Speculative acceptance can introduce a previously unseen
                # verification shape. Existing captures must survive AND run.
                assert all(bank.entries[key].graph is probe for key, probe in probes.items())
                row["existing_graph_replays_after_publication"] = sum(probe.replays for probe in probes.values())
                row["new_graph_shapes_after_publication"] = bank.captures - graph_before[1]
                assert row["existing_graph_replays_after_publication"] > 0
            report[mode] = row
            Path(args.output).write_text(json.dumps(report, indent=2, default=str) + "\n")
            print(mode, row, flush=True)
        finally:
            rollout.close()
            if fresh:
                fresh.close()
        del source, rollout, fresh
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
