# SPDX-License-Identifier: Apache-2.0
"""Real-model single-GPU rollout modes, trace replay, and live-weight updates."""

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path

import torch
from vllm_rlt import LLM, CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig, SpeculativeConfig
from vllm_rlt.models.ouro import OuroForCausalLM

from examples.ouro.model import OuroTrainingModel, input_execution_depths


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    prompts = [[2, 3, 5, 7], [11, 13, 17]]
    cases = ["full", "early_exit", "prefix", "async", "graphs", "graphs_async", "speculative", "graphs_speculative"]
    report = {}
    for mode in cases:
        asynchronous = "async" in mode
        speculative = "speculative" in mode
        graphs = "graphs" in mode
        cache = CacheConfig(num_blocks=128, block_size=2, enable_prefix_caching=mode == "prefix")
        scheduler = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=32)
        execution = ExecutionConfig(
            cuda_graphs=graphs, async_scheduling=asynchronous, cuda_graph_max_batch_size=8, cuda_graph_max_graphs=8
        )
        exit_config = ExitConfig("ouro_delayed" if asynchronous else "ouro")
        spec_config = SpeculativeConfig(3) if speculative else None
        params = SamplingParams(
            max_tokens=8,
            temperature=1.0,
            logprobs=0,
            seed=123,
            ignore_eos=True,
            exit_threshold=0.0 if mode == "early_exit" or asynchronous else 1.0,
        )
        native = OuroForCausalLM.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
        llm = LLM(
            native,
            cache_config=cache,
            scheduler_config=scheduler,
            attention_backend="triton",
            execution_config=execution,
            exit_config=exit_config,
            speculative_config=spec_config,
        )
        pointers = [p.data_ptr() for p in native.parameters()]
        outputs = llm.generate(prompts, params)
        train = OuroTrainingModel(native).eval()
        errors = []
        for output in outputs:
            ids = torch.tensor([output.prompt_token_ids + output.token_ids], device="cuda")
            depths = torch.tensor(
                [input_execution_depths(len(output.prompt_token_ids), output.exit_depths, max_loops=4)], device="cuda"
            )
            logits = train(ids, execution_depths=depths).logits[0]
            targets = torch.tensor(output.token_ids, device="cuda")[:, None]
            rows = logits[len(output.prompt_token_ids) - 1 : -1].float().log_softmax(-1)
            selected = rows.gather(-1, targets).squeeze(-1)
            errors.extend((selected - torch.tensor(output.log_probs, device="cuda")).tolist())
        llm.pause_generation()
        llm.start_weight_update(1)
        # Stream individual physical tensors to avoid a second full GPU snapshot.
        for name, tensor in native.named_parameters():
            value = tensor * 0.95 if name == "lm_head.weight" else tensor
            llm.update_weights([(name, value)])
        llm.finish_weight_update()
        llm.resume_generation()
        updated = llm.generate(prompts, params)
        assert all(output.weight_version == 1 for output in updated)
        assert pointers == [p.data_ptr() for p in native.parameters()]
        clone = OuroForCausalLM(native.config).to(device="cuda", dtype=torch.bfloat16)
        clone.load_state_dict(native.state_dict(), strict=True)
        fresh = LLM(
            clone,
            cache_config=cache,
            scheduler_config=scheduler,
            attention_backend="triton",
            execution_config=execution,
            exit_config=exit_config,
            speculative_config=spec_config,
        )
        expected = fresh.generate(prompts, params)
        equal = all(
            a.token_ids == b.token_ids and a.exit_depths == b.exit_depths and a.log_probs == b.log_probs
            for a, b in zip(updated, expected, strict=True)
        )
        assert equal, mode
        bank = llm.engine.speculative_runner.graphs if speculative else llm.engine.model_runner.graphs
        replays = bank.replays if graphs else 0
        if graphs:
            assert replays > 0
        # A real stop-ID response must retain exactly one trace and likelihood.
        stopped = llm.generate([prompts[0]], replace(params, stop_token_ids=(updated[0].token_ids[0],)))[0]
        assert len(stopped.token_ids) == len(stopped.exit_depths) == len(stopped.log_probs) == 1
        ratio = torch.tensor(errors).exp()
        report[mode] = {
            "exit_depths": [out.exit_depths for out in outputs],
            "weight_sync_fresh_equal": equal,
            "graph_replays": replays,
            "stop_alignment": True,
            "max_abs_selected_logprob": max(abs(value) for value in errors),
            "ratio_min": ratio.min().item(),
            "ratio_max": ratio.max().item(),
        }
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(mode, report[mode], flush=True)
        fresh.close()
        llm.close()
        del train, native, clone, fresh, llm, logits, rows
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
