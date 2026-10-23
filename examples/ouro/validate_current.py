# SPDX-License-Identifier: Apache-2.0
"""Generation-only qualification of the public #83 engine on a native tiny model.

Run from the verl checkout with the selected engine on PYTHONPATH::

    python -m examples.ouro.validate_current --output current.json

No server, checkpoint publication, P/D transport, or training update is started.
A real checkpoint may be selected with --model-path. CUDA/P/D qualification is
separate work; CPU results never imply that a device transfer occurred.
"""

import argparse
import copy
import json
import math
import subprocess
from dataclasses import fields, replace
from pathlib import Path

import torch
import vllm_rlt
from vllm_rlt import LLM, CacheConfig, SamplingParams, SchedulerConfig, SpeculativeConfig
from vllm_rlt.engine.llm_engine import LLMEngine, StageClient
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM
from vllm_rlt.request import RequestOutput

from examples.ouro.backend import TokenBatchDriver, engine_kwargs, require_publication, selected_logprobs
from examples.ouro.model import OuroTrainingModel, input_execution_depths


def tiny_native_model(device="cpu"):
    """Construct a reproducible native source without importing test utilities."""
    with torch.random.fork_rng():
        torch.manual_seed(123)
        model = OuroForCausalLM(
            OuroConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                total_ut_steps=4,
                max_position_embeddings=128,
            )
        )
    return model.to(device=device, dtype=torch.float32).eval()


def _engine_source():
    source = Path(vllm_rlt.__file__).resolve()
    root = source.parent.parent
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        head, dirty = None, None
    return {"module": str(source), "git_head": head, "git_dirty": dirty}


def _aligned(output, max_loops):
    scores = selected_logprobs(output)
    count = len(output.token_ids)
    assert count > 0 and len(output.exit_depths) == len(scores) == count
    assert output.exit_depths[0] == max_loops, "first prediction must use full-depth prefill"
    assert all(type(depth) is int and 1 <= depth <= max_loops for depth in output.exit_depths)
    assert all(math.isfinite(score) and score <= 0 for score in scores)
    assert type(output.sampling_params.seed) is int, "effective seed was not returned"


def _same(actual, expected):
    assert actual.token_ids == expected.token_ids, "effective seed did not replay tokens"
    assert actual.exit_depths == expected.exit_depths, "effective seed did not replay exit depths"
    torch.testing.assert_close(
        torch.tensor(selected_logprobs(actual)), torch.tensor(selected_logprobs(expected)), atol=2e-6, rtol=0
    )


@torch.no_grad()
def _target_error(replay, prompt, output):
    tokens = prompt + output.token_ids
    device = next(replay.parameters()).device
    depths = input_execution_depths(len(prompt), output.exit_depths, max_loops=replay.config.total_ut_steps)
    logits = replay(
        torch.tensor([tokens], device=device), execution_depths=torch.tensor([depths], device=device)
    ).logits[0]
    rows = logits[len(prompt) - 1 : -1].float().log_softmax(-1)
    selected = rows.gather(1, torch.tensor(output.token_ids, device=device)[:, None]).squeeze(1)
    observed = torch.tensor(selected_logprobs(output), device=device)
    torch.testing.assert_close(observed, selected, atol=3e-5, rtol=0)
    return float((observed - selected).abs().max())


def _case(backend, replay, prompts, params, transport):
    driver = TokenBatchDriver(backend)
    try:
        first = driver.generate(prompts, params)
        second = driver.generate(prompts, [output.sampling_params for output in first])
        errors, rows = [], []
        for prompt, actual, expected in zip(prompts, second, first, strict=True):
            _aligned(expected, replay.config.total_ut_steps)
            _same(actual, expected)
            errors.append(_target_error(replay, prompt, expected))
            assert expected.finished and expected.finish_reason == "length"
            rows.append(
                {
                    "prompt_token_ids": prompt,
                    "token_ids": expected.token_ids,
                    "exit_depths": expected.exit_depths,
                    "logprobs": selected_logprobs(expected),
                    "effective_seed": expected.sampling_params.seed,
                }
            )
        return {
            "transport": transport,
            "seed_replay_validated": True,
            "target_logprobs_validated": True,
            "max_abs_target_logprob_error": max(errors),
            "outputs": rows,
        }
    finally:
        driver.close()


def _drain(client, limit=1000):
    final = None
    for _ in range(limit):
        if not client.has_unfinished_requests():
            assert final is not None, "request completed without a final output"
            return final
        for output in client.step():
            if output.finished:
                final = output
    raise RuntimeError("validation request exceeded 1000 engine steps")


def _lifecycle(source, options, device):
    client = StageClient(source, device=device, dtype=torch.float32, **options)
    driver = TokenBatchDriver(client)
    try:
        params = SamplingParams(max_tokens=8, temperature=0.8, top_k=7, seed=947, logprobs=0, ignore_eos=True)
        (baseline,) = driver.generate([[2, 3]], params)
        _aligned(baseline, source.config.total_ut_steps)
        stop = replace(params, stop_token_ids=[baseline.token_ids[0]])
        (stopped,) = driver.generate([[2, 3]], stop)
        assert stopped.finish_reason == "stop" and stopped.token_ids == baseline.token_ids[:1]
        assert stopped.exit_depths == baseline.exit_depths[:1]
        assert selected_logprobs(stopped) == selected_logprobs(baseline)[:1]
        client.add_request("abort-and-reuse", [2, 3], params)
        delivered = None
        for _ in range(1000):
            for output in client.step():
                delivered = output
            if delivered is not None and len(delivered.token_ids) >= 2:
                break
        else:
            raise RuntimeError("abort validation did not reach two delivered tokens")
        aborted = client.abort_request("abort-and-reuse")
        _aligned(aborted, source.config.total_ut_steps)
        assert aborted.finished and aborted.finish_reason == "abort"
        count = len(aborted.token_ids)
        assert 2 <= count < 8
        assert aborted.token_ids == baseline.token_ids[:count]
        assert aborted.exit_depths == baseline.exit_depths[:count]
        assert selected_logprobs(aborted) == selected_logprobs(baseline)[:count]
        assert not client.has_request("abort-and-reuse") and not client.has_unfinished_requests()
        client.add_request("abort-and-reuse", [2, 3], params)
        _same(_drain(client), baseline)
        assert not client.has_unfinished_requests()
        used = client.stats()["used_blocks"]
        assert used == 0, "stop or abort leaked KV blocks"
        return {
            "stop_validated": True,
            "abort_validated": True,
            "abort_reuse_seed_validated": True,
            "drained_used_blocks": used,
        }
    finally:
        driver.close()


def run_validation(model_path=None, device="cpu"):
    """Validate generated traces and target probabilities, without inventing versions."""
    if "logprobs" not in {field.name for field in fields(RequestOutput)}:
        raise NotImplementedError("current validation requires #83 RequestOutput.logprobs; selected engine is legacy")
    source = (
        tiny_native_model(device)
        if model_path is None
        else OuroForCausalLM.from_pretrained(model_path, device=device, dtype=torch.float32).eval()
    )
    replay = OuroTrainingModel(copy.deepcopy(source)).eval()
    options = {
        "cache_config": CacheConfig(256, 2),
        "scheduler_config": SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=16),
        **engine_kwargs(),
    }
    prompts = [[2, 3], [7, 8, 9]]
    common = dict(
        max_tokens=8, temperature=0.8, top_k=7, top_p=0.8, seed=None, min_loops=1, logprobs=0, ignore_eos=True
    )
    cases = {}
    publication_limitation = "Weight publication was not exercised by generation qualification."
    for name, threshold in (("full_depth", 1.0), ("early_exit", 0.0)):
        backend = LLM(source, device=device, dtype=torch.float32, **options)
        try:
            # Record the observed capability failure without attaching output versions.
            try:
                require_publication(backend)
            except NotImplementedError as exc:
                publication_limitation = str(exc)
            cases[name] = _case(
                backend,
                replay,
                prompts,
                SamplingParams(exit_threshold=threshold, **common),
                "LLM/StageClient subprocess",
            )
        finally:
            backend.close()
    if source.config.total_ut_steps < 2:
        raise ValueError("speculative validation requires at least two model recurrence depths")
    speculative = SpeculativeConfig(
        3, draft_loops=min(2, source.config.total_ut_steps - 1), target_loops=source.config.total_ut_steps
    )
    cases["speculative"] = _case(
        LLMEngine(source, speculative_config=speculative, **options),
        replay,
        prompts,
        SamplingParams(exit_threshold=1.0, **common),
        "native StageEngine",
    )
    return {
        "device": str(device),
        "model": str(model_path) if model_path else "native tiny Ouro (seed 123)",
        "engine_source": _engine_source(),
        "logprobs_mode": "raw_logprobs",
        "generation_validated": True,
        "publication_validated": False,
        "publication_limitation": publication_limitation,
        "pd_validated": False,
        "pd_limitation": "Actual CUDA P/D transfer remains unvalidated.",
        "cases": cases,
        **_lifecycle(source, options, device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_validation(args.model_path, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validated current-engine generation; publication_validated=False; report: {args.output}")


if __name__ == "__main__":
    main()
