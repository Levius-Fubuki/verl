# SPDX-License-Identifier: Apache-2.0
"""Actual current-engine qualification; no model runners or schedulers are patched."""

import importlib.util
import json
import subprocess
import sys
from dataclasses import fields

import pytest


def current_contract():
    from vllm_rlt import SamplingParams
    from vllm_rlt.request import RequestOutput

    missing = {"logprobs", "sampling_params"} - {field.name for field in fields(RequestOutput)}
    if missing or "logprobs" not in {field.name for field in fields(SamplingParams)}:
        pytest.skip(f"legacy engine lacks #83 rollout fields: {sorted(missing) or ['SamplingParams.logprobs']}")


def validation_api():
    assert importlib.util.find_spec("examples.ouro.validate_current") is not None, (
        "standalone current-engine validation command is missing"
    )
    from examples.ouro.validate_current import run_validation, tiny_native_model

    return run_validation, tiny_native_model


def test_standalone_validation_api_exists():
    current_contract()
    validation_api()


@pytest.fixture(scope="module")
def report():
    current_contract()
    run_validation, _ = validation_api()
    return run_validation()


@pytest.mark.parametrize("name", ["full_depth", "early_exit", "speculative"])
def test_actual_generation_replays_effective_seed_and_target_scores(report, name):
    case = report["cases"][name]
    assert case["seed_replay_validated"] is True
    assert case["target_logprobs_validated"] is True
    assert case["max_abs_target_logprob_error"] < 3e-5
    assert case["outputs"]
    for output in case["outputs"]:
        assert type(output["effective_seed"]) is int
        assert len(output["token_ids"]) == len(output["exit_depths"]) == len(output["logprobs"]) == 8
        assert output["exit_depths"][0] == 4
        assert "weight_version" not in output
    if name == "early_exit":
        assert any(depth < 4 for output in case["outputs"] for depth in output["exit_depths"][1:])
    else:
        assert all(depth == 4 for output in case["outputs"] for depth in output["exit_depths"])


def test_publication_is_explicitly_unavailable(report):
    assert report["generation_validated"] is True
    assert report["publication_validated"] is False
    assert "publication" in report["publication_limitation"]
    assert report["pd_validated"] is False
    assert report["cases"]["full_depth"]["transport"] == "LLM/StageClient subprocess"
    assert report["cases"]["speculative"]["transport"] == "native StageEngine"


def test_real_stage_client_stop_abort_and_cleanup(report):
    assert report["stop_validated"] is True
    assert report["abort_validated"] is True
    assert report["abort_reuse_seed_validated"] is True
    assert report["drained_used_blocks"] == 0


def test_actual_old_consumer_field_fails_and_new_driver_reads_native_scores():
    current_contract()
    _, tiny_native_model = validation_api()
    from vllm_rlt import CacheConfig, SamplingParams
    from vllm_rlt.engine.llm_engine import LLMEngine

    from examples.ouro.backend import TokenBatchDriver, engine_kwargs, require_publication, selected_logprobs

    backend = LLMEngine(tiny_native_model(), cache_config=CacheConfig(128, 2), **engine_kwargs())
    driver = TokenBatchDriver(backend)
    try:
        (output,) = driver.generate([[2, 3]], SamplingParams(max_tokens=2, logprobs=0, seed=9, ignore_eos=True))
        # Actual old consumer failure, rather than an import failure or a fake output.
        with pytest.raises(AttributeError, match="log_probs") as failure:
            _ = output.log_probs
        print(f"old consumer failure on actual #83 RequestOutput: {failure.value}")
        assert len(selected_logprobs(output)) == 2
        with pytest.raises(NotImplementedError, match="publication"):
            require_publication(backend)
    finally:
        driver.close()


def test_standalone_cli_writes_generation_only_report(tmp_path):
    current_contract()
    validation_api()
    destination = tmp_path / "current.json"
    completed = subprocess.run(
        [sys.executable, "-m", "examples.ouro.validate_current", "--output", str(destination)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(destination.read_text())
    assert result["generation_validated"] is True
    assert result["publication_validated"] is False
    assert result["device"] == "cpu"
    assert result["engine_source"]["git_head"] == "45f0c6737dde1f5be858ffc05fcd35b866474459"
    assert result["engine_source"]["git_dirty"] is False
