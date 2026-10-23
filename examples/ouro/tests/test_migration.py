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
"""Current-contract adapter checks with a modeled versioned publication endpoint.

The real #83 engine has no publication API; these tests do not qualify native
weight updates. Real unversioned engine generation is covered separately.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from examples.ouro.tests.test_replay import native_model


class VersionedEndpoint:
    """Model only the proposed public transaction; deliberately no pause API."""

    def __init__(self):
        self.version = 0
        self.fail = False
        self.closed = False

    def get_weight_version(self):
        return self.version

    def start_weight_update(self, version=None):
        self.attempt = version

    def update_weights(self, weights):
        list(weights)
        if self.fail:
            raise RuntimeError("injected publication failure")

    def finish_weight_update(self):
        self.version = self.attempt

    def generate(self, prompts, params):
        return [
            SimpleNamespace(
                request_id=str(i),
                prompt_token_ids=prompt,
                token_ids=[5, 7],
                exit_depths=[4, 2],
                finished=True,
                finish_reason="length",
                logprobs=[-1.0, -2.0],
                sampling_params=sample,
                weight_version=self.version,
            )
            for i, (prompt, sample) in enumerate(zip(prompts, params, strict=True))
        ]

    def close(self):
        self.closed = True


def config_pair(tmp_path):
    from verl.workers.config import HFModelConfig, RolloutConfig

    (tmp_path / "config.json").write_text(json.dumps(native_model().config.to_dict()))
    model = HFModelConfig(
        path=str(tmp_path),
        load_tokenizer=False,
        external_lib="examples.ouro.hf",
        override_config={"attn_implementation": "eager"},
    )
    rollout = RolloutConfig(
        name="vllm_rlt",
        mode="async",
        tensor_model_parallel_size=1,
        response_length=4,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        ignore_eos=True,
    )
    return rollout, model


def make_rollout(tmp_path, backend=None):
    from examples.ouro.rollout import OuroRollout

    config, model = config_pair(tmp_path)
    return OuroRollout(
        config, model, None, backend=backend or VersionedEndpoint(), backend_logprobs_mode="raw_logprobs", device="cpu"
    )


def prompts():
    from verl import DataProto

    return DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[2, 3]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
    )


def test_reads_current_scores_and_propagates_probability_contract(tmp_path):
    from examples.ouro.worker import training_batch_from_rollout

    rollout = make_rollout(tmp_path)
    output = rollout.generate_sequences(prompts())
    assert output.meta_info["rlt_engine_contract"] == "rollout_outputs"
    assert output.meta_info["rlt_logprobs_mode"] == "raw_logprobs"
    assert output.batch["rollout_log_probs"].tolist() == [[-1.0, -2.0, 0.0, 0.0]]
    batch = training_batch_from_rollout(
        output,
        advantages=torch.ones(1, 4),
        expected_version=0,
        world_size=1,
        global_batch_size=1,
        global_token_count=2,
    )
    assert batch["rollout_log_probs"].values().tolist() == [-1.0, -2.0]
    output.meta_info["rlt_logprobs_mode"] = "processed_logprobs"
    with pytest.raises(ValueError, match="raw"):
        training_batch_from_rollout(
            output,
            advantages=torch.ones(1, 4),
            expected_version=0,
            world_size=1,
            global_batch_size=1,
            global_token_count=2,
        )
    rollout.close()


def test_local_release_does_not_require_engine_pause_resume(tmp_path):
    rollout = make_rollout(tmp_path)
    asyncio.run(rollout.release())
    with pytest.raises(RuntimeError, match="released"):
        rollout.generate_sequences(prompts())
    asyncio.run(rollout.resume(["weights", "kv_cache"]))
    assert rollout.generate_sequences(prompts()).batch["response_mask"].sum() == 2
    rollout.close()


def test_transaction_failure_blocks_generation_and_resume_until_complete_retry(tmp_path):
    endpoint = VersionedEndpoint()
    rollout = make_rollout(tmp_path, endpoint)
    endpoint.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        asyncio.run(rollout.update_weights([("x", torch.ones(2))], global_steps=0))
    with pytest.raises(RuntimeError, match="publication"):
        rollout.generate_sequences(prompts())
    with pytest.raises(RuntimeError, match="publication"):
        asyncio.run(rollout.resume(["weights"]))
    endpoint.fail = False
    assert asyncio.run(rollout.update_weights([("x", torch.ones(2))], global_steps=0)) == 2
    assert rollout.generate_sequences(prompts()).batch["weight_version"].tolist() == [2]
    rollout.close()


def test_closed_rollout_rejects_update_without_consuming_weights(tmp_path):
    rollout = make_rollout(tmp_path)
    rollout.close()

    def weights():
        raise AssertionError("closed update consumed weight iterator")
        yield

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(rollout.update_weights(weights(), global_steps=0))


def test_unversioned_current_backend_rejects_training_at_initialization(tmp_path):
    config, model = config_pair(tmp_path)
    from examples.ouro.rollout import OuroRollout

    with pytest.raises(NotImplementedError, match="publication"):
        OuroRollout(config, model, None, backend=SimpleNamespace(), device="cpu")


def test_recipe_preflights_publication_before_cuda_rank_or_model_setup(tmp_path):
    import os
    import subprocess
    import sys

    from vllm_rlt import LLM

    if hasattr(LLM, "start_weight_update"):
        pytest.skip("rollout-only engine capability gate is tested against #83")
    env = {key: value for key, value in os.environ.items() if key not in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.ouro.train",
            "--model",
            "/nonexistent/ouro",
            "--output",
            str(tmp_path / "no-run.json"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode != 0
    assert "publication" in result.stderr
    assert "KeyError" not in result.stderr
    assert not (tmp_path / "no-run.json").exists()


def test_current_worker_rejects_missing_engine_probability_provenance(tmp_path):
    from examples.ouro.worker import training_batch_from_rollout

    rollout = make_rollout(tmp_path)
    output = rollout.generate_sequences(prompts())
    del output.meta_info["rlt_logprobs_mode"]
    with pytest.raises(ValueError, match="raw"):
        training_batch_from_rollout(
            output,
            advantages=torch.ones(1, 4),
            expected_version=0,
            world_size=1,
            global_batch_size=1,
            global_token_count=2,
        )
    rollout.close()


def test_restored_counters_require_full_republication_and_same_contract(tmp_path):
    rollout = make_rollout(tmp_path)
    asyncio.run(rollout.update_weights([("x", torch.ones(2))], global_steps=0))
    state = rollout.state_dict()
    restored = make_rollout(tmp_path)
    restored.load_state_dict(state)
    with pytest.raises(RuntimeError, match="publication"):
        restored.generate_sequences(prompts())
    assert asyncio.run(restored.update_weights([("x", torch.ones(2))], global_steps=0)) == 2
    assert restored.generate_sequences(prompts()).batch["weight_version"].tolist() == [2]
    bad = {**state, "engine_contract": "prototype"}
    with pytest.raises(ValueError, match="contract"):
        restored.load_state_dict(bad)
    rollout.close()
    restored.close()


@pytest.mark.parametrize("mode", [None, "processed_logprobs"])
def test_injected_backend_requires_explicit_raw_probability_mode(tmp_path, mode):
    from examples.ouro.rollout import OuroRollout

    config, model = config_pair(tmp_path)
    with pytest.raises(ValueError, match="backend_logprobs_mode"):
        OuroRollout(
            config,
            model,
            None,
            backend=VersionedEndpoint(),
            backend_logprobs_mode=mode,
            device="cpu",
        )


def test_rejected_restore_keeps_sampler_cursor_and_publisher_unchanged(tmp_path):
    rollout = make_rollout(tmp_path)
    before = rollout.state_dict()
    with pytest.raises(ValueError):
        rollout.load_state_dict({**before, "request_counter": 123, "publisher": {}})
    assert rollout.state_dict() == before
    assert rollout.generate_sequences(prompts()).batch["sampling_seed"].tolist() == [rollout.config.seed]
    rollout.close()
