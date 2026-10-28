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
"""Current #78 is an explicit consumer contract, separate from old evidence."""

from types import SimpleNamespace

import pytest

from examples.ouro.backend import engine_kwargs, sampling_kwargs, selected_logprobs
from examples.ouro.worker import rollout_to_tensordict


def output(mode="raw"):
    return SimpleNamespace(
        finished=True,
        weight_version=3,
        sampling_params=SimpleNamespace(temperature=1, top_p=1, top_k=-1, seed=17, logprobs_mode=mode),
        prompt_token_ids=[2, 3],
        token_ids=[5, 7],
        exit_depths=[4, 2],
        log_probs=[-1.0, -2.0],
    )


def test_explicit_current_publication_profile_preserves_actual_scores_and_version():
    item = output()
    assert engine_kwargs("publication_outputs") == {}
    assert sampling_kwargs("publication_outputs") == {"logprobs_mode": "raw"}
    assert selected_logprobs(item, "publication_outputs") is item.log_probs
    batch = rollout_to_tensordict(
        [item],
        expected_version=3,
        max_loops=4,
        contract="publication_outputs",
        logprobs_mode="raw_logprobs",
    )
    assert batch["weight_version"].tolist() == [3]
    assert batch["rollout_log_probs"].values().tolist() == [0.0, -1.0, -2.0, 0.0]


@pytest.mark.parametrize("provenance,mode", [(None, "raw"), ("raw_logprobs", "processed")])
def test_current_publication_profile_requires_explicit_and_matching_raw_provenance(provenance, mode):
    with pytest.raises(ValueError, match="raw"):
        rollout_to_tensordict(
            [output(mode)],
            expected_version=3,
            max_loops=4,
            contract="publication_outputs",
            logprobs_mode=provenance,
        )


def test_publication_profile_never_guesses_new_output_spelling():
    with pytest.raises(ValueError, match="log_probs"):
        selected_logprobs(SimpleNamespace(logprobs=[-0.1]), "publication_outputs")


def test_native_publication_profile_reaches_adapter_and_training_conversion(tmp_path):
    import asyncio

    import torch
    from vllm_rlt import LLM, CacheConfig

    from examples.ouro.rollout import OuroRollout
    from examples.ouro.tests.test_migration import config_pair, prompts
    from examples.ouro.tests.test_replay import native_model
    from examples.ouro.worker import training_batch_from_rollout

    if not hasattr(LLM, "start_weight_update"):
        pytest.skip("requires actual #78 publication implementation")
    source = native_model()
    backend = LLM(source, cache_config=CacheConfig(64, 2), device="cpu", dtype=torch.float32)
    config, model_config = config_pair(tmp_path)
    config.engine_kwargs = {"vllm_rlt": {"contract": "publication_outputs"}}
    rollout = OuroRollout(
        config,
        model_config,
        None,
        backend=backend,
        backend_logprobs_mode="raw_logprobs",
        device="cpu",
    )
    try:
        assert asyncio.run(rollout.update_weights(source.named_parameters(), global_steps=0)) == 1
        actual = rollout.generate_sequences(prompts())
        assert actual.meta_info["rlt_engine_contract"] == "publication_outputs"
        batch = training_batch_from_rollout(
            actual,
            advantages=torch.ones(1, 4),
            expected_version=1,
            world_size=1,
            global_batch_size=1,
            global_token_count=4,
        )
        assert batch["weight_version"].tolist() == [1]
        assert torch.isfinite(batch["rollout_log_probs"].values()).all()
    finally:
        rollout.close()
