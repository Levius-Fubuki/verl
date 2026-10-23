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
import asyncio
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from examples.ouro.tests.test_replay import native_model
from verl import DataProto
from verl.workers.config import HFModelConfig, RolloutConfig


def make_rollout(tmp_path):
    from vllm_rlt import CacheConfig
    from vllm_rlt.engine.llm_engine import LLMEngine

    from examples.ouro.rollout import OuroRollout

    if not hasattr(LLMEngine, "start_weight_update"):
        pytest.skip("real versioned-rollout acceptance requires the public publication extension")
    native = native_model()
    (tmp_path / "config.json").write_text(json.dumps(native.config.to_dict()))
    save_file(native.state_dict(), tmp_path / "model.safetensors")
    model = HFModelConfig(
        path=str(tmp_path),
        load_tokenizer=False,
        external_lib="examples.ouro.hf",
        override_config={"attn_implementation": "eager"},
    )
    config = RolloutConfig(
        name="vllm_rlt",
        mode="async",
        tensor_model_parallel_size=1,
        response_length=4,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        ignore_eos=True,
        engine_kwargs={"vllm_rlt": {"contract": "prototype"}},
    )
    return OuroRollout(
        config,
        model,
        None,
        device="cpu",
        attention_backend="torch",
        backend=LLMEngine(native, cache_config=CacheConfig(num_blocks=32, block_size=2)),
        backend_logprobs_mode="raw_logprobs",
    )


def prompts():
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[0, 2, 3], [5, 7, 11]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        },
        non_tensors={"uid": np.array(["a", "b"], dtype=object)},
    )


def test_real_base_rollout_metadata_publication_and_restart(tmp_path):
    from verl.workers.rollout.base import BaseRollout, get_rollout_class

    rollout = make_rollout(tmp_path)
    assert isinstance(rollout, BaseRollout)
    assert get_rollout_class("vllm_rlt", "async") is type(rollout)
    native = rollout.backend.model
    asyncio.run(rollout.update_weights(native.named_parameters(), global_steps=0))
    output = rollout.generate_sequences(prompts())
    assert output.batch["responses"].shape == (2, 4)
    assert output.batch["weight_version"].tolist() == [1, 1]
    assert len(set(output.batch["sampling_seed"].tolist())) == 2
    assert output.batch["response_mask"].sum().item() == 8
    assert (output.batch["rollout_log_probs"] <= 0).all()
    state = rollout.state_dict()
    expected = rollout.generate_sequences(prompts())
    second = make_rollout(tmp_path)
    second.load_state_dict(state)
    asyncio.run(second.update_weights(native.named_parameters(), global_steps=0))
    actual = second.generate_sequences(prompts())
    torch.testing.assert_close(actual.batch["responses"], expected.batch["responses"], atol=0, rtol=0)
    torch.testing.assert_close(actual.batch["sampling_seed"], expected.batch["sampling_seed"], atol=0, rtol=0)
    torch.testing.assert_close(actual.batch["rollout_log_probs"], expected.batch["rollout_log_probs"], atol=0, rtol=0)
    rollout.close()
    second.close()


def test_standard_ppo_response_alignment_and_empty_rejection(tmp_path):
    from examples.ouro.worker import training_batch_from_rollout
    from verl.workers.utils.padding import no_padding_2_padding

    rollout = make_rollout(tmp_path)
    output = rollout.generate_sequences(prompts())
    batch = training_batch_from_rollout(
        output,
        advantages=torch.ones_like(output.batch["rollout_log_probs"]),
        expected_version=0,
        world_size=2,
        global_batch_size=2,
        global_token_count=8,
    )
    assert batch["response_mask"].offsets().diff().tolist() == [4, 4]
    assert batch["loss_mask"].unbind()[0].tolist() == [False, False, True, True, True, True]
    values = batch["input_ids"].values().float()
    predicted = torch.nested.nested_tensor_from_jagged(values, batch["input_ids"].offsets())
    actual = no_padding_2_padding(predicted, batch)
    for row, native in enumerate(output.non_tensor_batch["rlt_outputs"]):
        inputs = native.prompt_token_ids + native.token_ids
        assert actual[row].tolist() == inputs[len(native.prompt_token_ids) - 1 : -1]
    bad = output.non_tensor_batch["rlt_outputs"].copy()
    bad[0] = replace(bad[0], finish_reason="abort")
    output.non_tensor_batch["rlt_outputs"] = bad
    with pytest.raises(ValueError, match="abort"):
        training_batch_from_rollout(
            output,
            advantages=torch.ones(2, 4),
            expected_version=0,
            world_size=1,
            global_batch_size=2,
            global_token_count=8,
        )
    rollout.close()


def test_invalid_publication_and_lifecycle(tmp_path):
    rollout = make_rollout(tmp_path)
    with pytest.raises(ValueError, match="wire_format"):
        asyncio.run(rollout.update_weights([], wire_format="delta_flush", global_steps=0))
    asyncio.run(rollout.release())
    with pytest.raises(RuntimeError, match="released"):
        rollout.generate_sequences(prompts())
    asyncio.run(rollout.resume(["weights", "kv_cache"]))
    assert rollout.generate_sequences(prompts()).batch["responses"].shape == (2, 4)
    rollout.close()


def test_short_response_microbatch_matches_standard_ppo_width(tmp_path):
    from examples.ouro.worker import training_batch_from_rollout
    from verl.workers.config import ActorConfig
    from verl.workers.utils.losses import ppo_loss

    rollout = make_rollout(tmp_path)
    output = rollout.generate_sequences(prompts())
    first = output.non_tensor_batch["rlt_outputs"][0]
    output = output[[0]]
    output.non_tensor_batch["rlt_outputs"][0] = replace(
        first, token_ids=first.token_ids[:2], exit_depths=first.exit_depths[:2], log_probs=first.log_probs[:2]
    )
    batch = training_batch_from_rollout(
        output, advantages=torch.ones(1, 4), expected_version=0, world_size=1, global_batch_size=1, global_token_count=2
    )
    model_values = torch.zeros(batch["input_ids"].values().shape)
    model_output = {"log_probs": torch.nested.nested_tensor_from_jagged(model_values, batch["input_ids"].offsets())}
    loss, _ = ppo_loss(ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1), model_output, batch)
    assert torch.isfinite(loss)
    rollout.close()
