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
"""Online lookahead must execute the predicted depth and preserve causal KV."""

import importlib.util

import pytest
import torch

from examples.ouro.tests.test_replay import engine_logits, native_model


def api():
    assert importlib.util.find_spec("examples.ouro.nplus2") is not None, "online N+2 module is missing"
    from examples.ouro.nplus2 import LookaheadHead, generate

    return LookaheadHead, generate


def artifact(bias=20.0):
    return {
        "schema": 1,
        "source_depth": 1,
        "target_depth": 3,
        "max_loops": 4,
        "model_revision": "model-pin",
        "mean": torch.zeros(32),
        "scale": torch.ones(32),
        "weight": torch.zeros(1, 32),
        "bias": torch.tensor([bias]),
    }


def test_positive_prediction_executes_through_three_and_replays_selected_scores(tmp_path):
    cls, generate = api()
    path = tmp_path / "head.pt"
    torch.save(artifact(), path)
    head = cls.load(path, model_revision="model-pin", hidden_size=32, max_loops=4)
    native = native_model()
    result = generate(native, [2, 3, 5], max_tokens=5, policy="nplus2", head=head)
    assert result["exit_depths"] == [4, 3, 3, 3, 3]
    assert result["selected_depths"] == result["exit_depths"]
    assert result["prediction_depths"] == [None, 1, 1, 1, 1]
    inputs = torch.tensor([2, 3, 5] + result["token_ids"][:-1])
    logits = engine_logits(native, inputs, result["input_depths"])
    expected = logits[2:].float().log_softmax(-1).gather(1, torch.tensor(result["token_ids"])[:, None])
    torch.testing.assert_close(torch.tensor(result["log_probs"]), expected[:, 0], atol=3e-6, rtol=3e-5)
    assert result["used_blocks_after_free"] == 0


def test_negative_predictor_matches_full_depth(tmp_path):
    cls, generate = api()
    path = tmp_path / "head.pt"
    torch.save(artifact(-20.0), path)
    head = cls.load(path, model_revision="model-pin", hidden_size=32, max_loops=4)
    native = native_model()
    a = generate(native, [2, 3], max_tokens=4, policy="nplus2", head=head)
    b = generate(native, [2, 3], max_tokens=4, policy="full")
    assert a["token_ids"] == b["token_ids"]
    assert a["log_probs"] == b["log_probs"]
    assert a["exit_depths"] == [4] * 4


@pytest.mark.parametrize(
    "change",
    [
        {"model_revision": "wrong"},
        {"target_depth": 2},
        {"scale": torch.zeros(32)},
        {"weight": torch.full((1, 32), float("nan"))},
    ],
)
def test_invalid_or_mismatched_head_fails_closed(tmp_path, change):
    cls, _ = api()
    data = artifact()
    data.update(change)
    path = tmp_path / "bad.pt"
    torch.save(data, path)
    with pytest.raises(ValueError):
        cls.load(path, model_revision="model-pin", hidden_size=32, max_loops=4)


def test_original_gate_baseline_preserves_native_minimum_two_loops():
    _, generate = api()
    result = generate(native_model(), [2, 3], max_tokens=3, policy="gate", threshold=0.0)
    assert result["exit_depths"] == [4, 2, 2]
