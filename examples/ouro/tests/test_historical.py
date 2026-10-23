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
"""Historical GPU drivers must reject a different engine before allocating CUDA."""

import os
import subprocess
import sys

import pytest
from vllm_rlt.request import RequestOutput


@pytest.mark.skipif(
    "logprobs" not in RequestOutput.__dataclass_fields__, reason="requires the current rollout-only engine"
)
@pytest.mark.parametrize("module", ["validate_adapter", "validate_runtime", "validate_worker"])
def test_historical_gpu_driver_rejects_current_engine_before_execution(module, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            f"examples.ouro.{module}",
            "--model",
            "/nonexistent/ouro",
            "--output",
            str(tmp_path / "unused.json"),
        ],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode != 0
    assert "historical" in result.stderr.lower()
    assert "bdce287" in result.stderr
    assert not (tmp_path / "unused.json").exists()
