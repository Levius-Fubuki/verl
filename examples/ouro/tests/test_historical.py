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
