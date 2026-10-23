# SPDX-License-Identifier: Apache-2.0
"""Seal historical GPU drivers to their originally qualified engine sources."""

import hashlib
import json
from pathlib import Path

import vllm_rlt


def require_historical_prototype():
    expected = json.loads((Path(__file__).parent / "results/2026-10-08/source-manifest.json").read_text())["engine"]
    root = Path(vllm_rlt.__file__).resolve().parent
    actual = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*.py")}
    if actual != expected:
        raise NotImplementedError(
            "This is a historical GPU driver qualified only against engine bdce287. "
            "Restore that exact prototype for historical reproduction; use "
            "examples.ouro.validate_current for current-engine generation checks. "
            "Old publication/graph/NIXL evidence does not qualify another engine revision."
        )
