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
