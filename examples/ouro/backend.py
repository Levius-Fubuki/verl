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
"""Explicit engine contracts and a token batch driver using public methods only.

``rollout_outputs`` is the #83 engine contract. ``publication_outputs`` explicitly
selects #78's current publication implementation and its legacy score spelling.
``prototype`` is reserved for historical reproduction. No profile is selected
by inspecting an output or retrying a failed call.
"""

from uuid import uuid4


def _check_contract(contract):
    if contract not in ("rollout_outputs", "prototype", "publication_outputs"):
        raise ValueError(
            f"unknown backend contract {contract!r}; choose 'rollout_outputs', 'prototype', or 'publication_outputs'"
        )


def selected_logprobs(output, contract="rollout_outputs"):
    """Return the engine's selected probabilities without converting or inventing them."""
    _check_contract(contract)
    attribute = "logprobs" if contract == "rollout_outputs" else "log_probs"
    try:
        return getattr(output, attribute)
    except AttributeError as exc:
        raise ValueError(f"backend contract {contract!r} requires output.{attribute}") from exc


def sampling_kwargs(contract="rollout_outputs"):
    """Probability options accepted by this contract's SamplingParams."""
    _check_contract(contract)
    return {} if contract == "rollout_outputs" else {"logprobs_mode": "raw"}


def engine_kwargs(contract="rollout_outputs"):
    """Probability options accepted when constructing this contract's engine."""
    _check_contract(contract)
    return {"logprobs_mode": "raw_logprobs"} if contract == "rollout_outputs" else {}


def require_publication(backend):
    """Fail before training if the backend lacks public transactional publication."""
    required = ("start_weight_update", "update_weights", "finish_weight_update", "get_weight_version")
    missing = [name for name in required if not callable(getattr(backend, name, None))]
    if missing:
        raise NotImplementedError(
            "backend is missing public weight publication methods: "
            + ", ".join(missing)
            + ". Engine #83 alone lacks the training publication contract; use a backend with "
            "the public publication extension. Synthetic versions, direct weight copies, and "
            "private RPC are not substitutes."
        )


class TokenBatchDriver:
    """Run an exclusive token batch on an LLM facade or a public request endpoint.

    A native ``generate`` owns its request lifecycle. Otherwise this driver owns
    unique IDs admitted through ``add_request`` and aborts outstanding owned work
    on failure. No scheduler, worker, or facade internals are accessed.
    """

    def __init__(self, backend):
        self.backend = backend
        self._prefix = uuid4().hex
        self._counter = 0
        self._closed = False

    @staticmethod
    def _validate_output(output, prompt):
        if output.prompt_token_ids != list(prompt):
            raise RuntimeError("engine returned foreign prompt metadata")

    def generate(self, prompts, params, *, trace_ids=None):
        """Generate with optional semantic trace IDs, independent of transport IDs.

        Trace IDs require a public request endpoint: the native LLM facade's
        generate contract cannot carry per-request trace identity.
        """
        if self._closed:
            raise RuntimeError("token batch driver is closed")
        prompts = list(prompts)
        all_params = params if isinstance(params, list | tuple) else [params] * len(prompts)
        if len(all_params) != len(prompts):
            raise ValueError("provide one sampling parameter object per prompt")
        if trace_ids is not None:
            if isinstance(trace_ids, str | bytes):
                raise ValueError("trace_ids must contain one nonempty string per prompt")
            try:
                trace_ids = list(trace_ids)
            except TypeError as exc:
                raise ValueError("trace_ids must contain one nonempty string per prompt") from exc
            if len(trace_ids) != len(prompts):
                raise ValueError("trace_ids count must match the prompt count")
            if any(not isinstance(trace_id, str) or not trace_id for trace_id in trace_ids):
                raise ValueError("trace_ids must contain one nonempty string per prompt")
        native_generate = getattr(self.backend, "generate", None)
        if trace_ids is not None and callable(native_generate):
            raise NotImplementedError(
                "trace_ids require a public request endpoint such as StageClient or PDEngine; "
                "the native generate facade cannot forward semantic trace IDs"
            )
        unfinished = getattr(self.backend, "has_unfinished_requests", None)
        if callable(unfinished) and unfinished():
            raise RuntimeError("finish existing backend requests before generating a batch")
        if not prompts:
            return []
        if callable(native_generate):
            native_params = list(params) if isinstance(params, tuple) else params
            outputs = list(native_generate(prompts, native_params))
            if len(outputs) != len(prompts):
                raise RuntimeError("engine returned an incomplete token batch")
            seen = set()
            for output, prompt in zip(outputs, prompts, strict=True):
                if not output.finished:
                    raise RuntimeError("engine returned an unfinished token batch")
                if output.request_id in seen:
                    raise RuntimeError("engine returned a duplicate request output")
                seen.add(output.request_id)
                self._validate_output(output, prompt)
            return outputs
        required = ("add_request", "step", "has_unfinished_requests", "abort_request")
        missing = [name for name in required if not callable(getattr(self.backend, name, None))]
        if missing:
            raise TypeError("backend requires public generate or request methods: " + ", ".join(missing))
        owned, results = {}, {}
        try:
            for index, (prompt, request_params) in enumerate(zip(prompts, all_params, strict=True)):
                request_id = f"ouro-{self._prefix}-{self._counter}"
                self._counter += 1
                # Include an admission attempt in cleanup: a transport may fail
                # after the endpoint accepted the request.
                owned[request_id] = list(prompt)
                request_options = {} if trace_ids is None else {"trace_id": trace_ids[index]}
                self.backend.add_request(request_id, owned[request_id], request_params, **request_options)
            while self.backend.has_unfinished_requests():
                for output in self.backend.step():
                    request_id = output.request_id
                    if request_id not in owned:
                        raise RuntimeError(f"engine returned foreign request output {request_id!r}")
                    if request_id in results:
                        raise RuntimeError(f"engine returned duplicate finished output {request_id!r}")
                    self._validate_output(output, owned[request_id])
                    if output.finished:
                        results[request_id] = output
            if len(results) != len(owned):
                raise RuntimeError("engine returned an incomplete token batch")
            return [results[request_id] for request_id in owned]
        except BaseException as exc:
            failures = []
            has_request = getattr(self.backend, "has_request", None)
            for request_id in owned:
                if request_id in results:
                    continue
                if callable(has_request):
                    try:
                        if not has_request(request_id):
                            continue
                    except BaseException as cleanup_exc:
                        failures.append(f"has_request({request_id!r}): {cleanup_exc!r}")
                try:
                    self.backend.abort_request(request_id)
                except BaseException as cleanup_exc:
                    failures.append(f"abort_request({request_id!r}): {cleanup_exc!r}")
            if failures:
                raise RuntimeError("owned request cleanup failed: " + "; ".join(failures)) from exc
            raise

    def close(self):
        """Delegate optional public close once; a failed close remains retryable."""
        if not self._closed:
            close = getattr(self.backend, "close", None)
            if callable(close):
                close()
            self._closed = True
