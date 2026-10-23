# SPDX-License-Identifier: Apache-2.0
"""Public backend contract tests; engine integration runs with a tiny CPU model."""

from types import SimpleNamespace

import pytest

from examples.ouro.backend import (
    TokenBatchDriver,
    engine_kwargs,
    require_publication,
    sampling_kwargs,
    selected_logprobs,
)


def output(request_id, prompt, *, finished=True):
    return SimpleNamespace(request_id=request_id, prompt_token_ids=list(prompt), finished=finished)


class PublicEndpoint:
    """Minimal public endpoint for deterministic failure injection, without engine internals."""

    def __init__(self, failure=None, has_request=True):
        self.active = {}
        self.added = []
        self.aborted = []
        self.failure = failure
        self.close_calls = 0
        if not has_request:
            self.has_request = None

    def add_request(self, request_id, prompt_token_ids, sampling_params):
        self.active[request_id] = list(prompt_token_ids)
        self.added.append((request_id, sampling_params))
        if self.failure == "admission" and len(self.added) == 2:
            raise ValueError("admission failed after accepting request")

    def has_request(self, request_id):
        return request_id in self.active

    def has_unfinished_requests(self):
        return bool(self.active)

    def step(self):
        if self.failure in ("step", "cleanup"):
            raise ValueError("step failed")
        if self.failure == "interrupt":
            raise KeyboardInterrupt("cancel generation")
        if self.failure == "foreign":
            return [output("someone-else", [17])]
        if self.failure == "incomplete":
            self.active.clear()
            return []
        result = [output(key, prompt) for key, prompt in reversed(tuple(self.active.items()))]
        self.active.clear()
        if self.failure == "duplicate":
            result.append(result[0])
        return result

    def abort_request(self, request_id):
        self.aborted.append(request_id)
        if self.failure == "cleanup":
            raise OSError("abort transport failed")
        self.active.pop(request_id, None)

    def close(self):
        self.close_calls += 1


@pytest.fixture
def cpu_engine(request):
    import torch
    from vllm_rlt import CacheConfig, ExitConfig, SchedulerConfig
    from vllm_rlt.engine.llm_engine import LLMEngine
    from vllm_rlt.request import RequestOutput

    if "logprobs" not in RequestOutput.__dataclass_fields__:
        pytest.skip("current rollout-output tests require pinned #83; prototype runs retain explicit separate coverage")
    from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

    torch.manual_seed(123)
    config = OuroConfig(
        vocab_size=37,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        total_ut_steps=4,
        max_position_embeddings=128,
    )
    model = OuroForCausalLM(config).to(device="cpu", dtype=torch.float32)
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=64, block_size=2),
        scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=4),
        attention_backend="torch",
        exit_config=ExitConfig("trace", depths_by_request=request.param) if hasattr(request, "param") else None,
        **engine_kwargs(),
    )
    try:
        yield engine
    finally:
        engine.close()


def test_current_cpu_engine_outputs_have_selected_raw_logprobs(cpu_engine):
    import math

    from vllm_rlt import SamplingParams

    params = SamplingParams(max_tokens=3, logprobs=0, ignore_eos=True, **sampling_kwargs())
    outputs = TokenBatchDriver(cpu_engine).generate([[2, 3], [5, 7, 11]], params)
    assert [item.prompt_token_ids for item in outputs] == [[2, 3], [5, 7, 11]]
    for item in outputs:
        values = selected_logprobs(item)
        assert item.finished and len(item.token_ids) == len(values) == 3
        assert all(math.isfinite(value) and value <= 0 for value in values)
        with pytest.raises((AttributeError, ValueError), match="log_probs|prototype"):
            selected_logprobs(item, "prototype")
    assert not cpu_engine.has_unfinished_requests()


def test_explicit_profiles_do_not_guess_or_fall_back():
    current = SimpleNamespace(logprobs=[-0.3])
    prototype = SimpleNamespace(log_probs=[-0.4])
    assert selected_logprobs(current) is current.logprobs
    assert selected_logprobs(prototype, "prototype") is prototype.log_probs
    assert sampling_kwargs() == {}
    assert sampling_kwargs("prototype") == {"logprobs_mode": "raw"}
    assert engine_kwargs() == {"logprobs_mode": "raw_logprobs"}
    assert engine_kwargs("prototype") == {}
    with pytest.raises((AttributeError, ValueError), match="logprobs|rollout_outputs"):
        selected_logprobs(prototype)
    with pytest.raises((AttributeError, ValueError), match="log_probs|prototype"):
        selected_logprobs(current, "prototype")


@pytest.mark.parametrize("helper", [engine_kwargs, sampling_kwargs, lambda name: selected_logprobs(None, name)])
def test_unknown_contract_rejected(helper):
    with pytest.raises(ValueError, match="contract.*rollout_outputs.*prototype"):
        helper("auto")


def test_publication_requires_all_public_methods():
    methods = ("start_weight_update", "update_weights", "finish_weight_update", "get_weight_version")
    endpoint = SimpleNamespace(**{name: lambda: None for name in methods})
    require_publication(endpoint)
    endpoint.finish_weight_update = None
    with pytest.raises(NotImplementedError, match="finish_weight_update") as caught:
        require_publication(endpoint)
    assert "#83" in str(caught.value) and "training" in str(caught.value)
    assert "publication" in str(caught.value)


def test_current_engine_cannot_claim_publication(cpu_engine):
    with pytest.raises(NotImplementedError, match="publication"):
        require_publication(cpu_engine)


def test_public_loop_preserves_order_parameters_and_unique_string_ids():
    endpoint = PublicEndpoint()
    driver = TokenBatchDriver(endpoint)
    params = [object(), object()]
    prompts = [[2, 3], [5]]
    assert [item.prompt_token_ids for item in driver.generate(prompts, params)] == prompts
    driver.generate([[7]], params[0])
    other = TokenBatchDriver(endpoint)
    other.generate([[11]], params[1])
    ids = [key for key, _ in endpoint.added]
    assert all(isinstance(key, str) and key for key in ids)
    assert len(set(ids)) == 4
    assert endpoint.added[0][1] is params[0] and endpoint.added[1][1] is params[1]


def test_rejects_existing_work_without_aborting_it():
    endpoint = PublicEndpoint()
    endpoint.active["user-owned"] = [1]
    with pytest.raises(RuntimeError, match="existing|preexisting"):
        TokenBatchDriver(endpoint).generate([[2]], object())
    assert endpoint.active == {"user-owned": [1]}
    assert endpoint.aborted == [] and endpoint.added == []


def test_parameter_count_checked_before_admission():
    endpoint = PublicEndpoint()
    with pytest.raises(ValueError, match="parameter|SamplingParams"):
        TokenBatchDriver(endpoint).generate([[2], [3]], [object()])
    assert endpoint.added == []


@pytest.mark.parametrize("failure", ["step", "admission", "foreign", "incomplete", "duplicate"])
@pytest.mark.parametrize("has_request", [True, False])
def test_failures_reject_partial_batches_and_clean_owned_requests(failure, has_request):
    endpoint = PublicEndpoint(failure, has_request)
    with pytest.raises((RuntimeError, ValueError), match="step failed|admission failed|foreign|incomplete|duplicate"):
        TokenBatchDriver(endpoint).generate([[2], [3]], object())
    assert not endpoint.active
    assert set(endpoint.aborted).issubset({key for key, _ in endpoint.added})
    if failure in ("step", "admission", "foreign"):
        assert set(endpoint.aborted) == {key for key, _ in endpoint.added}


def test_cancellation_also_aborts_owned_requests():
    endpoint = PublicEndpoint("interrupt")
    with pytest.raises(KeyboardInterrupt, match="cancel"):
        TokenBatchDriver(endpoint).generate([[2]], object())
    assert not endpoint.active


def test_cleanup_failure_reports_original_and_each_abort_error():
    endpoint = PublicEndpoint("cleanup")
    with pytest.raises(RuntimeError, match="cleanup.*failed") as caught:
        TokenBatchDriver(endpoint).generate([[2], [3]], object())
    assert isinstance(caught.value.__cause__, ValueError)
    assert "step failed" in str(caught.value.__cause__)
    assert "abort transport failed" in str(caught.value)
    assert set(endpoint.aborted) == {key for key, _ in endpoint.added}


def test_native_generate_is_delegated_without_assuming_engine_or_scheduler():
    prompts, params = [[2], [3]], [object(), object()]
    expected = [output("native-0", [2]), output("native-1", [3])]

    class Facade:
        def generate(self, received_prompts, received_params):
            assert received_prompts == prompts and received_params is params
            return expected

    assert TokenBatchDriver(Facade()).generate(prompts, params) == expected


@pytest.mark.parametrize("failure", ["count", "unfinished", "foreign", "duplicate"])
def test_native_generate_results_must_be_complete_and_aligned(failure):
    results = [output("a", [2]), output("b", [3])]
    if failure == "count":
        results.pop()
    elif failure == "unfinished":
        results[0].finished = False
    elif failure == "foreign":
        results[0].prompt_token_ids = [17]
    elif failure == "duplicate":
        results[1].request_id = "a"
    facade = SimpleNamespace(generate=lambda prompts, params: results)
    with pytest.raises(RuntimeError, match="incomplete|unfinished|foreign|duplicate"):
        TokenBatchDriver(facade).generate([[2], [3]], object())


def test_empty_batch_and_optional_idempotent_close():
    endpoint = PublicEndpoint()
    driver = TokenBatchDriver(endpoint)
    assert driver.generate([], object()) == []
    assert endpoint.added == []
    driver.close()
    driver.close()
    assert endpoint.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        driver.generate([[2]], object())
    TokenBatchDriver(SimpleNamespace()).close()


def test_partial_public_endpoint_rejected_actionably():
    with pytest.raises(TypeError, match="add_request|step|public"):
        TokenBatchDriver(SimpleNamespace()).generate([[2]], object())


@pytest.mark.parametrize("cpu_engine", [{"semantic-sample": [4, 1, 3]}], indirect=True)
def test_trace_ids_reuse_semantic_trace_across_cpu_batches(cpu_engine):
    from vllm_rlt import SamplingParams

    driver = TokenBatchDriver(cpu_engine)
    params = SamplingParams(max_tokens=3, min_loops=1, logprobs=0, ignore_eos=True)
    first = driver.generate([[2, 3]], params, trace_ids=["semantic-sample"])[0]
    second = driver.generate([[2, 3]], params, trace_ids=["semantic-sample"])[0]
    assert first.exit_depths == second.exit_depths == [4, 1, 3]
    assert first.token_ids == second.token_ids
    assert first.logprobs == second.logprobs
    assert first.request_id != second.request_id
    assert first.request_id != "semantic-sample" and second.request_id != "semantic-sample"
    assert not cpu_engine.has_unfinished_requests()


@pytest.mark.parametrize("trace_ids", [[], ["a", "b"], [""], [None], [3], "a"])
def test_invalid_trace_ids_rejected_before_admission(trace_ids):
    endpoint = PublicEndpoint()
    with pytest.raises(ValueError, match="trace_ids.*(count|string|prompt)"):
        TokenBatchDriver(endpoint).generate([[2]], object(), trace_ids=trace_ids)
    assert endpoint.added == [] and endpoint.active == {} and endpoint.aborted == []


@pytest.mark.parametrize("prompts,trace_ids", [([[2]], ["sample"]), ([], [])])
def test_native_facade_rejects_trace_ids_before_any_backend_work(prompts, trace_ids):
    calls = []
    facade = SimpleNamespace(
        generate=lambda *args: calls.append("generate"),
        has_unfinished_requests=lambda: calls.append("has_unfinished_requests"),
    )
    with pytest.raises(NotImplementedError, match="trace_ids.*public.*request.*endpoint"):
        TokenBatchDriver(facade).generate(prompts, object(), trace_ids=trace_ids)
    assert calls == []


def test_native_facade_receives_tuple_parameters_as_a_list():
    params = (object(), object())
    expected = [output("native-0", [2]), output("native-1", [3])]

    class ListFacade:
        def generate(self, prompts, received_params):
            assert isinstance(received_params, list)
            assert received_params[0] is params[0] and received_params[1] is params[1]
            return expected

    assert TokenBatchDriver(ListFacade()).generate([[2], [3]], params) == expected
