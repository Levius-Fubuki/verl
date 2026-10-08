# SPDX-License-Identifier: Apache-2.0
import pytest


def test_arithmetic_reward_is_defined_by_answer_not_sample_index():
    from examples.ouro.recipe_utils import arithmetic_reward

    assert arithmetic_reward(" 5\n", 5) == 1.0
    assert arithmetic_reward(" 4\n", 5) == 0.5
    assert arithmetic_reward(" no numeric answer", 5) == 0.0


def test_dataset_schedule_and_provenance_validation():
    from examples.ouro.recipe_utils import scheduled_examples, validate_resume_config

    first = scheduled_examples(step=0, count=2)
    assert first == scheduled_examples(step=0, count=2)
    assert first != scheduled_examples(step=1, count=2)
    config = {"model_revision": "x", "response_length": 32, "seed": 123, "steps": 3}
    validate_resume_config(config, {**config, "steps": 5})
    with pytest.raises(ValueError, match="response_length"):
        validate_resume_config(config, {**config, "response_length": 64})


def test_probability_envelope_rejects_nonfinite_and_outliers():
    import torch

    from examples.ouro.recipe_utils import check_probability_envelope

    mask = torch.tensor([[True, False]])
    stats = check_probability_envelope(torch.tensor([[-1.0, float("nan")]]), torch.tensor([[-1.1, 0.0]]), mask)
    assert stats["max_abs_logprob"] == pytest.approx(0.1)
    assert stats["out_of_band_fraction"] == 0.0
    with pytest.raises(ValueError, match="envelope"):
        check_probability_envelope(torch.tensor([[-0.1, 0.0]]), torch.tensor([[-2.0, 0.0]]), mask)
    with pytest.raises(ValueError, match="finite"):
        check_probability_envelope(torch.tensor([[float("nan"), 0.0]]), torch.tensor([[-2.0, 0.0]]), mask)
