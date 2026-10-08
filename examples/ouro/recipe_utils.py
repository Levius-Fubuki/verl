# SPDX-License-Identifier: Apache-2.0
"""Deterministic acceptance dataset/reward, separate from the training runner."""

import re

import torch


def arithmetic_reward(text, answer):
    """Score the first integer: exact=1, numeric proximity otherwise, no number=0.

    This dense reward supplies a small executable RL task. A few integration
    iterations do not establish held-out arithmetic quality or convergence.
    """
    match = re.search(r"[-+]?\d+", text)
    return 0.0 if match is None else 1.0 / (1.0 + abs(int(match.group()) - answer))


def scheduled_examples(*, step, count):
    if type(step) is not int or step < 0 or type(count) is not int or count < 1:
        raise ValueError("step/count must be nonnegative/positive integers")
    examples = [(a, b) for a in range(1, 9) for b in range(1, 9)]
    selected = [examples[(step * count + index) % len(examples)] for index in range(count)]
    return [
        {"prompt": f"Question: What is {a} + {b}?\nAnswer:", "answer": a + b, "uid": f"addition-{a}-{b}"}
        for a, b in selected
    ]


def validate_resume_config(saved, current):
    for key, value in saved.items():
        if key not in ("steps", "output", "checkpoint_at", "checkpoint_dir", "resume") and current.get(key) != value:
            raise ValueError(f"resume configuration changed: {key}")


def check_probability_envelope(
    log_prob, rollout_log_prob, response_mask, *, max_abs_logprob=0.5, min_ratio=0.5, max_ratio=2.0
):
    """Fail the acceptance run on invalid/outlying behavior likelihood ratios.

    This BF16 smoke envelope is declared before running the gated recipe. It
    does not establish 1e-3 kernel parity. Report the raw discrepancies and TIS
    separately; changing these bounds changes experiment acceptance.
    """
    values = (log_prob.float() - rollout_log_prob.float())[response_mask.bool()]
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("valid response probability differences must be nonempty and finite")
    ratios = values.exp()
    outside = (values.abs() > max_abs_logprob) | (ratios < min_ratio) | (ratios > max_ratio)
    result = {
        "max_abs_logprob": values.abs().max().item(),
        "ratio_min": ratios.min().item(),
        "ratio_max": ratios.max().item(),
        "out_of_band_fraction": outside.float().mean().item(),
    }
    if bool(outside.any()):
        raise ValueError(f"BF16 behavior probability envelope exceeded: {result}")
    return result
