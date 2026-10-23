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
"""Single-rank real verl/FSDP + native rollout/update experiment.

Uses a synthetic advantage to exercise PPO plumbing, not to claim task learning.
Requires the pinned 0z5a training-contract engine prototype documented in README.
"""

import argparse
import gc
import hashlib
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from vllm_rlt import LLM, SamplingParams
from vllm_rlt.config import CacheConfig, SchedulerConfig

from examples.ouro.worker import OuroFSDPEngine, rollout_to_tensordict
from verl.trainer.config import CheckpointConfig
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig


def tensor_state_digest(engine):
    digest = hashlib.sha256()
    for name, tensor in engine.module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    for state in engine.optimizer.state.values():
        for key, value in sorted(state.items()):
            digest.update(key.encode())
            if isinstance(value, torch.Tensor):
                digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                digest.update(str(value).encode())
    digest.update(json.dumps(engine.lr_scheduler.state_dict(), sort_keys=True).encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--layer-wrap", action="store_true")
    parser.add_argument("--early-exit", action="store_true")
    parser.add_argument("--rollout-correction", action="store_true")
    args = parser.parse_args()
    from examples.ouro.historical import require_historical_prototype

    require_historical_prototype()
    torch.manual_seed(123)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "world_size": 1,
        "scope": "verl FSDPEngine + PPO loss, synthetic advantages, single-rank NO_SHARD",
        "packed": args.packed,
        "layer_wrap": args.layer_wrap,
        "tokens_per_response": args.tokens,
    }
    params = SamplingParams(
        max_tokens=args.tokens,
        temperature=1.0,
        logprobs=0,
        seed=123,
        ignore_eos=True,
        exit_threshold=0.0 if args.early_exit else 1.0,
    )
    llm = LLM(
        args.model,
        device="cuda",
        dtype=torch.bfloat16,
        attention_backend="triton",
        cache_config=CacheConfig(num_blocks=128),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64),
    )
    # Token-ID prompts deliberately avoid downloading/executing model tokenizer code.
    prompts = [[2, 3, 5, 7], [11, 13, 17]]
    outputs = llm.generate(prompts, params)
    report["rollouts_before"] = [asdict(output) for output in outputs]
    batch = rollout_to_tensordict(
        outputs, expected_version=0, max_loops=4, contract="prototype", packed=args.packed, device="cuda"
    )
    with tempfile.TemporaryDirectory() as temp:
        dist.init_process_group("nccl", init_method="file://" + temp + "/init", rank=0, world_size=1)
        try:
            model_config = HFModelConfig(
                path=args.model,
                load_tokenizer=False,
                external_lib="examples.ouro.hf",
                use_remove_padding=args.packed,
                override_config={"attn_implementation": "eager"},
            )
            config = FSDPEngineConfig(
                use_orig_params=True,
                wrap_policy={} if args.layer_wrap else {"disable": True},
                use_torch_compile=False,
                use_remove_padding=args.packed,
                mixed_precision={"param_dtype": "bf16", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
            )
            engine = OuroFSDPEngine(
                model_config,
                config,
                FSDPOptimizerConfig(lr=1e-6, total_training_steps=4, override_optimizer_config={"foreach": False}),
                CheckpointConfig(),
            )
            engine.initialize()
            engine.module.eval()
            with torch.no_grad():
                _, result = engine.forward_step(batch, None, forward_only=True)
            old = result["model_output"]["log_probs"].values().float()
            behavior = batch["rollout_log_probs"].values()
            mask = batch["response_mask"].values()
            difference = (old - behavior)[mask]
            ratios = difference.exp()
            report["rollout_replay"] = {
                "max_abs_selected_logprob": difference.abs().max().item(),
                "ratio_min": ratios.min().item(),
                "ratio_max": ratios.max().item(),
                "fraction_outside_0.8_1.2": ((ratios < 0.8) | (ratios > 1.2)).float().mean().item(),
            }
            loss_config = OmegaConf.create(
                {
                    "clip_ratio": 0.2,
                    "clip_ratio_low": None,
                    "clip_ratio_high": None,
                    "clip_ratio_c": 3.0,
                    "global_batch_info": {},
                }
            )
            # Preserve the observed behavior likelihoods; do not hide kernel drift
            # by substituting replay likelihoods as if they came from the sampler.
            advantages = torch.where(torch.arange(old.numel(), device="cuda") % 2 == 0, 1.0, -1.0)
            is_weights = None
            if args.rollout_correction:
                is_weights, _ = compute_rollout_correction_weights(
                    (old - behavior)[None], mask[None], rollout_is="token", rollout_is_threshold=2.0
                )
                report["correction"] = {
                    "mode": "verl token TIS, cap=2, detached; biased low-variance experiment",
                    "weight_min": is_weights[0, mask].min().item(),
                    "weight_max": is_weights[0, mask].max().item(),
                    "note": "Preserves rollout probabilities; does not establish kernel parity or RL convergence",
                }

            def objective(model_output, data, dp_group):
                current = model_output["log_probs"].values().float()
                return compute_policy_loss_vanilla(
                    (old if args.rollout_correction else behavior)[None],
                    current[None],
                    advantages[None],
                    mask[None],
                    config=loss_config,
                    rollout_is_weights=is_weights,
                )

            report["steps"] = []
            watched = engine.module.module.model.layers[0].self_attn.q_proj.weight
            original = watched.detach().clone()
            torch.cuda.reset_peak_memory_stats()
            for step in range(2):
                engine.module.train()
                engine.optimizer_zero_grad()
                loss, result = engine.forward_step(batch, objective, forward_only=False)
                assert torch.isfinite(loss)
                loss.backward()
                norm = engine.optimizer_step()
                engine.lr_scheduler.step()
                report["steps"].append({"loss": loss.item(), "grad_norm": float(norm), "metrics": result["metrics"]})
                assert torch.isfinite(torch.as_tensor(norm))
                engine.optimizer_zero_grad()
                state = engine.module.state_dict()
                llm.start_weight_update(step + 1)
                llm.update_weights((name, tensor.detach().to(torch.bfloat16)) for name, tensor in state.items())
                llm.finish_weight_update()
                del state
                assert llm.get_weight_version() == step + 1
            report["weight_changed"] = not torch.equal(original, watched)
            assert report["weight_changed"]
            updated = llm.generate(prompts, params)
            report["rollouts_after"] = [asdict(output) for output in updated]
            assert all(output.weight_version == 2 for output in updated)
            # Fresh engine after updates must see the same physical checkpoint.
            from vllm_rlt.models.ouro import OuroForCausalLM as NativeModel

            fresh_model = NativeModel(llm.engine.model.config).to(device="cuda", dtype=torch.bfloat16)
            fresh_model.load_state_dict(llm.engine.model.state_dict(), strict=True)
            fresh = LLM(
                fresh_model,
                attention_backend="triton",
                cache_config=CacheConfig(num_blocks=128),
                scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=64),
            )
            fresh_outputs = fresh.generate(prompts, params)
            report["weight_sync_fresh_equal"] = all(
                a.token_ids == b.token_ids and a.exit_depths == b.exit_depths and a.log_probs == b.log_probs
                for a, b in zip(updated, fresh_outputs, strict=True)
            )
            assert report["weight_sync_fresh_equal"]
            fresh.close()
            del fresh, fresh_model
            if args.checkpoint:
                engine.checkpoint_manager.save_checkpoint(args.checkpoint, global_step=2)
                report["checkpoint_saved"] = args.checkpoint

                def continuation(engine):
                    engine.module.train()
                    engine.optimizer_zero_grad()
                    loss, _ = engine.forward_step(batch, objective, forward_only=False)
                    loss.backward()
                    engine.optimizer_step()
                    engine.lr_scheduler.step()
                    engine.optimizer_zero_grad()
                    return loss.item(), tensor_state_digest(engine)

                uninterrupted = continuation(engine)
                engine.checkpoint_manager.load_checkpoint(args.checkpoint)
                resumed = continuation(engine)
                report["checkpoint_resume_exact"] = uninterrupted == resumed
                report["resume_loss_and_state_sha256"] = resumed
                assert report["checkpoint_resume_exact"]
            report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            del engine
            gc.collect()
        finally:
            llm.close()
            dist.destroy_process_group()
    report["seconds"] = time.monotonic() - started
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
