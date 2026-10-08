# SPDX-License-Identifier: Apache-2.0
"""Reproducible Ouro BF16 verl/FSDP training with fresh versioned rollouts.

Run with torchrun on one/two GPUs. Uses verl BaseRollout, FSDPEngine, standard
PPO loss, GRPO advantages and checkpoint manager. Ray AgentLoop integration is
outside this contributor-fork recipe. See README for pins and acceptance scope.
"""

import argparse
import asyncio
import functools
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from examples.ouro.recipe_utils import (
    arithmetic_reward,
    check_probability_envelope,
    scheduled_examples,
    validate_resume_config,
)
from examples.ouro.rollout import OuroRollout
from examples.ouro.worker import OuroFSDPEngine, training_batch_from_rollout
from verl import DataProto
from verl.trainer.config import CheckpointConfig
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig, FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig, RolloutConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

MODEL_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
ENGINE_REVISION = "bdce2876e3f78ac7efaa6cbf5783c185da2e9d19"


def shard_fingerprint(engine):
    """Hash local physical shards, Adam moments/steps and scheduler state."""
    digest = hashlib.sha256()
    for name, parameter in engine.module.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        for key, value in sorted(engine.optimizer.state.get(parameter, {}).items()):
            digest.update(key.encode())
            if isinstance(value, torch.Tensor):
                digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                digest.update(str(value).encode())
    digest.update(json.dumps(engine.lr_scheduler.state_dict(), sort_keys=True).encode())
    return digest.hexdigest()


def prompt_batch(tokenizer, examples, samples):
    repeated = [example for example in examples for _ in range(samples)]
    token_ids = [tokenizer.encode(example["prompt"], add_special_tokens=False) for example in repeated]
    width = max(map(len, token_ids))
    ids = torch.full((len(repeated), width), tokenizer.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, tokens in enumerate(token_ids):
        ids[index, -len(tokens) :] = torch.tensor(tokens)
        mask[index, -len(tokens) :] = 1
    return DataProto.from_dict(
        tensors={"input_ids": ids, "attention_mask": mask},
        non_tensors={
            "uid": np.array([example["uid"] for example in repeated], dtype=object),
            "answer": np.array([example["answer"] for example in repeated], dtype=object),
        },
    )


def publish(engine, rollout, global_step, rank):
    """Every rank gathers; only the coordinator copies buckets into rollout."""
    started = time.monotonic()
    tensors, peft = engine.get_per_tensor_param()
    if peft is not None:
        raise ValueError("this recipe publishes full physical parameters, not LoRA")
    if rank == 0:
        asyncio.run(rollout.update_weights(tensors, global_steps=global_step))
        state = rollout.state_dict()
    else:
        # Consume per-unit all-gathers on every rank; exported tensors own their
        # BF16 storage after each temporary FSDP unshard context closes.
        for _ in tensors:
            pass
        state = None
    states = [state]
    dist.broadcast_object_list(states, src=0)
    dist.barrier()
    return states[0], time.monotonic() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--response-length", type=int, default=32)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--exit-threshold", type=float, default=1.0)
    parser.add_argument("--max-logprob-error", type=float, default=0.5)
    parser.add_argument("--min-behavior-ratio", type=float, default=0.5)
    parser.add_argument("--max-behavior-ratio", type=float, default=2.0)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-at", type=int, default=2)
    parser.add_argument("--resume")
    args = parser.parse_args()
    rank, world, local = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    total = args.prompts_per_step * args.samples_per_prompt
    if total % (world * args.micro_batch) or args.samples_per_prompt < 2:
        raise ValueError("global batch must divide rank/microbatch sizes; GRPO needs at least two samples")
    torch.cuda.set_device(local)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rollout = None
    config = {**vars(args), "world_size": world, "model_revision": MODEL_REVISION, "engine_revision": ENGINE_REVISION}
    report = {
        "config": config,
        "torch": torch.__version__,
        "iterations": [],
        "scope": "standard verl APIs, torchrun recipe",
    }
    report["revision_labels"] = (
        "Required revisions; actual runtime source/model hashes are recorded with the experiment artifacts"
    )
    try:
        model_config = HFModelConfig(
            path=args.model,
            load_tokenizer=False,
            external_lib="examples.ouro.hf",
            use_remove_padding=True,
            override_config={"attn_implementation": "eager"},
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        engine_cls = EngineRegistry.get_engine_cls("ouro", "fsdp")
        assert engine_cls is OuroFSDPEngine
        engine = engine_cls(
            model_config,
            FSDPEngineConfig(
                use_orig_params=True,
                use_remove_padding=True,
                use_torch_compile=False,
                use_no_sync_for_gradient_accumulation=False,
                mixed_precision={"param_dtype": "bf16", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
            ),
            FSDPOptimizerConfig(
                lr=args.lr, total_training_steps=args.steps, override_optimizer_config={"foreach": False}
            ),
            CheckpointConfig(),
        )
        engine.initialize()
        start_step, saved_state = 0, None
        if args.resume:
            saved = json.loads((Path(args.resume) / "trainer.json").read_text())
            validate_resume_config(saved["config"], config)
            start_step, saved_state = saved["global_step"], saved["rollout"]
            engine.load_checkpoint(args.resume, del_local_after_load=False)
        if rank == 0:
            rc = RolloutConfig(
                name="vllm_rlt",
                mode="async",
                tensor_model_parallel_size=1,
                response_length=args.response_length,
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                seed=args.seed,
                max_num_seqs=2,
                ignore_eos=False,
                engine_kwargs={
                    "vllm_rlt": {
                        "cache": {"num_blocks": 128},
                        "scheduler": {"max_num_seqs": 2, "max_num_batched_tokens": 64},
                        "sampling": {"exit_threshold": args.exit_threshold},
                        "bucket_bytes": 32 << 20,
                    }
                },
            )
            rollout = OuroRollout(rc, model_config, None, device=f"cuda:{local}")
            if saved_state:
                rollout.load_state_dict(saved_state)
        rollout_state, seconds = publish(engine, rollout, start_step, rank)
        report["initial_publication"] = {"version": rollout_state["publisher"]["committed_version"], "seconds": seconds}
        local_parameter_count = torch.tensor(sum(p.numel() for p in engine.module.parameters()), device=f"cuda:{local}")
        dist.all_reduce(local_parameter_count)
        report["physical_parameters"] = local_parameter_count.item()
        actor_config = ActorConfig(
            strategy="fsdp",
            rollout_n=args.samples_per_prompt,
            ppo_micro_batch_size_per_gpu=args.micro_batch,
            ppo_mini_batch_size=total,
            use_torch_compile=False,
            entropy_coeff=0.0,
            use_kl_loss=False,
        )
        objective = functools.partial(ppo_loss, actor_config)
        for step in range(start_step, args.steps):
            started = time.monotonic()
            payload = None
            if rank == 0:
                data = rollout.generate_sequences(
                    prompt_batch(
                        tokenizer, scheduled_examples(step=step, count=args.prompts_per_step), args.samples_per_prompt
                    )
                )
                texts = [
                    tokenizer.decode(output.token_ids, skip_special_tokens=True)
                    for output in data.non_tensor_batch["rlt_outputs"]
                ]
                scores = [
                    arithmetic_reward(text, answer)
                    for text, answer in zip(texts, data.non_tensor_batch["answer"], strict=True)
                ]
                mask = data.batch["response_mask"]
                token_rewards = torch.zeros_like(data.batch["rollout_log_probs"])
                for index, reward in enumerate(scores):
                    token_rewards[index, int(mask[index].sum()) - 1] = reward
                advantages, _ = compute_grpo_outcome_advantage(token_rewards, mask, data.non_tensor_batch["uid"])
                payload = (data, advantages, scores, texts)
            objects = [payload]
            dist.broadcast_object_list(objects, src=0)
            data, advantages, rewards, texts = objects[0]
            if not bool(advantages[data.batch["response_mask"]].abs().gt(0).any()):
                raise RuntimeError("acceptance batch has no nonzero GRPO advantage; increase task/sample diversity")
            selection = list(range(rank, total, world))
            local_data = data[selection]
            global_tokens = int(data.batch["response_mask"].sum())
            version = rollout_state["publisher"]["committed_version"]
            batch = training_batch_from_rollout(
                local_data,
                advantages=advantages[selection],
                expected_version=version,
                world_size=world,
                global_batch_size=total,
                global_token_count=global_tokens,
                device=f"cuda:{local}",
                packed=True,
            )
            for key, value in (
                ("use_dynamic_bsz", False),
                ("micro_batch_size_per_gpu", args.micro_batch),
                ("return_model_output", True),
            ):
                tu.assign_non_tensor_data(batch, key, value)
            engine.module.eval()
            with torch.no_grad():
                inferred = engine.forward_backward_batch(batch, None, forward_only=True)
            old = no_padding_2_padding(inferred["model_output"]["log_probs"], batch).float()
            behavior = torch.nested.to_padded_tensor(batch["rollout_log_probs"], 0.0, output_size=old.shape)
            response_mask = torch.nested.to_padded_tensor(batch["response_mask"], False, output_size=old.shape)
            check_probability_envelope(
                old,
                behavior,
                response_mask,
                max_abs_logprob=args.max_logprob_error,
                min_ratio=args.min_behavior_ratio,
                max_ratio=args.max_behavior_ratio,
            )
            weights, _ = compute_rollout_correction_weights(
                old - behavior, response_mask, rollout_is="token", rollout_is_threshold=2.0
            )
            lengths = batch["responses"].offsets().diff().tolist()
            offsets = batch["responses"].offsets()
            batch["old_log_probs"] = torch.nested.nested_tensor_from_jagged(
                torch.cat([old[i, :size] for i, size in enumerate(lengths)]), offsets
            )
            batch["rollout_is_weights"] = torch.nested.nested_tensor_from_jagged(
                torch.cat([weights[i, :size] for i, size in enumerate(lengths)]), offsets
            )
            delta = (old - behavior)[response_mask]
            stats = torch.tensor([delta.abs().max(), delta.exp().min(), delta.exp().max()], device=f"cuda:{local}")
            gathered_stats = [torch.empty_like(stats) for _ in range(world)]
            dist.all_gather(gathered_stats, stats)
            engine.module.train()
            engine.optimizer_zero_grad()
            tu.assign_non_tensor_data(batch, "return_model_output", False)
            losses = engine.forward_backward_batch(batch, objective, forward_only=False)
            norm = engine.optimizer_step()
            assert torch.isfinite(torch.as_tensor(norm)) and float(norm) > 0, (
                "RL update needs a nonzero finite gradient"
            )
            engine.lr_scheduler.step()
            engine.optimizer_zero_grad()
            fp32_state = all(
                p.dtype == torch.float32 and engine.optimizer.state[p]["exp_avg"].dtype == torch.float32
                for p in engine.module.parameters()
                if p.requires_grad and p.numel()
            )
            assert fp32_state, "master weights and Adam moments must remain FP32"
            rollout_state, publish_seconds = publish(engine, rollout, step + 1, rank)
            hashes = [None] * world
            dist.all_gather_object(hashes, shard_fingerprint(engine))
            row = {
                "global_step": step + 1,
                "rollout_version": version,
                "published_version": rollout_state["publisher"]["committed_version"],
                "rewards": rewards,
                "texts": texts,
                "token_ids": [out.token_ids for out in data.non_tensor_batch["rlt_outputs"]],
                "exit_depths": [out.exit_depths for out in data.non_tensor_batch["rlt_outputs"]],
                "sampling_seeds": data.batch["sampling_seed"].tolist(),
                "losses_rank0": losses["loss"],
                "grad_norm": float(norm),
                "shard_state_sha256": hashes,
                "probability_stats_by_rank": [s.tolist() for s in gathered_stats],
                "probability_envelope_passed": True,
                "fp32_master_and_adam": fp32_state,
                "publish_seconds": publish_seconds,
                "seconds": time.monotonic() - started,
                "peak_allocated_gib_rank0": torch.cuda.max_memory_allocated() / 2**30,
            }
            report["iterations"].append(row)
            if rank == 0:
                Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
            if args.checkpoint_dir and step + 1 == args.checkpoint_at:
                engine.save_checkpoint(args.checkpoint_dir, global_step=step + 1)
                if rank == 0:
                    (Path(args.checkpoint_dir) / "trainer.json").write_text(
                        json.dumps({"config": config, "global_step": step + 1, "rollout": rollout_state}, indent=2)
                    )
                dist.barrier()
    finally:
        if rollout is not None:
            rollout.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
