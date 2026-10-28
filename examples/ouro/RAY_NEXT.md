# Ray integration boundary — 2026-10-28

The exercised Ouro recipe uses torchrun and real verl training/rollout APIs.
It does not register a production Ray rollout replica. The current GPU report
must not be read as AgentLoop acceptance.

## Reuse the actual framework entry points

At the pinned contributor-fork baseline, `workers/rollout/base.py` resolves
`BaseRollout` implementations, while `workers/rollout/replica.py` maintains a
separate `RolloutReplicaRegistry`. Registering the existing OuroRollout in the
first registry does not implement the second. `get_rollout_replica_class` has
no `vllm_rlt` loader; PD dispatch explicitly accepts only vllm and sglang.

`experimental/agent_loop/agent_loop.py:AgentLoopManager` consumes an
`LLMServerClient`. Its token-in/token-out call in
`workers/rollout/llm_server.py` awaits `server.generate.remote(...)` and expects
`TokenOutput`, whose `extra_fields` can carry the recurrent execution metadata.
The next adapter should reuse these owners rather than create another trainer.

| Boundary | Existing reusable part | Missing production acceptance |
| --- | --- | --- |
| Replica allocation | RolloutReplica hybrid/colocated/standalone resource-pool methods | A registered native-RLT replica that launches and closes its actual server |
| Token requests | StageClient/PDEngine public add/step/abort; TokenBatchDriver | Async server.generate and cancellation while generation is in flight |
| Training metadata | Response-aligned token scores, depths, effective parameters, committed version | Preserve these through TokenOutput.extra_fields and the AgentLoop data conversion |
| Publication | OuroWeightPublisher full streams and failure fencing | Drain/admission coordination across actor calls and every replica acknowledgement |
| Recovery | Explicit profile, sampler cursor and publisher counters | Actor restart, stale-result rejection and republication before accepting requests |
| Memory switching | Adapter release/resume fences admission | Engine sleep/wake or KV release; existing release() does not free GPU memory |

## Contracts to preserve

- Carry the actual `exit_depths`, `sampling_params`, `weight_version`, selected
  raw scores and probability-profile provenance across the server boundary.
  Keep prompt prefill full-depth and shift output traces onto training inputs
  exactly once using the existing converter.
- Expose trainer `global_steps` separately from the committed weight version.
  A publication retry consumes another version, so `weight_version - 1` is not
  a reliable reconstruction of the optimizer step.
- One execution owner must serialize engine control and step calls. A blocking
  generate method in an async Ray actor would prevent abort/update calls from
  progressing. A threaded wrapper also needs explicit pipe/profiler/CUDA-owner
  ordering; merely calling to_thread is not a lifecycle design.
- A failed update must keep generation closed until every required replica has
  completed a fresh full publication. Never stamp a synthetic version onto an
  output or mutate model weights through a private worker handle.
- Reuse request-generation identity, including across actor restarts; a reused
  external request ID cannot authorize a late result from an older generation.

## First production validation increment

Implement the standalone one-GPU replica first, test real AgentLoop token
generation/abort/version metadata and actor cleanup, then connect one full
FSDP update and a fresh-process restore. Add P/D as a distinct topology after
that boundary works. The current two-GPU machine can test isolated dual-rank
training and isolated P/D sequentially. Simultaneous two-rank training with a
dedicated rollout needs three GPUs; dedicated P and D plus those ranks needs
four. Shared-device deployment instead requires explicitly tested memory and
execution coordination.

Unified #83 rollout naming plus #78 publication is still an upstream dependency.
The explicit publication_outputs experiment profile does not settle that API
decision. Full Ray acceptance, changed-world-size recovery and production
throughput remain unverified.
