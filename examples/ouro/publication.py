# SPDX-License-Identifier: Apache-2.0
"""Streaming full-weight publication for the pinned vllm-rlt training contract."""

from collections.abc import Iterable, Mapping

import torch


class OuroWeightPublisher:
    """Publish physical Ouro parameters once each, preserving their names.

    ``weights`` must yield full, plain floating tensors (for example the non-LoRA
    ``get_per_tensor_param`` generator), rather than FSDP shards, buffers, or
    expanded recurrent-loop weights. An optional caller-supplied manifest maps
    physical parameter names to shapes; the backend also validates full coverage.
    No additional engine inspection API is required.

    Each backend call receives at most ``bucket_bytes`` of detached tensors,
    except a single larger physical tensor, which is sent alone. This is a soft
    cap: the producer's current tensor can coexist with the previous bucket and
    dtype conversion can allocate one additional tensor. The publisher never
    stages a complete model on GPU. Publication requires an idle backend and a
    serialized caller; FSDP collectives must be consumed by all participating ranks.

    A fresh step zero publishes version one; ordinary steps publish step + 1.
    Retries or repeated publication at the same step consume a newer version.
    Earlier committed steps are rejected. A failed start, transfer, iterator, or
    commit consumes its attempted version and preserves native transaction fences
    on uncommitted peers; recovery requires a new *full* stream. The caller must
    keep shared admission closed after a PD error: a peer that already committed
    may have cleared its local fence. There is no rollback or finish on the error
    path. Checkpoint both counters, since PD peers may partially commit an attempted
    version without the coordinator acknowledging it.
    """

    def __init__(
        self,
        backend,
        *,
        bucket_bytes: int = 64 * 1024 * 1024,
        use_bf16: bool = True,
        expected_manifest: Mapping[str, tuple[int, ...]] | None = None,
    ):
        if type(bucket_bytes) is not int or bucket_bytes <= 0:
            raise ValueError("bucket_bytes must be a positive integer")
        if type(use_bf16) is not bool:
            raise ValueError("use_bf16 must be a boolean")
        self.backend = backend
        self.bucket_bytes = bucket_bytes
        self.use_bf16 = use_bf16
        self.expected_manifest = None
        if expected_manifest is not None:
            self.expected_manifest = {}
            for name, shape in expected_manifest.items():
                shape = tuple(shape)
                if not isinstance(name, str) or not name or any(type(dim) is not int or dim < 0 for dim in shape):
                    raise ValueError("manifest must map physical names to nonnegative integer shapes")
                self.expected_manifest[name] = shape
        self.committed_version = self._backend_version()
        self.last_attempted_version = self.committed_version
        self.global_step: int | None = None

    def _backend_version(self) -> int:
        version = self.backend.get_weight_version()
        if type(version) is not int or version < 0:
            raise ValueError("backend weight version must be a nonnegative integer")
        return version

    def state_dict(self) -> dict:
        """Counters only; model/optimizer checkpointing belongs to the trainer."""
        return {
            "last_attempted_version": self.last_attempted_version,
            "committed_version": self.committed_version,
            "global_step": self.global_step,
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore counters before republishing onto a fresh or existing backend.

        A fresh backend can still be at version zero. Restore does not claim its
        weights match the checkpoint; the next full publication confirms a new
        backend version. Persist state after failures as well as successful commits.
        Loading an older checkpoint never rewinds attempts observed by this instance.
        """
        if set(state) != {"last_attempted_version", "committed_version", "global_step"}:
            raise ValueError("publisher state must contain all three publication counters")
        attempted, committed, step = (
            state["last_attempted_version"],
            state["committed_version"],
            state["global_step"],
        )
        if (
            type(attempted) is not int
            or type(committed) is not int
            or not 0 <= committed <= attempted
            or (step is not None and (type(step) is not int or not 0 <= step < committed))
        ):
            raise ValueError("invalid publisher state counters")
        self.last_attempted_version = max(self.last_attempted_version, attempted)
        self.committed_version, self.global_step = committed, step

    def publish(self, weights: Iterable[tuple[str, torch.Tensor]], *, global_step: int) -> int:
        """Consume a complete tensor generator and return its confirmed version."""
        if type(global_step) is not int or global_step < 0:
            raise ValueError("global_step must be a nonnegative integer")
        if self.global_step is not None and global_step < self.global_step:
            raise ValueError("global_step must not precede the last committed global_step")
        version = max(global_step + 1, self.last_attempted_version + 1, self._backend_version() + 1)
        # Reserve before start: a failed PD start may already have reached a peer.
        self.last_attempted_version = version
        self.backend.start_weight_update(version=version)
        bucket, bucket_size, seen = [], 0, set()
        for name, tensor in weights:
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError(f"invalid or duplicate physical weight name: {name}")
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"physical weight must be a full tensor: {name}")
            tensor = tensor.detach()
            if type(tensor) is not torch.Tensor:
                raise ValueError(f"physical weight must be a plain full tensor: {name}")
            if not tensor.is_floating_point():
                raise ValueError(f"physical weight must be floating point: {name}")
            if self.expected_manifest is not None:
                if name not in self.expected_manifest or tuple(tensor.shape) != self.expected_manifest[name]:
                    raise ValueError(f"physical weight name or shape mismatch: {name}")
            seen.add(name)
            size = tensor.numel() * (2 if self.use_bf16 else tensor.element_size())
            if bucket and bucket_size + size > self.bucket_bytes:
                self.backend.update_weights(bucket)
                bucket, bucket_size = [], 0
            if self.use_bf16:
                tensor = tensor.to(dtype=torch.bfloat16)
            bucket.append((name, tensor))
            bucket_size += size
            if bucket_size >= self.bucket_bytes:
                self.backend.update_weights(bucket)
                bucket, bucket_size = [], 0
        if not seen:
            raise ValueError("full publication must include physical weights")
        if self.expected_manifest is not None and seen != self.expected_manifest.keys():
            raise ValueError("full publication must include every physical weight in the manifest")
        if bucket:
            self.backend.update_weights(bucket)
        self.backend.finish_weight_update()
        committed = self._backend_version()
        if committed != version:
            raise RuntimeError(f"backend committed weight version {committed}, expected {version}")
        self.committed_version, self.global_step = version, global_step
        return version
