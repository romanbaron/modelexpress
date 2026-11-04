# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local mirror strategy: reload the weights by swapping in a mirror's copy.

A NIXL mirror (examples/local_nixl_mirror/nixl_mirror.py) on the same GPU
holds a byte copy of the memory a READY MX source registered: the weights in
their final, processed layout, at the source's addresses. Every replica
restored from one CRIU snapshot has the same weights allocations at the same
addresses, so on a reload each weights allocation whose address and size
match a mirror region gets that region mapped behind its address in place of
its own memory. Nothing is loaded, copied or processed, tensors keep their
pointers, and CUDA graphs captured against them stay valid.

The reload follows vLLM's usual order after a level-2 sleep: wake_up(weights)
(which gives the weights fresh memory and restores the buffers saved at
sleep), then reload_weights. The engine's own memory is unmapped, not
released, until every swap succeeded, so a failure maps it back unchanged;
on success it is released.

What does not match -- an allocation the mirror has no region for, or a
region the source recreated elsewhere -- is finished by tensor name from the
mirror's copy (local_mirror.finish_by_name).

The mirror hands out one POSIX FD per region on a unix socket
(MX_LOCAL_MIRROR_RELOAD_SOCKET, {rank} = local GPU ordinal), labelled with the
source address it copied, and opens that socket only once its copy is
complete. A missing socket means no mirror, or one still copying: the
strategy is skipped, not waited for.

On failure the strategy restores the engine's own memory and asks the mirror
to free its copy, so the GPU has room for the KV cache, and the next strategy
loads the weights.
"""

from __future__ import annotations

import logging
import os
import socket
import time

from .. import envs, local_gpu
from ..adapter import EngineAdapter, StrategyFailed
from .base import LoadContext, LoadStrategy, _as_load_result
from .context import LoadResult

logger = logging.getLogger("modelexpress.strategy_local_mirror")


def _query(path: str) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as sock:
        sock.settimeout(10)
        sock.connect(path)
        local_gpu.send_msg(sock, {"want": "memory"})
        msg, _ = local_gpu.recv_msg(sock)
    return msg


class LocalMirrorStrategy(LoadStrategy):
    """Swap a local mirror's copy in behind the engine's weights."""

    name = "local_mirror"
    requires = (
        EngineAdapter.weight_allocations,
        EngineAdapter.swap_weight_allocation,
        EngineAdapter.restore_weight_allocation,
    )

    # The weights arrive processed, at their own addresses: no layerwise
    # reload to enter or leave, no post-processing, and no registration --
    # the mirror's NIXL agent already serves this memory.
    def prepare(self, result: LoadResult, ctx: LoadContext) -> None:
        pass

    def finalize(self, result: LoadResult, ctx: LoadContext) -> None:
        pass

    def abort(self, result: LoadResult, ctx: LoadContext) -> None:
        pass

    @staticmethod
    def _socket(ctx: LoadContext) -> str:
        return envs.MX_LOCAL_MIRROR_RELOAD_SOCKET.format(rank=ctx.device_id)

    def is_available(self, ctx: LoadContext) -> bool:
        if not envs.MX_LOCAL_MIRROR_RELOAD_SOCKET or not ctx.is_reload:
            return False
        if not super().is_available(ctx):
            return False
        if not ctx.adapter.weight_allocations():
            logger.info(
                f"[Worker {ctx.global_rank}] No awake weights allocations, local_mirror skipped"
            )
            return False
        path = self._socket(ctx)
        try:
            msg = _query(path)
        except OSError as e:
            logger.info(f"[Worker {ctx.global_rank}] No local mirror at {path} ({e}), skipped")
            return False
        if msg.get("gpu_uuid") != local_gpu.gpu_uuid(ctx.device_id):
            logger.warning(
                f"[Worker {ctx.global_rank}] Local mirror at {path} is on "
                f"{msg.get('gpu_uuid')}, not on this GPU; skipped"
            )
            return False
        if not msg.get("memory_bytes"):
            logger.info(f"[Worker {ctx.global_rank}] Local mirror at {path} holds nothing, skipped")
            return False
        return True

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        from ..local_mirror import _receive_regions, finish_by_name, release_mirror

        result = _as_load_result(result)
        path = self._socket(ctx)
        t0 = time.perf_counter()
        stats: dict = {"socket": path}
        swapped: list = []  # (token, the engine's own handle)
        mapped_sizes: dict[int, int] = {}
        regions: dict = {}
        try:
            header, regions = _receive_regions(path, ctx.device_id)
            stats["receive_s"] = round(time.perf_counter() - t0, 3)

            t = time.perf_counter()
            local_gpu.use_device(ctx.device_id)
            # Nothing may still be using the memory about to be unmapped.
            local_gpu.synchronize()
            left = 0
            for addr, size, token in ctx.adapter.weight_allocations():
                region = regions.get(addr)
                if region is None or region[0] != size or region[1] < size:
                    left += 1
                    continue
                handle = local_gpu.import_fd(region[2])
                try:
                    own = ctx.adapter.swap_weight_allocation(token, handle)
                except BaseException:
                    local_gpu.release(handle)
                    raise
                swapped.append((token, own))
                mapped_sizes[addr] = size
            local_gpu.synchronize()
            pending = {a: r for a, r in regions.items() if a not in mapped_sizes}
            stats.update(
                swapped=len(swapped),
                swapped_bytes=sum(mapped_sizes.values()),
                unmatched_allocations=left,
                unmatched_regions=len(pending),
                swap_s=round(time.perf_counter() - t, 3),
            )

            if left or pending:
                # Unmatched allocations keep the engine's own memory, which
                # the finish writes into.
                stats.update(
                    finish_by_name(
                        result.model, ctx, header["source_agent"], mapped_sizes, pending
                    )
                )
        except Exception as e:
            for token, own in reversed(swapped):
                try:
                    ctx.adapter.restore_weight_allocation(token, own)
                except Exception as undo:
                    logger.warning(
                        f"[Worker {ctx.global_rank}] Restoring a weights allocation "
                        f"failed: {undo}"
                    )
            try:
                release_mirror(path)
            except OSError as release_error:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Releasing the mirror at {path} "
                    f"failed: {release_error}"
                )
            raise StrategyFailed(f"local mirror at {path}: {e}", mutated=False) from e
        finally:
            for *_, fd in regions.values():
                os.close(fd)

        # The swap is final: free the engine's own copy.
        for _, own in swapped:
            local_gpu.release(own)
        stats["total_s"] = round(time.perf_counter() - t0, 3)
        logger.info(f"[Worker {ctx.global_rank}] [TIMING] local_mirror: {stats}")
        _record(ctx, stats)
        # The mirror's own agent serves this memory: nothing to register.
        result.publishable = False
        return result


def _record(ctx: LoadContext, stats: dict) -> None:
    """Also write the stats next to a snapshot restore's control files, when
    there is a control directory: a restored process's log may be lost."""
    directory = os.environ.get("SNAPSHOT_CONTROL_DIR", "/snapshot-control")
    if not os.path.isdir(directory):
        return
    try:
        with open(os.path.join(directory, f"vllm-restore-mirror-{ctx.global_rank}"), "w") as f:
            f.write(f"{stats}\n")
    except OSError:
        pass
