# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local mirror strategy: reload sleeping weights by mapping a mirror's copy.

A NIXL mirror (examples/local_nixl_mirror/nixl_mirror.py) on the same GPU holds a byte
copy of the memory a READY MX source registered: the weights in their final,
processed layout, at the source's addresses. Every replica restored from one
CRIU snapshot has the same weights allocations at the same addresses, so on a
reload after a level-2 sleep each sleeping allocation whose address and size
match a mirror region gets that region mapped at its reserved address.
Nothing is loaded, copied or processed, tensors keep their pointers, and CUDA
graphs captured against them stay valid.

What does not match -- an allocation the mirror has no region for, or a
region the source recreated elsewhere -- is finished by tensor name from the
mirror's copy (local_mirror.finish_by_name).

The mirror hands out one POSIX FD per region on a unix socket
(MX_LOCAL_MIRROR_RELOAD_SOCKET, {rank} = local GPU ordinal), labelled with the
source address it copied, and opens that socket only once its copy is
complete. A missing socket means no mirror, or one still copying: the
strategy is skipped, not waited for.

On failure the strategy unmaps what it mapped and asks the mirror to free its
copy, so the next strategy has room for the engine's own weights; the chain
then wakes the weights before trying it.
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
    """Map a local mirror's copy into the engine's sleeping weights."""

    name = "local_mirror"
    requires = (
        EngineAdapter.sleeping_weight_allocations,
        EngineAdapter.map_weight_allocation,
        EngineAdapter.unmap_weight_allocation,
    )
    # Maps into the sleeping allocations itself: it must see them asleep.
    accepts_sleeping_weights = True

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
        if not ctx.adapter.sleeping_weight_allocations():
            logger.info(f"[Worker {ctx.global_rank}] No sleeping weights, local_mirror skipped")
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
        mapped: list = []
        mapped_sizes: dict[int, int] = {}
        regions: dict = {}
        try:
            header, regions = _receive_regions(path, ctx.device_id)
            stats["receive_s"] = round(time.perf_counter() - t0, 3)

            t = time.perf_counter()
            local_gpu.use_device(ctx.device_id)
            left = 0
            for addr, size, token in ctx.adapter.sleeping_weight_allocations():
                region = regions.get(addr)
                if region is None or region[0] != size or region[1] < size:
                    left += 1
                    continue
                handle = local_gpu.import_fd(region[2])
                try:
                    ctx.adapter.map_weight_allocation(token, handle)
                except BaseException:
                    local_gpu.release(handle)
                    raise
                mapped.append(token)
                mapped_sizes[addr] = size
            local_gpu.synchronize()
            pending = {a: r for a, r in regions.items() if a not in mapped_sizes}
            stats.update(
                mapped=len(mapped),
                mapped_bytes=sum(mapped_sizes.values()),
                weights_left_asleep=left,
                unmatched_regions=len(pending),
                map_s=round(time.perf_counter() - t, 3),
            )

            if left or pending:
                # The finish writes into allocations that were not mapped, so
                # they need their own memory first.
                if left:
                    ctx.adapter.wake_sleeping_weights()
                stats.update(
                    finish_by_name(
                        result.model, ctx, header["source_agent"], mapped_sizes, pending
                    )
                )
        except Exception as e:
            for token in mapped:
                try:
                    ctx.adapter.unmap_weight_allocation(token)
                except Exception as undo:
                    logger.warning(f"[Worker {ctx.global_rank}] Unmapping failed: {undo}")
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
