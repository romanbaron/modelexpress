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

The swap is plain CUDA and needs nothing from the engine: the engine's
memory at an address comes from cuMemRetainAllocationHandle, and the
mirror's is mapped in its place. The engine's allocator is not told, so it
still records the handle of the memory that was released. Until the
engine's next sleep or free of a swapped allocation nothing reads that
record; that sleep or free then releases a handle that is gone (in vLLM,
CuMemAllocator.sleep raises).

What does not match -- an allocation the mirror has no region for, or a
region the source recreated elsewhere -- is finished by tensor name from the
mirror's copy (_finish_by_name).

The mirror hands out one POSIX FD per region on a unix socket
(MX_LOCAL_MIRROR_RELOAD_SOCKET, {rank} = local GPU ordinal), labelled with the
source address it copied, and opens that socket only once its copy is
complete. A missing socket means no mirror, or one still copying: the
strategy is skipped, not waited for.

On failure the strategy restores the engine's own memory and asks the mirror
to free its copy, so the GPU has room for the KV cache, and the next strategy
loads the weights.

Besides the strategy, this module holds what it needs around it: the cuMem
driver calls PyTorch does not expose (through cuda-python, as vmm/backend.py
does), the message framing of the mirror's socket, and the mirror client
(receive the regions, finish by tensor name, release the mirror).
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import socket
import time
from typing import Any, NamedTuple

import torch

from .. import envs, p2p_pb2
from ..adapter import StrategyFailed
from .base import LoadContext, LoadStrategy, _as_load_result
from .context import LoadResult

logger = logging.getLogger("modelexpress.strategy_local_mirror")

# SCM_RIGHTS caps one message at 253 FDs.
_MAX_FDS_PER_MSG = 250
# Fragment payload, well under the default unix socket buffer.
_FRAGMENT_BYTES = 60_000

_driver: Any = None


# --- CUDA driver (cuda-python) -------------------------------------------------


def _cu():
    """cuda-python's driver module, initialized. Raises ImportError when
    cuda-python is not installed."""
    global _driver
    if _driver is None:
        from cuda.bindings import driver

        # Published only after cuInit returns: another thread that sees it
        # set must not call into the driver before it is initialized.
        _check("cuInit", driver.cuInit(0), driver)
        _driver = driver
    return _driver


def _check(name: str, result: tuple, driver: Any = None) -> Any:
    """Raise RuntimeError on a failed driver call; return its value, if any."""
    cu = driver or _driver
    err, *values = result
    if err != cu.CUresult.CUDA_SUCCESS:
        _, text = cu.cuGetErrorString(err)
        detail = text.decode() if isinstance(text, bytes) else str(text)
        raise RuntimeError(f"{name} failed: {int(err)} {detail}")
    return values[0] if values else None


def _call(name: str, *args: Any) -> Any:
    cu = _cu()
    return _check(name, getattr(cu, name)(*args))


def _gpu_uuid(ordinal: int) -> str:
    dev = _call("cuDeviceGet", ordinal)
    raw = _call("cuDeviceGetUuid", dev).bytes
    if isinstance(raw, str):
        raw = raw.encode("latin-1")
    h = bytes(raw).hex()
    return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def _import_fd(fd: int) -> int:
    cu = _cu()
    handle = _call(
        "cuMemImportFromShareableHandle",
        fd,
        cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    )
    return int(handle)


def _release(handle: int) -> None:
    _call("cuMemRelease", handle)


def _retain_handle(va: int) -> int | None:
    """A new reference to the cuMem allocation mapped at `va` (the same
    handle value it was mapped with), or None when no cuMem allocation is
    mapped there. Release it like any handle."""
    cu = _cu()
    err, handle = cu.cuMemRetainAllocationHandle(va)
    return int(handle) if err == cu.CUresult.CUDA_SUCCESS else None


def _reserve(size: int) -> int:
    return int(_call("cuMemAddressReserve", size, 0, 0, 0))


def _address_free(va: int, size: int) -> None:
    _call("cuMemAddressFree", va, size)


def _map_at(va: int, size: int, handle: int, ordinal: int) -> None:
    cu = _cu()
    _call("cuMemMap", va, size, 0, handle, 0)
    access = cu.CUmemAccessDesc()
    access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = ordinal
    access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _call("cuMemSetAccess", va, size, [access], 1)


def _unmap(va: int, size: int) -> None:
    _call("cuMemUnmap", va, size)


def _copy(dst: int, src: int, nbytes: int) -> None:
    """Device-to-device copy between raw addresses, in the current context."""
    _call("cuMemcpyDtoD", dst, src, nbytes)


# --- Mirror socket messages ----------------------------------------------------


def _send_msg(sock: socket.socket, msg: dict, fds: list[int] | tuple = ()) -> None:
    """Send one message; the FDs ride on its first fragment.

    Each fragment starts with a byte that is 1 when more fragments follow.
    """
    payload = json.dumps(msg).encode()
    chunks = [
        payload[i : i + _FRAGMENT_BYTES]
        for i in range(0, len(payload), _FRAGMENT_BYTES)
    ]
    for i, chunk in enumerate(chunks):
        more = b"\x01" if i + 1 < len(chunks) else b"\x00"
        socket.send_fds(sock, [more + chunk], list(fds) if i == 0 else [])


def _recv_msg(sock: socket.socket, maxfds: int = 0) -> tuple[dict, list[int]]:
    parts: list[bytes] = []
    all_fds: list[int] = []
    while True:
        data, fds, _, _ = socket.recv_fds(sock, _FRAGMENT_BYTES + 1, maxfds)
        if not data:
            raise ConnectionError("peer closed the local GPU socket")
        all_fds.extend(fds)
        parts.append(data[1:])
        if data[:1] == b"\x00":
            return json.loads(b"".join(parts)), all_fds


# --- Mirror client -------------------------------------------------------------


def _receive_regions(path: str, device: int):
    """The mirror's regions as {source_addr: (size, allocation_size, fd)},
    plus its header."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    sock.settimeout(60)
    regions: dict[int, tuple[int, int, int]] = {}
    try:
        sock.connect(path)
        _send_msg(sock, {"want": "regions"})
        while True:
            msg, fds = _recv_msg(sock, _MAX_FDS_PER_MSG)
            if "error" in msg:
                raise RuntimeError(f"mirror error: {msg['error']}")
            if len(fds) != len(msg["batch"]):
                raise RuntimeError(f"{len(fds)} FDs for {len(msg['batch'])} regions")
            for (addr, size, allocation_size), fd in zip(msg["batch"], fds):
                regions[addr] = (size, allocation_size, fd)
            if msg["last"]:
                break
    except BaseException:
        for *_, fd in regions.values():
            os.close(fd)
        raise
    finally:
        sock.close()
    if msg["gpu_uuid"] != _gpu_uuid(device):
        for *_, fd in regions.values():
            os.close(fd)
        raise RuntimeError(f"mirror is on {msg['gpu_uuid']}, not on this GPU")
    return msg, regions


def _source_manifest(ctx, source_agent: str) -> dict:
    """The tensor manifest of the MX source the mirror copied, found by the
    NIXL agent name among the READY sources of this model and rank."""
    from ..metadata.worker_server import fetch_tensor_manifest

    listed = ctx.mx_client.list_sources(
        identity=ctx.identity, status_filter=p2p_pb2.SOURCE_STATUS_READY
    )
    for inst in listed.instances:
        if inst.worker_rank != ctx.worker_rank:
            continue
        meta = ctx.mx_client.get_metadata(inst.mx_source_id, inst.worker_id)
        if not meta.found or meta.worker.agent_name != source_agent:
            continue
        protos, _ = fetch_tensor_manifest(
            meta.worker.worker_grpc_endpoint,
            inst.mx_source_id,
            timeout=30,
            worker_id=inst.worker_id,
        )
        return {t.name: t for t in protos}
    raise LookupError(f"no READY source with agent {source_agent} for this model and rank")


def _release_mirror(path: str) -> None:
    """Ask the mirror to free its copy (it keeps running, holding nothing).
    For a reload that cannot use the copy: the memory is needed for the
    engine's own weights."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as sock:
        sock.settimeout(60)
        sock.connect(path)
        _send_msg(sock, {"want": "release"})
        _recv_msg(sock)


def _finish_by_name(
    model: torch.nn.Module, ctx, source_agent: str, mapped: dict, pending: dict
) -> dict:
    """Fill every tensor of this engine that mapping did not put in place
    from the mirror's copy of the source tensor of the same name.

    mapped: {source_addr: size} of the mirror regions mapped at their own
    address. pending: {source_addr: (size, allocation_size, fd)} of the
    regions that matched no allocation; the caller owns and closes the FDs.

    Matched as an RDMA receive matches: every tensor this engine has must be
    in the source's manifest with the same size; tensors only the source has
    are skipped (a source that reloaded from disk can hold extra processed
    tensors, e.g. DeepSeek-V4's per-layer MoE scale transposes, that the
    snapshot's model never had).

    In place means: at the same address here as in the source, inside a
    mapped region. Everything else -- tensors the source recreated
    elsewhere, and tensors in allocations that were not mapped -- is read
    from the mirror's copy (a mapped region, or a pending one mapped for the
    copy) and written to this engine's tensor, which must have memory behind
    it. All reads go to a staging buffer before any write, because the
    destinations can lie in mapped mirror memory that later reads come from.
    Raises before the first write when a tensor of this engine has no source
    counterpart of the same size."""
    t0 = time.perf_counter()
    device = ctx.device_id
    manifest = _source_manifest(ctx, source_agent)
    # The same discovery the source's manifest came from: it also promotes
    # hidden tensors (e.g. MoE scales held by the quant config) to buffers.
    local = ctx.adapter.discover_tensors(LoadResult(value=model, model=model))

    mapped_starts = sorted(mapped)
    pending_starts = sorted(pending)

    def find(starts, sizes, addr, nbytes):
        i = bisect.bisect_right(starts, addr) - 1
        if i >= 0 and addr + nbytes <= starts[i] + sizes(starts[i]):
            return starts[i]
        return None

    copies, in_place = [], 0
    source_only = len(set(manifest) - set(local))
    for name, t in local.items():
        entry = manifest.get(name)
        if entry is None:
            raise LookupError(f"{name} of this model is not in the source manifest")
        nbytes = t.numel() * t.element_size()
        if nbytes == 0:
            continue
        if nbytes != entry.size:
            raise ValueError(f"{name}: source has {entry.size} bytes, this model {nbytes}")
        dst, src = t.data_ptr(), entry.addr
        mapped_at = find(mapped_starts, mapped.__getitem__, src, nbytes)
        if dst == src and mapped_at is not None:
            in_place += 1
            continue
        if mapped_at is not None:
            copies.append((name, dst, src, nbytes, None))
            continue
        region = find(pending_starts, lambda a: pending[a][0], src, nbytes)
        if region is None:
            raise LookupError(f"{name} at 0x{src:x} is in no mirror region")
        copies.append((name, dst, src, nbytes, region))

    torch.cuda.set_device(device)
    temp = {}  # pending region -> (va, allocation_size, handle)
    try:
        for *_, region in copies:
            if region is not None and region not in temp:
                _, allocation_size, fd = pending[region]
                handle = _import_fd(fd)
                va = _reserve(allocation_size)
                _map_at(va, allocation_size, handle, device)
                temp[region] = (va, allocation_size, handle)
        total = sum(nbytes for _, _, _, nbytes, _ in copies)
        if total:
            staging = torch.empty(total, dtype=torch.uint8, device=torch.device("cuda", device))
            base, offset, placed = staging.data_ptr(), 0, []
            for name, dst, src, nbytes, region in copies:
                va = src if region is None else temp[region][0] + (src - region)
                _copy(base + offset, va, nbytes)
                placed.append((dst, offset, nbytes))
                offset += nbytes
            torch.cuda.synchronize(device)
            for dst, off, nbytes in placed:
                _copy(dst, base + off, nbytes)
            torch.cuda.synchronize(device)
            del staging
    finally:
        for va, allocation_size, handle in temp.values():
            _unmap(va, allocation_size)
            _address_free(va, allocation_size)
            _release(handle)
    torch.cuda.empty_cache()
    return {
        "in_place": in_place,
        "copied": len(copies),
        "copied_bytes": sum(c[3] for c in copies),
        "source_only_skipped": source_only,
        "unmatched_regions_read": len(temp),
        "copied_examples": [c[0] for c in copies[:8]],
        "finish_s": round(time.perf_counter() - t0, 3),
    }


# --- Strategy ------------------------------------------------------------------


def _query(path: str) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as sock:
        sock.settimeout(10)
        sock.connect(path)
        _send_msg(sock, {"want": "memory"})
        msg, _ = _recv_msg(sock)
    return msg


class LocalMirrorStrategy(LoadStrategy):
    """Swap a local mirror's copy in behind the engine's weights."""

    name = "local_mirror"
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
        path = self._socket(ctx)
        try:
            msg = _query(path)
        except OSError as e:
            logger.info(f"[Worker {ctx.global_rank}] No local mirror at {path} ({e}), skipped")
            return False
        try:
            uuid = _gpu_uuid(ctx.device_id)
        except ImportError as e:
            logger.info(f"[Worker {ctx.global_rank}] cuda-python unavailable ({e}), skipped")
            return False
        if msg.get("gpu_uuid") != uuid:
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

        result = _as_load_result(result)
        path = self._socket(ctx)
        t0 = time.perf_counter()
        stats: dict = {"socket": path}
        swapped: list[_Swap] = []
        mapped_sizes: dict[int, int] = {}
        regions: dict = {}
        try:
            header, regions = _receive_regions(path, ctx.device_id)
            stats["receive_s"] = round(time.perf_counter() - t0, 3)

            t = time.perf_counter()
            torch.cuda.set_device(ctx.device_id)
            # Nothing may still be using the memory about to be unmapped.
            torch.cuda.synchronize(ctx.device_id)
            # Each region was copied from a source allocation of the same
            # address and size; it is swapped in where the engine has an awake
            # weights allocation exactly there.
            for addr, (size, allocation_size, fd) in regions.items():
                if allocation_size < size:
                    continue
                swap = _swap_in(ctx, addr, size, fd)
                if swap is not None:
                    swapped.append(swap)
                    mapped_sizes[addr] = size
            torch.cuda.synchronize(ctx.device_id)
            pending = {a: r for a, r in regions.items() if a not in mapped_sizes}
            stats.update(
                swapped=len(swapped),
                swapped_bytes=sum(mapped_sizes.values()),
                unmatched_regions=len(pending),
                swap_s=round(time.perf_counter() - t, 3),
            )
            if not swapped:
                # Not a replica of the mirror's source (no allocation at the
                # source's addresses): another strategy loads faster.
                raise RuntimeError("no mirror region matches a weights allocation")

            # Every tensor not in place -- in a weights allocation the mirror
            # had no region for, or recreated elsewhere by the source -- is
            # written from the mirror's copy, by name. This also verifies
            # that every tensor of the engine has its source counterpart.
            stats.update(
                _finish_by_name(result.model, ctx, header["source_agent"], mapped_sizes, pending)
            )
        except Exception as e:
            for swap in reversed(swapped):
                try:
                    _swap_back(ctx, swap)
                except Exception as undo:
                    logger.warning(
                        f"[Worker {ctx.global_rank}] Restoring a weights allocation "
                        f"failed: {undo}"
                    )
            try:
                _release_mirror(path)
            except OSError as release_error:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Releasing the mirror at {path} "
                    f"failed: {release_error}"
                )
            raise StrategyFailed(f"local mirror at {path}: {e}", mutated=False) from e
        finally:
            for *_, fd in regions.values():
                os.close(fd)

        # The swap is final: free the engine's own memory. retain_handle gave
        # back the very handle the engine created it with, so releasing it
        # twice drops both the engine's reference and the retained one. The
        # engine's allocator still records that handle (see the module
        # docstring).
        for swap in swapped:
            _release(swap.own)
            _release(swap.own)
        stats["total_s"] = round(time.perf_counter() - t0, 3)
        logger.info(f"[Worker {ctx.global_rank}] [TIMING] local_mirror: {stats}")
        _record(ctx, stats)
        # The mirror's own agent serves this memory: nothing to register.
        result.publishable = False
        return result


class _Swap(NamedTuple):
    address: int
    size: int
    own: int  # the engine's memory, unmapped, retained for a rollback


def _swap_in(ctx: LoadContext, address: int, size: int, fd: int) -> _Swap | None:
    """Map the mirror region `fd` behind the engine's memory mapped at
    address..address+size, in place of it. None, changing nothing, when the
    engine has no allocation mapped exactly there."""
    # The engine's memory at this address; None when nothing is mapped there
    # (no allocation, or one that is asleep).
    own = _retain_handle(address)
    if own is None:
        return None
    try:
        mirror = _import_fd(fd)
    except BaseException:
        _release(own)
        raise
    try:
        # cuMemUnmap only takes whole mappings: a failure means the engine's
        # allocation there is not exactly this region, and nothing changed.
        _unmap(address, size)
    except RuntimeError:
        _release(mirror)
        _release(own)
        return None
    try:
        _map_at(address, size, mirror, ctx.device_id)
    except BaseException:
        try:
            _unmap(address, size)
        except RuntimeError:
            pass
        _map_at(address, size, own, ctx.device_id)
        _release(mirror)
        _release(own)
        raise
    # The mapping keeps the mirror's memory alive; the engine's next unmap
    # of this address (sleep, free) drops it.
    _release(mirror)
    return _Swap(address, size, own)


def _swap_back(ctx: LoadContext, swap: _Swap) -> None:
    """Undo _swap_in: the engine's own memory is mapped back, unchanged."""
    _unmap(swap.address, swap.size)
    _map_at(swap.address, swap.size, swap.own, ctx.device_id)
    _release(swap.own)


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
