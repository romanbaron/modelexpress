# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client side of a local NIXL mirror, for the local_mirror load strategy.

A NIXL mirror (examples/local_nixl_mirror/nixl_mirror.py) on the same GPU
holds a byte-for-byte copy of every region an MX source registered, already
post-processed. It hands that copy out over a unix socket as one POSIX FD per
region, labelled with the source address it copied, and opens the socket
only once its copy is complete.

These helpers receive the regions, finish by tensor name what mapping at the
source's addresses did not put in place, and ask the mirror to release its
copy when a reload cannot use it.
"""

from __future__ import annotations

import bisect
import os
import socket
import time

import torch

from . import local_gpu, p2p_pb2
from .load_strategy.context import LoadResult


def _receive_regions(path: str, device: int):
    """The mirror's regions as {source_addr: (size, allocation_size, fd)},
    plus its header."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    sock.settimeout(60)
    regions: dict[int, tuple[int, int, int]] = {}
    try:
        sock.connect(path)
        local_gpu.send_msg(sock, {"want": "regions"})
        while True:
            msg, fds = local_gpu.recv_msg(sock, local_gpu.MAX_FDS_PER_MSG)
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
    if msg["gpu_uuid"] != local_gpu.gpu_uuid(device):
        for *_, fd in regions.values():
            os.close(fd)
        raise RuntimeError(f"mirror is on {msg['gpu_uuid']}, not on this GPU")
    return msg, regions


def _source_manifest(ctx, source_agent: str) -> dict:
    """The tensor manifest of the MX source the mirror copied, found by the
    NIXL agent name among the READY sources of this model and rank."""
    from .metadata.worker_server import fetch_tensor_manifest

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


def release_mirror(path: str) -> None:
    """Ask the mirror to free its copy (it keeps running, holding nothing).
    For a reload that cannot use the copy: the memory is needed for the
    engine's own weights."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as sock:
        sock.settimeout(60)
        sock.connect(path)
        local_gpu.send_msg(sock, {"want": "release"})
        local_gpu.recv_msg(sock)


def finish_by_name(
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
                handle = local_gpu.import_fd(fd)
                va = local_gpu.reserve(allocation_size)
                local_gpu.map_at(va, allocation_size, handle, device)
                temp[region] = (va, allocation_size, handle)
        total = sum(nbytes for _, _, _, nbytes, _ in copies)
        if total:
            staging = torch.empty(total, dtype=torch.uint8, device=torch.device("cuda", device))
            base, offset, placed = staging.data_ptr(), 0, []
            for name, dst, src, nbytes, region in copies:
                va = src if region is None else temp[region][0] + (src - region)
                local_gpu.copy(base + offset, va, nbytes)
                placed.append((dst, offset, nbytes))
                offset += nbytes
            torch.cuda.synchronize(device)
            for dst, off, nbytes in placed:
                local_gpu.copy(dst, base + off, nbytes)
            torch.cuda.synchronize(device)
            del staging
    finally:
        for va, allocation_size, handle in temp.values():
            local_gpu.unmap(va, allocation_size)
            local_gpu.address_free(va, allocation_size)
            local_gpu.release(handle)
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
