# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First load on top of a local mirror's GPU memory.

A NIXL mirror on the same GPU holds a byte-for-byte copy of every region an MX
source registered, already post-processed. Instead of allocating the model and
receiving into it, the engine builds the model on the meta device, runs the
same pre-receive processing an RDMA target runs (so the layout matches what the
source published), and points every tensor at the mirror's copy of the source
tensor with the same name. Nothing is allocated for weights and nothing is
copied.

The mirror hands out one POSIX FD per region over MX_LOCAL_MIRROR_SOCKET,
labelled with the source address it copied; the source's tensor manifest (from
MX) gives each tensor's source address, so tensor -> region -> offset is plain
arithmetic. Everything is checked before the first region is mapped; until
then any failure returns None and the normal strategy chain runs instead.

The mirror opens its socket only once its copy is complete, so a mirror that
is still copying looks absent; MX waits up to MX_LOCAL_MIRROR_TIMEOUT for it.

The adopted weights are not vLLM allocator memory, so a level-2 sleep leaves
them mapped and intact; a reload after it only registers and publishes them
again instead of transferring weights that were never lost.

A restored engine (CRIU snapshot taken after a level-2 sleep) has the
opposite problem: its weights are vLLM allocator memory whose physical pages
were released before the checkpoint. That reload is the local_mirror load
strategy (load_strategy/local_mirror_strategy.py), which uses the helpers
here: receiving the mirror's FDs, finishing by tensor name, and releasing
the mirror.
"""

from __future__ import annotations

import bisect
import logging
import os
import socket
import time

import torch

from . import envs, local_gpu, p2p_pb2
from .load_strategy.base import publish_source_if_supported, register_tensors
from .load_strategy.context import LoadResult

logger = logging.getLogger("modelexpress.local_mirror")

# Imported handles and mappings stay alive for the life of the process: the
# model's weights live in them.
_mappings: list[tuple[int, int, int]] = []
# (owner module, attribute, kind, data_ptr) of every tensor bound to them.
_adopted: list[tuple[torch.nn.Module, str, str, int]] = []


class _MetaBackend:
    """The engine's accelerator backend, also counting meta tensors as device
    tensors, so MX's attribute capture and discovery see the meta model."""

    def __init__(self, backend):
        self._backend = backend

    def is_accel_tensor(self, tensor):
        return tensor.device.type == "meta" or self._backend.is_accel_tensor(tensor)

    def __getattr__(self, name):
        return getattr(self._backend, name)


def _query_memory(path: str) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as sock:
        sock.settimeout(60)
        sock.connect(path)
        local_gpu.send_msg(sock, {"want": "memory"})
        msg, _ = local_gpu.recv_msg(sock)
    return msg


def _wait_for_mirror(
    device: int, path: str | None = None, timeout: float | None = None
) -> tuple[str, int] | None:
    """(socket path, bytes held) of the mirror on this GPU once its copy is
    complete, or None when none is configured or it is not ready within
    the timeout (default MX_LOCAL_MIRROR_SOCKET / MX_LOCAL_MIRROR_TIMEOUT).
    A mirror opens its socket only after its copy, so until then connecting
    fails and this keeps trying."""
    path = path or envs.MX_LOCAL_MIRROR_SOCKET
    if not path:
        return None
    if timeout is None:
        timeout = envs.MX_LOCAL_MIRROR_TIMEOUT
    start = time.monotonic()
    waiting = False
    while True:
        try:
            msg = _query_memory(path)
            break
        except OSError as e:
            if time.monotonic() - start >= timeout:
                logger.warning(
                    f"Local mirror at {path} not ready after {timeout:.0f}s, "
                    f"loading normally: {e}"
                )
                return None
            if not waiting:
                logger.info(
                    f"Waiting up to {timeout:.0f}s for the local mirror at {path} "
                    f"to finish copying"
                )
                waiting = True
            time.sleep(0.5)
    if waiting:
        logger.info(f"Local mirror at {path} ready after {time.monotonic() - start:.1f}s")
    if msg.get("gpu_uuid") != local_gpu.gpu_uuid(device):
        logger.warning(f"Local mirror at {path} is on {msg.get('gpu_uuid')}, not on this GPU")
        return None
    return path, int(msg["memory_bytes"])


def local_mirror_memory(device: int) -> int:
    """Bytes the local mirror holds on this GPU, which a first load uses
    instead of allocating; 0 without a ready mirror on this GPU."""
    ready = _wait_for_mirror(device)
    return 0 if ready is None else ready[1]


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


def _build_meta_model(vllm_config, model_config, prefix: str):
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_default_torch_dtype(model_config.dtype), torch.device("meta"):
        return initialize_model(
            vllm_config=vllm_config, model_config=model_config, prefix=prefix
        )


def _process_on_meta(model, model_config, backend) -> None:
    """What VllmAdapter.before_rdma_receive does, on the meta device."""
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    from .engines.vllm.adapter import _VLLM_PRE_RDMA_FINALIZER_NAMES
    from .tensor_utils import adopt_hidden_tensors, capture_tensor_attrs

    with capture_tensor_attrs(backend):
        for _, module in model.named_modules():
            for finalizer in _VLLM_PRE_RDMA_FINALIZER_NAMES:
                if callable(getattr(module, finalizer, None)):
                    getattr(module, finalizer)()
        process_weights_after_loading(model, model_config, torch.device("meta"))
    adopt_hidden_tensors(model, backend)


def _plan(model, backend, manifest: dict, regions: dict) -> list:
    """For every model tensor: (owner module, attribute, kind, tensor,
    source_addr of its region, byte offset in the region, bytes).
    Raises unless every tensor is in the manifest with the same size and lies
    inside one of the mirror's regions."""
    from .tensor_utils import iter_module_tensors

    starts = sorted(regions)
    plan = []
    for name, tensor, kind in iter_module_tensors(model, backend):
        t = tensor.data if hasattr(tensor, "data") else tensor
        owner_name, _, attr = name.rpartition(".")
        owner = model.get_submodule(owner_name)
        if t.numel() == 0:
            plan.append((owner, attr, kind, t, None, 0, 0))
            continue
        if t.is_contiguous():
            key, nbytes = name, t.numel() * t.element_size()
        else:
            key, nbytes = f"{name}.__storage", t.untyped_storage().nbytes()
        entry = manifest.get(key)
        if entry is None:
            raise LookupError(f"{key} is not in the source manifest")
        if entry.size != nbytes:
            raise ValueError(f"{key}: source has {entry.size} bytes, model {nbytes}")
        i = bisect.bisect_right(starts, entry.addr) - 1
        if i < 0 or entry.addr + nbytes > starts[i] + regions[starts[i]][0]:
            raise LookupError(f"{key} at 0x{entry.addr:x} is in no mirror region")
        plan.append((owner, attr, kind, t, starts[i], entry.addr - starts[i], nbytes))
    return plan


def _map_regions(regions: dict, used: set, device: int) -> dict[int, int]:
    """Import and map each used region; source_addr -> local VA."""
    vas = {}
    for addr in sorted(used):
        _, allocation_size, fd = regions[addr]
        handle = local_gpu.import_fd(fd)
        va = local_gpu.reserve(allocation_size)
        local_gpu.map_at(va, allocation_size, handle, device)
        _mappings.append((va, allocation_size, handle))
        vas[addr] = va
    return vas


def _tensor_at(ptr: int, nbytes: int, like: torch.Tensor, device: torch.device):
    """A tensor shaped like `like` over nbytes of existing memory at ptr."""
    storage = torch._C._construct_storage_from_data_pointer(ptr, device, nbytes)
    return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
        {
            "size": like.shape,
            "stride": like.stride(),
            "storage_offset": like.storage_offset() if not like.is_contiguous() else 0,
            "dtype": like.dtype,
        },
        storage,
    )


def _bind(plan: list, vas: dict[int, int], device: torch.device) -> None:
    for owner, attr, kind, t, region, offset, nbytes in plan:
        if region is None:
            new = torch.empty(t.shape, dtype=t.dtype, device=device)
        else:
            new = _tensor_at(vas[region] + offset, nbytes, t, device)
        # Swapped in place: the same Python object (vLLM's parameter class,
        # weight_loader and quant attributes, and any other reference to it)
        # now holds the new storage. set_data cannot move meta to CUDA.
        old = owner._parameters[attr] if kind == "parameter" else owner._buffers[attr]
        if isinstance(old, torch.nn.Parameter):
            new = torch.Tensor._make_subclass(type(old), new, old.requires_grad)
        # swap_tensors exchanges the attribute dicts too.
        new.__dict__.update(old.__dict__)
        torch.utils.swap_tensors(old, new)
        if region is not None:
            _adopted.append((owner, attr, kind, old.data_ptr()))


def _tensor_of(owner: torch.nn.Module, attr: str, kind: str):
    tensors = owner._parameters if kind == "parameter" else owner._buffers
    return tensors.get(attr)


def _publish(result, ctx) -> None:
    if ctx.p2p_enabled:
        register_tensors(result, ctx)
        publish_source_if_supported(result, ctx)


def release_mirror(path: str) -> None:
    """Ask the mirror to free its copy (it keeps running, holding nothing).
    For a reload that cannot use the copy: the memory is needed for the
    engine's own weights."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as sock:
        sock.settimeout(60)
        sock.connect(path)
        local_gpu.send_msg(sock, {"want": "release"})
        local_gpu.recv_msg(sock)


def reload_on_local_mirror(model: torch.nn.Module, ctx) -> bool:
    """On a reload of a model whose weights a first load adopted from the
    local mirror: True, after registering and publishing them again, when
    every adopted tensor still points at the mirror memory (sleep never
    unmaps it, so there is nothing to transfer). False otherwise; reload
    normally. (A reload of sleeping vLLM allocator memory is the local_mirror
    load strategy instead.)"""
    if not _adopted:
        return False
    modules = {id(m) for m in model.modules()}
    for owner, attr, kind, ptr in _adopted:
        tensor = _tensor_of(owner, attr, kind)
        if id(owner) not in modules or tensor is None or tensor.data_ptr() != ptr:
            logger.info(
                f"[Worker {ctx.global_rank}] {attr} no longer on the local mirror's "
                f"memory, reloading normally"
            )
            return False
    logger.info(
        f"[Worker {ctx.global_rank}] Reload: all {len(_adopted)} adopted tensors are "
        f"still on the local mirror's memory, nothing to transfer"
    )
    _publish(LoadResult(value=model, model=model), ctx)
    return True


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

    local_gpu.use_device(device)
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
            local_gpu.synchronize()
            for dst, off, nbytes in placed:
                local_gpu.copy(dst, base + off, nbytes)
            local_gpu.synchronize()
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


def load_from_local_mirror(vllm_config, model_config, prefix: str, ctx):
    """The model with its weights on the local mirror's memory, or None when
    there is no usable mirror (nothing was mapped; load normally)."""
    ready = _wait_for_mirror(ctx.device_id)
    if ready is None:
        return None
    path = ready[0]

    t0 = time.perf_counter()
    timings = {}
    # With its index: torch treats "cuda" and "cuda:0" as different devices.
    device = torch.device(ctx.target_device.type, ctx.device_id)
    regions: dict = {}
    model = None
    try:
        header, regions = _receive_regions(path, ctx.device_id)
        timings["fds"] = time.perf_counter() - t0
        t = time.perf_counter()
        manifest = _source_manifest(ctx, header["source_agent"])
        timings["manifest"] = time.perf_counter() - t
        t = time.perf_counter()
        model = _build_meta_model(vllm_config, model_config, prefix)
        backend = _MetaBackend(ctx.adapter.accelerator_backend)
        _process_on_meta(model, model_config, backend)
        timings["meta_model"] = time.perf_counter() - t
        plan = _plan(model, backend, manifest, regions)
    except Exception as e:
        logger.warning(
            f"[Worker {ctx.global_rank}] Local mirror at {path} not usable, "
            f"loading normally: {e}"
        )
        for *_, fd in regions.values():
            os.close(fd)
        if model is not None:
            ctx.adapter._unregister_model_layers(model)
        return None

    # From here on the model's weights are the mirror's memory.
    t = time.perf_counter()
    used = {region for _, _, _, _, region, _, _ in plan if region is not None}
    vas = _map_regions(regions, used, ctx.device_id)
    for *_, fd in regions.values():
        os.close(fd)
    _bind(plan, vas, device)
    timings["map_bind"] = time.perf_counter() - t

    left = [n for n, p in model.named_parameters() if p.is_meta]
    left += [n for n, b in model.named_buffers() if b.is_meta]
    if left:
        raise RuntimeError(f"{len(left)} tensors still on meta, e.g. {left[:5]}")

    t = time.perf_counter()
    result = ctx.adapter.after_rdma_receive(LoadResult(value=model, model=model))
    timings["after_receive"] = time.perf_counter() - t

    total = sum(nbytes for *_, nbytes in plan)
    logger.info(
        f"[Worker {ctx.global_rank}] Adopted local mirror of {header['source_agent']} "
        f"({header['source_endpoint']}): {len(plan)} tensors in {len(used)} regions, "
        f"{total / 1e9:.2f} GB, no copy; "
        + ", ".join(f"{k} {v:.3f}s" for k, v in timings.items())
        + f"; total {time.perf_counter() - t0:.3f}s"
    )

    _publish(result, ctx)
    return result.model
