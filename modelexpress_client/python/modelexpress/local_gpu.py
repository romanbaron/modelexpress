# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Share GPU memory between processes on the same GPU through cuMem POSIX FDs.

Used by the NIXL mirror (exporting side, examples/local_nixl_mirror/nixl_mirror.py)
and inside the engine by the local_mirror load strategy and local_mirror.py
(importing side). PyTorch exposes none of the cuMem VMM calls this needs
(exportable allocations, FD export/import, mapping at a given address), so
they go through cuda-python's driver bindings, as vmm/backend.py does. The
mirror does not import torch; the engine side uses torch for everything else.

Messages are AF_UNIX SOCK_SEQPACKET, one JSON object per message (split into
fragments, since a datagram cannot exceed the socket buffer), with up to
MAX_FDS_PER_MSG FDs attached; the mirror's requests and replies are
described in nixl_mirror.serve_local.
"""

from __future__ import annotations

import json
import socket
from typing import Any

# SCM_RIGHTS caps one message at 253 FDs.
MAX_FDS_PER_MSG = 250
# Fragment payload, well under the default unix socket buffer.
_FRAGMENT_BYTES = 60_000

_driver: Any = None
_primary_contexts: dict[int, Any] = {}


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


def gpu_uuid(ordinal: int) -> str:
    dev = _call("cuDeviceGet", ordinal)
    raw = _call("cuDeviceGetUuid", dev).bytes
    if isinstance(raw, str):
        raw = raw.encode("latin-1")
    h = bytes(raw).hex()
    return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def use_device(ordinal: int) -> None:
    """Make the device's primary context current on this thread (the
    context the CUDA runtime, and so torch, uses)."""
    ctx = _primary_contexts.get(ordinal)
    if ctx is None:
        dev = _call("cuDeviceGet", ordinal)
        ctx = _call("cuDevicePrimaryCtxRetain", dev)
        _primary_contexts[ordinal] = ctx
    _call("cuCtxSetCurrent", ctx)


def create(ordinal: int, size: int) -> int:
    """Allocate exportable physical memory; size must be granularity-aligned."""
    cu = _cu()
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = (
        cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = ordinal
    # Matches vLLM's cumem allocator, so the memory can be registered with a NIC.
    prop.allocFlags.gpuDirectRDMACapable = 1
    return int(_call("cuMemCreate", size, prop, 0))


def export_fd(handle: int) -> int:
    cu = _cu()
    fd = _call(
        "cuMemExportToShareableHandle",
        handle,
        cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        0,
    )
    return int(fd)


def import_fd(fd: int) -> int:
    cu = _cu()
    handle = _call(
        "cuMemImportFromShareableHandle",
        fd,
        cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    )
    return int(handle)


def release(handle: int) -> None:
    _call("cuMemRelease", handle)


def retain_handle(va: int) -> int | None:
    """A new reference to the cuMem allocation mapped at `va` (the same
    handle value it was mapped with), or None when no cuMem allocation is
    mapped there. Release it like any handle."""
    cu = _cu()
    err, handle = cu.cuMemRetainAllocationHandle(va)
    return int(handle) if err == cu.CUresult.CUDA_SUCCESS else None


def reserve(size: int) -> int:
    return int(_call("cuMemAddressReserve", size, 0, 0, 0))


def address_free(va: int, size: int) -> None:
    _call("cuMemAddressFree", va, size)


def map_at(va: int, size: int, handle: int, ordinal: int) -> None:
    cu = _cu()
    _call("cuMemMap", va, size, 0, handle, 0)
    access = cu.CUmemAccessDesc()
    access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = ordinal
    access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _call("cuMemSetAccess", va, size, [access], 1)


def unmap(va: int, size: int) -> None:
    _call("cuMemUnmap", va, size)


def copy(dst: int, src: int, nbytes: int) -> None:
    """Device-to-device copy between raw addresses, in the current context."""
    _call("cuMemcpyDtoD", dst, src, nbytes)


def send_msg(sock: socket.socket, msg: dict, fds: list[int] | tuple = ()) -> None:
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


def recv_msg(sock: socket.socket, maxfds: int = 0) -> tuple[dict, list[int]]:
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
