# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Share GPU memory between processes on the same GPU through cuMem POSIX FDs.

Used by the NIXL mirror (exporting side, examples/local_nixl_mirror/nixl_mirror.py) and
inside the engine by the local_mirror load strategy and local_mirror.py
(importing side). Calls libcuda directly through ctypes on the device's
primary context, the same context the engine's CUDA runtime made current.

Messages are AF_UNIX SOCK_SEQPACKET, one JSON object per message (split into
fragments, since a datagram cannot exceed the socket buffer), with up to
MAX_FDS_PER_MSG FDs attached; the mirror's requests and replies are
described in nixl_mirror.serve_local.
"""

from __future__ import annotations

import ctypes
import json
import socket
from typing import Any

# SCM_RIGHTS caps one message at 253 FDs.
MAX_FDS_PER_MSG = 250
# Fragment payload, well under the default unix socket buffer.
_FRAGMENT_BYTES = 60_000

CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CUmemAllocFlags),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


_libcuda: ctypes.CDLL | None = None
_primary_contexts: dict[int, ctypes.c_void_p] = {}


def _cu() -> ctypes.CDLL:
    global _libcuda
    if _libcuda is None:
        # Published only after cuInit returns: another thread that sees it
        # set must not call into the driver before it is initialized.
        lib = ctypes.CDLL("libcuda.so.1")
        _check_rc(lib, "cuInit", lib.cuInit(0))
        _libcuda = lib
    return _libcuda


def _check(name: str, *args: Any) -> None:
    lib = _cu()
    _check_rc(lib, name, getattr(lib, name)(*args))


def _check_rc(lib: ctypes.CDLL, name: str, rc: int) -> None:
    if rc != 0:
        err = ctypes.c_char_p()
        lib.cuGetErrorString(rc, ctypes.byref(err))
        raise RuntimeError(f"{name} failed: {rc} {(err.value or b'').decode()}")


def device_count() -> int:
    count = ctypes.c_int()
    _check("cuDeviceGetCount", ctypes.byref(count))
    return count.value


def gpu_uuid(ordinal: int) -> str:
    dev = ctypes.c_int()
    _check("cuDeviceGet", ctypes.byref(dev), ordinal)
    raw = (ctypes.c_ubyte * 16)()
    _check("cuDeviceGetUuid", raw, dev)
    h = bytes(raw).hex()
    return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def ordinal_for_uuid(uuid: str) -> int:
    for ordinal in range(device_count()):
        if gpu_uuid(ordinal) == uuid:
            return ordinal
    raise RuntimeError(f"GPU {uuid} is not visible to this process")


def use_device(ordinal: int) -> None:
    """Make the device's primary context current on this thread."""
    ctx = _primary_contexts.get(ordinal)
    if ctx is None:
        dev = ctypes.c_int()
        _check("cuDeviceGet", ctypes.byref(dev), ordinal)
        ctx = ctypes.c_void_p()
        _check("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev)
        _primary_contexts[ordinal] = ctx
    _check("cuCtxSetCurrent", ctx)


def create(ordinal: int, size: int) -> int:
    """Allocate exportable physical memory; size must be granularity-aligned."""
    prop = _CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = ordinal
    # Matches vLLM's cumem allocator, so the memory can be registered with a NIC.
    prop.allocFlags.gpuDirectRDMACapable = 1
    handle = ctypes.c_uint64()
    _check(
        "cuMemCreate",
        ctypes.byref(handle),
        ctypes.c_size_t(size),
        ctypes.byref(prop),
        ctypes.c_uint64(0),
    )
    return handle.value


def export_fd(handle: int) -> int:
    fd = ctypes.c_int()
    _check(
        "cuMemExportToShareableHandle",
        ctypes.byref(fd),
        ctypes.c_uint64(handle),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        ctypes.c_uint64(0),
    )
    return fd.value


def import_fd(fd: int) -> int:
    handle = ctypes.c_uint64()
    _check(
        "cuMemImportFromShareableHandle",
        ctypes.byref(handle),
        ctypes.c_void_p(fd),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    )
    return handle.value


def release(handle: int) -> None:
    _check("cuMemRelease", ctypes.c_uint64(handle))


def reserve(size: int) -> int:
    va = ctypes.c_uint64()
    _check(
        "cuMemAddressReserve",
        ctypes.byref(va),
        ctypes.c_size_t(size),
        ctypes.c_size_t(0),
        ctypes.c_uint64(0),
        ctypes.c_uint64(0),
    )
    return va.value


def address_free(va: int, size: int) -> None:
    _check("cuMemAddressFree", ctypes.c_uint64(va), ctypes.c_size_t(size))


def map_at(va: int, size: int, handle: int, ordinal: int) -> None:
    _check(
        "cuMemMap",
        ctypes.c_uint64(va),
        ctypes.c_size_t(size),
        ctypes.c_size_t(0),
        ctypes.c_uint64(handle),
        ctypes.c_uint64(0),
    )
    desc = _CUmemAccessDesc()
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = ordinal
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _check(
        "cuMemSetAccess",
        ctypes.c_uint64(va),
        ctypes.c_size_t(size),
        ctypes.byref(desc),
        ctypes.c_size_t(1),
    )


def unmap(va: int, size: int) -> None:
    _check("cuMemUnmap", ctypes.c_uint64(va), ctypes.c_size_t(size))


def memset_u32(va: int, value: int, size: int) -> None:
    _check(
        "cuMemsetD32_v2",
        ctypes.c_uint64(va),
        ctypes.c_uint(value),
        ctypes.c_size_t(size // 4),
    )


def synchronize() -> None:
    _check("cuCtxSynchronize")


def read(va: int, nbytes: int) -> bytes:
    buf = ctypes.create_string_buffer(nbytes)
    _check("cuMemcpyDtoH_v2", buf, ctypes.c_uint64(va), ctypes.c_size_t(nbytes))
    return buf.raw


def write(va: int, data: bytes) -> None:
    _check("cuMemcpyHtoD_v2", ctypes.c_uint64(va), data, ctypes.c_size_t(len(data)))


def copy(dst: int, src: int, nbytes: int) -> None:
    """Device-to-device copy within the current context."""
    _check(
        "cuMemcpyDtoD_v2", ctypes.c_uint64(dst), ctypes.c_uint64(src), ctypes.c_size_t(nbytes)
    )


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
