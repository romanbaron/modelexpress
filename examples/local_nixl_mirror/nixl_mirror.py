# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mirror one NIXL agent's VRAM onto one local GPU, then serve the copy.

Given only the agent's listen-thread endpoint (ip:port), the mirror asks it
for its metadata blob, reads the agent name and every registered VRAM region
out of it, allocates one cuMem allocation per region (exportable and
RDMA-capable, mapped back to back and registered as one NIXL region) and
READs every region in one transfer.

It then serves the copy two ways:
  - its NIXL agent keeps the copy registered and answers metadata requests on
    --nixl-port, so another mirror can copy from it;
  - local processes on the same GPU get it over a unix socket (--socket), as
    one POSIX FD per region labelled with the source address it was copied
    from, so an engine can map the copy as its weights without allocating or
    copying. The socket exists only once the copy is complete.

One process mirrors one GPU; mirror-node.sh runs one per GPU of a node.

Usage: nixl_mirror.py --endpoint IP:PORT [--gpu N] [--nixl-port PORT] [--socket PATH]
"""

import argparse
import json
import os
import socket
import struct
import time
import uuid
from typing import Any

from nixl._api import nixl_agent, nixl_agent_config

# cuMem allocations are multiples of this.
GRANULARITY = 2 << 20
# nixl_mem_t, in enum order.
MEM_TYPES = ["DRAM", "VRAM", "BLK", "OBJ", "FILE"]
GPU = 0


# --- CUDA driver (cuda-python) and socket messages ------------------------
# The same calls and framing as the local_mirror strategy's, kept here so the
# mirror needs neither torch nor the client package.

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


def cu_gpu_uuid(ordinal: int) -> str:
    dev = _call("cuDeviceGet", ordinal)
    raw = _call("cuDeviceGetUuid", dev).bytes
    if isinstance(raw, str):
        raw = raw.encode("latin-1")
    h = bytes(raw).hex()
    return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def cu_use_device(ordinal: int) -> None:
    """Make the device's primary context current on this thread (the
    context the CUDA runtime, and so torch, uses)."""
    ctx = _primary_contexts.get(ordinal)
    if ctx is None:
        dev = _call("cuDeviceGet", ordinal)
        ctx = _call("cuDevicePrimaryCtxRetain", dev)
        _primary_contexts[ordinal] = ctx
    _call("cuCtxSetCurrent", ctx)


def cu_create(ordinal: int, size: int) -> int:
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


def cu_export_fd(handle: int) -> int:
    cu = _cu()
    fd = _call(
        "cuMemExportToShareableHandle",
        handle,
        cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        0,
    )
    return int(fd)


def cu_release(handle: int) -> None:
    _call("cuMemRelease", handle)


def cu_reserve(size: int) -> int:
    return int(_call("cuMemAddressReserve", size, 0, 0, 0))


def cu_address_free(va: int, size: int) -> None:
    _call("cuMemAddressFree", va, size)


def cu_map_at(va: int, size: int, handle: int, ordinal: int) -> None:
    cu = _cu()
    _call("cuMemMap", va, size, 0, handle, 0)
    access = cu.CUmemAccessDesc()
    access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = ordinal
    access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _call("cuMemSetAccess", va, size, [access], 1)


def cu_unmap(va: int, size: int) -> None:
    _call("cuMemUnmap", va, size)


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


def log(msg):
    # Under CUDA_VISIBLE_DEVICES (as mirror-node.sh runs it), name the GPU by
    # what is visible rather than by its ordinal, which is then always 0.
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES") or GPU
    print(f"[{time.strftime('%H:%M:%S')}] [mirror gpu {gpu}] {msg}", flush=True)


def aligned(size):
    return -(-size // GRANULARITY) * GRANULARITY


# --- Reading the source agent's metadata -----------------------------------


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"peer closed after {len(buf)} of {n} bytes")
        buf += chunk
    return bytes(buf)


def fetch_metadata(host, port, timeout=10):
    """The agent's full metadata blob, as its listen thread hands it out.

    The wire protocol: every frame is a native size_t length followed by the
    message. "NIXLCOMM:SEND" asks for the metadata; the reply is
    "NIXLCOMM:LOAD" followed by the blob. This is what fetch_remote_metadata
    does internally, except that it loads the blob without returning it.
    """
    with socket.create_connection((host, port), timeout=timeout) as sock:
        request = b"NIXLCOMM:SEND"
        sock.sendall(struct.pack("<Q", len(request)) + request)
        (size,) = struct.unpack("<Q", _recv_exact(sock, 8))
        reply = _recv_exact(sock, size)
    if not reply.startswith(b"NIXLCOMM:LOAD"):
        raise ValueError(f"unexpected reply {reply[:20]!r}")
    return reply[len(b"NIXLCOMM:LOAD"):]


class _SerDes:
    """Reader for nixlSerDes: "nixlSerDes|", then per field the tag, a
    native size_t length, the value and "|"."""

    def __init__(self, blob):
        if not blob.startswith(b"nixlSerDes|"):
            raise ValueError("not a nixlSerDes blob")
        self.blob, self.offset = blob, len(b"nixlSerDes|")

    def get(self, tag):
        tag = tag.encode()
        if self.blob[self.offset:self.offset + len(tag)] != tag:
            raise ValueError(f"expected tag {tag!r} at offset {self.offset}")
        self.offset += len(tag)
        (n,) = struct.unpack_from("<Q", self.blob, self.offset)
        self.offset += 8
        value = self.blob[self.offset:self.offset + n]
        self.offset += n
        if self.blob[self.offset:self.offset + 1] != b"|":
            raise ValueError(f"missing separator after tag {tag!r}")
        self.offset += 1
        return value

    def get_int(self, tag):
        return int.from_bytes(self.get(tag), "little")


def parse_metadata(blob):
    """Agent name and its registered VRAM regions [(addr, len, devId)], as
    nixlAgentData::getLocalMD writes them: Agent, Conns and per backend t/c,
    then MemSection and per (memory type, backend) section the backend name
    and a descriptor list whose entries are addr, len, devId (3 x u64) plus
    the backend's own metadata for the region (the UCX rkey). Every section
    is walked; only VRAM is kept."""
    sd = _SerDes(blob)
    agent = sd.get("Agent").decode()
    for _ in range(sd.get_int("Conns")):
        sd.get("t")
        sd.get("c")
    if sd.get("") != b"MemSection":
        raise ValueError("no MemSection")
    regions = set()
    for _ in range(sd.get_int("nixlSecElms")):
        sd.get("bknd")
        kind = sd.get("nixlDList")
        if kind != b"nixlSDList":
            raise ValueError(f"unexpected descriptor list {kind!r}")
        mem_type = MEM_TYPES[sd.get_int("t")]
        for _ in range(sd.get_int("n")):
            region = struct.unpack_from("<QQQ", sd.get(""))
            if mem_type == "VRAM":
                regions.add(region)
    return agent, sorted(regions)


def read_endpoint(host, port, timeout=600):
    """The blob, agent name and VRAM regions of the agent at host:port,
    retrying until it is up and has VRAM registered."""
    deadline = time.time() + timeout
    while True:
        try:
            blob = fetch_metadata(host, port)
            agent, regions = parse_metadata(blob)
            if regions:
                return blob, agent, regions
        except OSError:
            pass
        if time.time() > deadline:
            raise SystemExit(f"FAIL: no VRAM regions at {host}:{port} after {timeout}s")
        time.sleep(0.01)


# --- Copying ----------------------------------------------------------------


def allocate(sizes):
    """One exportable, RDMA-capable allocation per size, mapped back to back
    in one reserved range so the copy registers as a single region. Each
    allocation stays separate: a reader imports each by FD and maps it at its
    own address (cuMemMap needs offset 0 into a handle).
    Returns [(va, handle)] and the (base, span) of the range."""
    sizes = [aligned(s) for s in sizes]
    span = sum(sizes)
    base = cu_reserve(span)
    local, off = [], 0
    for size in sizes:
        handle = cu_create(GPU, size)
        cu_map_at(base + off, size, handle, GPU)
        local.append((base + off, handle))
        off += size
    return local, base, span


def copy(host, port, nixl_port):
    """Copy the agent at host:port onto GPU. Returns the source agent's name,
    the agent holding the copy (keep it alive: it keeps the registration and
    the listen thread) and [(source_addr, size, handle)] per region."""
    timings = {}
    t0 = time.perf_counter()
    blob, source_agent, regions = read_endpoint(host, port)
    timings["fetch"] = time.perf_counter() - t0
    total = sum(size for _, size, _ in regions)
    log(f"source agent {source_agent} at {host}:{port}: {len(regions)} VRAM regions, "
        f"{total / 2**20:.0f} MiB")

    t0 = time.perf_counter()
    cu_use_device(GPU)
    name = f"mirror{GPU}-{uuid.uuid4().hex[:8]}"
    agent = nixl_agent(
        name, nixl_agent_config(backends=["UCX"], enable_listen_thread=True, listen_port=nixl_port)
    )
    timings["setup"] = time.perf_counter() - t0

    # The blob already fetched is loaded as is: no second round trip.
    loaded = agent.add_remote_agent(blob)
    loaded = loaded.decode() if isinstance(loaded, bytes) else loaded
    if loaded != source_agent:
        raise SystemExit(f"FAIL: loaded agent {loaded!r}, parsed {source_agent!r}")

    t0 = time.perf_counter()
    local, base, span = allocate([size for _, size, _ in regions])
    timings["allocate"] = time.perf_counter() - t0

    # Registered once: the same registration lets this transfer write into
    # the copy and later lets other mirrors read it.
    t0 = time.perf_counter()
    registration = agent.register_memory([(base, span, GPU, "")], mem_type="VRAM")
    timings["register"] = time.perf_counter() - t0

    # Entry i of the source's list is read into entry i of ours.
    remote_side = agent.prep_xfer_dlist(source_agent, regions, mem_type="VRAM")
    local_side = agent.prep_xfer_dlist(
        "", [(va, size, GPU) for (va, _), (_, size, _) in zip(local, regions)], mem_type="VRAM"
    )
    indices = list(range(len(regions)))
    xfer = agent.make_prepped_xfer("READ", local_side, indices, remote_side, indices)
    t0 = time.perf_counter()
    state = agent.transfer(xfer)
    while state not in ("DONE", "ERR"):
        state = agent.check_xfer_state(xfer)
    timings["transfer"] = time.perf_counter() - t0
    if state != "DONE":
        raise SystemExit(f"FAIL: transfer state {state}")
    agent.release_xfer_handle(xfer)
    agent.release_dlist_handle(local_side)
    agent.release_dlist_handle(remote_side)
    agent.remove_remote_agent(source_agent)

    gbps = total * 8 / timings["transfer"] / 1e9
    log("timing: " + ", ".join(f"{k} {v:.3f}s" for k, v in timings.items())
        + f"; {total / 1e9:.2f} GB at {gbps:.1f} Gb/s; serving on port {nixl_port}")
    copied = [(addr, size, handle) for (_, handle), (addr, size, _) in zip(local, regions)]

    def release():
        """Free the copy: deregister it and drop every mapping and handle.
        Memory a client imported stays valid until that client releases it."""
        agent.deregister_memory(registration)
        for (va, handle), (_, size, _) in zip(local, regions):
            cu_unmap(va, aligned(size))
            cu_release(handle)
        cu_address_free(base, span)

    return source_agent, agent, copied, release


# --- Serving local processes -------------------------------------------------


def serve_local(path, source_agent, source_endpoint, regions, release):
    """Answer local requests on a unix socket, one client at a time, forever.

    {"want": "memory"} gets {"memory_bytes", "gpu_uuid"}: whether the copy is
    ready (the socket exists only once it is), on which GPU, and how much it
    holds. {"want": "release"} frees the copy (for a client that cannot use
    it and needs the memory) and gets {"released": bytes}; the mirror keeps
    running, holding nothing: memory_bytes is 0 and regions get {"error"}.
    {"want": "regions"} gets messages {"source_agent",
    "source_endpoint", "gpu_uuid", "batch": [[source_addr, size,
    allocation_size], ...], "last"} with one FD per batch entry (send_msg
    framing, at most MAX_FDS_PER_MSG FDs a message). The client owns the FDs
    it receives; the mirror keeps its own handles and mappings, so the memory
    stays valid for both.

    regions: [(source_addr, size, handle)]; release: frees them.
    """
    gpu_uuid = cu_gpu_uuid(GPU)
    memory_bytes = sum(aligned(size) for _, size, _ in regions)
    # Exported once, before any client asks: sending dups them (SCM_RIGHTS),
    # so the same FDs serve every client.
    fds = [cu_export_fd(handle) for _, _, handle in regions]
    step = MAX_FDS_PER_MSG

    if os.path.exists(path):
        os.unlink(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    server.bind(path)
    server.listen()
    log(f"handing out {len(regions)} regions on {path} ({gpu_uuid})")
    while True:
        conn, _ = server.accept()
        try:
            with conn:
                request, _ = recv_msg(conn)
                if request.get("want") == "memory":
                    send_msg(conn, {"memory_bytes": memory_bytes, "gpu_uuid": gpu_uuid})
                    continue
                if request.get("want") == "release":
                    released = memory_bytes
                    if regions:
                        for fd in fds:
                            os.close(fd)
                        release()
                        regions, fds, memory_bytes = [], [], 0
                        log(f"released the copy ({released / 2**30:.2f} GiB) on request")
                    send_msg(conn, {"released": released})
                    continue
                if not regions:
                    send_msg(conn, {"error": "the copy was released"})
                    continue
                for i in range(0, len(regions), step):
                    send_msg(conn, {
                        "source_agent": source_agent,
                        "source_endpoint": source_endpoint,
                        "gpu_uuid": gpu_uuid,
                        "batch": [[a, s, aligned(s)] for a, s, _ in regions[i:i + step]],
                        "last": i + step >= len(regions),
                    }, fds[i:i + step])
            log(f"handed out {len(regions)} regions")
        except OSError as e:
            log(f"local client failed: {e}")


def main():
    global GPU
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--endpoint", required=True, help="IP:PORT of the source agent's listen thread")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ordinal to copy onto")
    parser.add_argument("--nixl-port", type=int, default=7555,
                        help="listen port of this mirror's own NIXL agent")
    parser.add_argument("--socket", default="/tmp/mx-local-mirror-0.sock",
                        help="unix socket handing the copy to local processes")
    args = parser.parse_args()
    GPU = args.gpu
    host, port = args.endpoint.rsplit(":", 1)

    # agent stays referenced here while serve_local runs (forever), which
    # keeps the copy registered and its listen thread answering.
    source_agent, agent, regions, release = copy(host, int(port), args.nixl_port)
    serve_local(args.socket, source_agent, args.endpoint, regions, release)


if __name__ == "__main__":
    main()
