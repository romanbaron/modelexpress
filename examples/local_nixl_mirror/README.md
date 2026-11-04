# Local NIXL Mirror

A NIXL mirror copies a READY ModelExpress source's GPU memory onto a local GPU
and hands that copy to an engine on the same GPU as cuMem allocations. The
engine maps the copy as its weights: it doesn't allocate, copy or
post-process them.

The main use is a vLLM replica restored from a CRIU/cuda-checkpoint snapshot
that was taken after a level-2 sleep. The snapshot keeps the weights' virtual
addresses, but the memory behind them is gone. Every replica restored from one
snapshot has the same weights allocations at the same addresses as the source
it was snapshotted from. A mirror of that source therefore holds exactly the
bytes each allocation needs, at a matching address and size.

## Files

- [`nixl_mirror.py`](nixl_mirror.py) mirrors one GPU and serves the copy:
  1. It reads the source agent's metadata from its listen-thread endpoint.
  2. It maps one exportable cuMem allocation per registered VRAM region, back
     to back, and registers them as one NIXL region.
  3. It READs every region in a single transfer.
  4. It keeps the copy registered on its own NIXL agent, so another mirror can
     copy from it.
  5. It serves the copy on a unix socket as one POSIX FD per region, each
     labelled with the source address it was copied from. The socket exists
     only once the copy is complete.
- [`mirror-node.sh`](mirror-node.sh) `BASE_HOST:BASE_PORT NUM_GPUS` runs one
  mirror per GPU. Rank `r` copies the agent at `BASE_PORT + r` onto GPU `r`,
  each mirror with `CUDA_VISIBLE_DEVICES` set to its own GPU. If any mirror
  exits, it stops the rest.

`nixl_mirror.py` uses `modelexpress.local_gpu`, which is ctypes over the CUDA
driver, so it doesn't need torch. It also works with a standalone copy of
`local_gpu.py` placed next to it.

## Socket protocol

The socket is a `SOCK_SEQPACKET` unix socket that carries JSON messages
(`local_gpu.send_msg` / `recv_msg`).

| Request | Reply |
|---|---|
| `{"want": "memory"}` | `{"memory_bytes", "gpu_uuid"}` |
| `{"want": "regions"}` | Messages `{"source_agent", "source_endpoint", "gpu_uuid", "batch": [[source_addr, size, allocation_size], ...], "last"}`, each with one FD per batch entry attached via `SCM_RIGHTS`. The client owns the FDs it receives. |
| `{"want": "release"}` | `{"released": bytes}`. The mirror frees its copy and keeps running empty: `memory_bytes` becomes 0 and a later `regions` request gets `{"error"}`. Memory that a client already imported stays valid until that client releases it. |

## ModelExpress side

### Reload of sleeping weights: the `local_mirror` load strategy

This is the first strategy in the chain. To enable it, set
`MX_LOCAL_MIRROR_RELOAD_SOCKET`, for example
`/tmp/mx-local-mirror-{rank}.sock`; `{rank}` is replaced with the local GPU
ordinal.

The strategy only runs when all of these are true:

- the load is a reload;
- the engine has weights allocations that are asleep;
- the mirror answers from the same GPU UUID with a non-empty copy.

Each sleeping allocation whose address and size match a mirror region gets
that region's FD imported and mapped at its reserved address. Anything left
over is finished by tensor name from the mirror's copy:

- allocations without a matching region are woken and written;
- regions without a matching allocation are copied by name.

On failure, the strategy:

1. unmaps what it mapped;
2. asks the mirror to release its copy, so the engine has room for its own
   weights;
3. raises `StrategyFailed(mutated=False)`.

Before trying any later strategy in the chain, the chain wakes the sleeping
weights (`EngineAdapter.wake_sleeping_weights`). Only a strategy with
`accepts_sleeping_weights = True` sees them asleep.

### First load: `MX_LOCAL_MIRROR_SOCKET`

A fresh engine on a GPU that runs a mirror builds the model on the meta
device. It then runs the same pre-receive processing as an RDMA target and
points every tensor at the mirror's copy of the source tensor with the same
name.

To size the engine's memory budget, vLLM must ask for
`get_external_weight_memory`. MX waits up to `MX_LOCAL_MIRROR_TIMEOUT` for the
mirror to finish copying.

## vLLM requirements

- Sleep mode must be on (`CuMemAllocator`), and the reload must follow a
  level-2 sleep: `reload_weights` runs before `wake_up(tags=["weights"])`.
  This order lets the mirror's memory replace the weights instead of sitting
  next to a woken copy.
- If a later strategy runs a layerwise reload, the buffers that the worker
  saved at sleep must be restored before the weights are woken. The adapter
  therefore calls `CuMemAllocator.wake_weights_hook` when the worker sets it,
  and the worker sets it to `lambda: self.wake_up(tags=["weights"])`.
  Without the hook, the allocator alone wakes the weights. The weights then
  have memory again, but the saved buffers stay unrestored.

## Usage

Run the mirror as the placeholder process of a pod on the node where the
restored replica will run:

```bash
/opt/mirror/mirror-node.sh "$SOURCE_POD_IP:5555" 8
```

Then set the variable on the restored engine:

```bash
MX_LOCAL_MIRROR_RELOAD_SOCKET=/tmp/mx-local-mirror-{rank}.sock
```

## Results (GLM-5.3 FP8, TP8 on B200, eager)

| Weights path on restore | Reload |
|---|---|
| `local_mirror`: 4 allocations mapped per GPU, finish 0.46 s | `reload_weights` 0.75 s |
| Fallback to `rdma` | 4.15 s |
| Fallback to `default` (disk) | 63.2 s |

The output was correct on all three paths.
