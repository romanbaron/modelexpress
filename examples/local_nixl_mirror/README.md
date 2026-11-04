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

This is the first strategy in the chain. To enable it, set
`MX_LOCAL_MIRROR_RELOAD_SOCKET`, for example
`/tmp/mx-local-mirror-{rank}.sock`; `{rank}` is replaced with the local GPU
ordinal.

The strategy only runs when all of these are true:

- the load is a reload;
- every weights allocation of the engine has memory (the engine woke them);
- the mirror answers from the same GPU UUID with a non-empty copy.

Each weights allocation whose address and size match a mirror region gets
that region's FD imported and mapped behind its address, in place of the
engine's own memory. Anything left over is finished by tensor name from the
mirror's copy:

- allocations without a matching region keep their own memory and are
  written;
- regions without a matching allocation are copied by name.

The engine's own memory is only unmapped during the swap. It is released
once everything succeeded. On failure, the strategy:

1. maps the engine's own memory back, unchanged;
2. asks the mirror to release its copy, so the GPU has room for the KV cache;
3. raises `StrategyFailed(mutated=False)`, and the next strategy loads the
   weights.

## vLLM requirements

- Sleep mode must be on (`CuMemAllocator`). The strategy reads its records
  (`pointer_to_data`, the `(device, size, address, p_handle)` handle) and
  rewrites the handle slot, so a later sleep or free releases the mirror's
  memory like the engine's own.
- The reload follows vLLM's usual order after a level-2 sleep:
  `wake_up(tags=["weights"])`, then `reload_weights`, then
  `wake_up(tags=["kv_cache"])`. The weights wake restores the buffers saved at
  sleep, so any strategy, including a fallback, reloads the weights over
  correct buffers.
- Memory: between the weights wake and the swap, the GPU holds the engine's
  fresh weights and the mirror's copy, about twice the weights. The KV cache
  must stay asleep until after the reload, which frees the engine's copy.

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
| `local_mirror`: 4 allocations swapped per GPU | `reload_weights` 0.71 s |
| Fallback to `rdma` | 3.71 s |
| Fallback to `default` (disk) | 61.0 s |

The output was correct on all three paths.
