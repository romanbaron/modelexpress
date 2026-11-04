#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Mirror every GPU of an engine's node: one nixl_mirror.py per GPU, rank r
# copying the agent at BASE_PORT + r onto GPU r.
#
# Usage: mirror-node.sh BASE_HOST:BASE_PORT NUM_GPUS
#
# Env:
#   NIXL_PORT      base listen port of the mirrors' own NIXL agents; rank r
#                  listens on NIXL_PORT + r (default 7555, clear of an engine's
#                  5555 + r on the same node)
#   MIRROR_SOCKET  local socket path, {rank} replaced by r
#                  (default /tmp/mx-local-mirror-{rank}.sock)
#
# A mirror serves until killed, so any mirror exiting is a failure: the rest
# are stopped and this exits non-zero, so a partial mirror never serves.
set -u

if [ $# -ne 2 ]; then
  echo "usage: $0 BASE_HOST:BASE_PORT NUM_GPUS" >&2
  exit 2
fi
host=${1%:*}
port=${1##*:}
gpus=$2
nixl_port=${NIXL_PORT:-7555}
socket_template=${MIRROR_SOCKET:-/tmp/mx-local-mirror-\{rank\}.sock}
here=$(cd "$(dirname "$0")" && pwd)

stop() {
  trap - TERM INT
  kill $(jobs -p) 2>/dev/null
  wait
}
trap 'stop; exit 143' TERM INT

# Each mirror sees only its own GPU (as ordinal 0), so its CUDA driver
# initialises one device, not all of them. GPU r is the r-th entry of an
# existing CUDA_VISIBLE_DEVICES, or ordinal r.
IFS=, read -r -a visible <<< "${CUDA_VISIBLE_DEVICES:-}"

for r in $(seq 0 $((gpus - 1))); do
  CUDA_VISIBLE_DEVICES=${visible[$r]:-$r} python3 -u "$here/nixl_mirror.py" \
    --endpoint "$host:$((port + r))" \
    --gpu 0 \
    --nixl-port $((nixl_port + r)) \
    --socket "${socket_template//\{rank\}/$r}" &
done

wait -n
status=$?
echo "[mirror-node] a mirror exited (status $status); stopping the rest" >&2
stop
exit 1
