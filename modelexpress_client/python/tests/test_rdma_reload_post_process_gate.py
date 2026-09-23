# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reload target must not be put through vLLM post-load processing twice.

``before_rdma_receive`` exists to bring a freshly dummy-allocated target up to
the processed layout the source published. A reload target is already in that
layout -- it was post-processed before the checkpoint -- and vLLM's post-load
processing is not idempotent, so running it again corrupts the model.

Observed on a CRIU-restored Qwen3-Next-80B replica pulling weights from a peer:

    convert_moe_weights_to_flashinfer_trtllm_block_layout
      w13_rows, w13_cols = w13_weight[0].view(torch.uint8).shape
    ValueError: too many values to unpack (expected 2)

That conversion reshapes MoE experts from ``[E, rows, cols]`` to
``[E, cols/block_k, rows, block_k]``. On the second pass it is handed its own
4-D output, so ``w13_weight[0]`` is 3-D and the two-way unpack fails. The RDMA
attempt dies, the chain falls back to reinit_for_retry, and on a model this size
that rebuild then OOMs -- so a non-idempotent reshape surfaces as an
out-of-memory crash several layers away from its cause.

``after_rdma_receive`` was already gated for the same reason; this covers the
matching gate on the way in.

Run: pytest tests/test_rdma_reload_post_process_gate.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modelexpress.load_strategy.rdma_strategy import RdmaStrategy


def _ctx(*, skip_post_process: bool):
    return SimpleNamespace(
        global_rank=0,
        skip_post_process=skip_post_process,
        adapter=MagicMock(),
    )


def _result():
    return SimpleNamespace(skip_allocate=False, model=MagicMock())


@pytest.fixture
def strategy(monkeypatch):
    strat = RdmaStrategy()
    # The transfer itself is not under test; only which hooks run around it.
    monkeypatch.setattr(strat, "_receive_from_peer", lambda *a, **k: None)
    return strat


def test_reload_skips_post_load_processing(strategy):
    """The restore path: the model is already processed, so leave it alone."""
    ctx = _ctx(skip_post_process=True)
    ctx.adapter.prepare_rdma_target.side_effect = lambda r: r

    strategy._load_as_target(_result(), ctx, MagicMock(), "src-id", "worker-id")

    ctx.adapter.before_rdma_receive.assert_not_called()
    ctx.adapter.after_rdma_receive.assert_not_called()


def test_cold_load_still_runs_post_load_processing(strategy):
    """The gate must not disarm the cold path, where the work is required.

    A dummy-allocated target has none of the processed layout, so skipping here
    would hand the source's weights to a model whose tensors are the wrong shape.
    """
    ctx = _ctx(skip_post_process=False)
    ctx.adapter.prepare_rdma_target.side_effect = lambda r: r
    ctx.adapter.before_rdma_receive.side_effect = lambda r: r

    strategy._load_as_target(_result(), ctx, MagicMock(), "src-id", "worker-id")

    ctx.adapter.before_rdma_receive.assert_called_once()
    ctx.adapter.after_rdma_receive.assert_called_once()


def test_target_allocation_is_skipped(strategy):
    """RDMA receives into buffers the target already owns, so never allocate."""
    ctx = _ctx(skip_post_process=True)
    captured = {}

    def _capture(r):
        captured["skip_allocate"] = r.skip_allocate
        return r

    ctx.adapter.prepare_rdma_target.side_effect = _capture

    strategy._load_as_target(_result(), ctx, MagicMock(), "src-id", "worker-id")

    assert captured["skip_allocate"] is True
