# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which phases wrap a strategy is the chain's decision, not the strategy's.

A strategy declares only how it delivers weights. The chain turns that into
the phases an attempt needs, so no strategy branches on whether it was called
for a cold load or a reload:

    cold load   prepare -> load -> finalize -> register
    reload      (layerwise reload) -> load -> (layerwise reload)

Registration has to see the model in its final layout, so it follows the
phases rather than sitting inside load(). RDMA is the exception both ways: its
target buffers must carry registrations before the source can write into them,
so it registers mid-transfer and declares itself out of the chain's step.

Run: pytest tests/test_load_phases.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from modelexpress.adapter import StrategyFailed
from modelexpress.load_strategy import _run_strategy_attempt, execute_load_strategies
from modelexpress.load_strategy.base import LoadStrategy
from modelexpress.load_strategy.context import LoadResult


class _RecordingStrategy(LoadStrategy):
    """Records the order its phases ran in."""

    name = "recording"

    def __init__(self, calls, *, fail_in=None):
        self.calls = calls
        self.fail_in = fail_in

    def prepare(self, result, ctx):
        self.calls.append("prepare")
        if self.fail_in == "prepare":
            raise RuntimeError("prepare blew up")

    def load(self, result, ctx):
        self.calls.append("load")
        if self.fail_in == "load":
            raise StrategyFailed("load blew up", mutated=False)
        return result

    def finalize(self, result, ctx):
        self.calls.append("finalize")
        if self.fail_in == "finalize":
            raise RuntimeError("finalize blew up")


def _ctx(*, is_reload=False):
    return SimpleNamespace(
        global_rank=0,
        is_reload=is_reload,
        adapter=MagicMock(),
        identity=SimpleNamespace(model_name="m"),
    )


def _result():
    model = MagicMock()
    return LoadResult(value=model, model=model)


def test_cold_load_runs_prepare_then_load_then_finalize():
    calls = []
    _run_strategy_attempt(_RecordingStrategy(calls), _result(), _ctx())
    assert calls == ["prepare", "load", "finalize"]


def test_reload_replaces_both_phases_with_layerwise_reload():
    """A streaming strategy's raw weights are processed per layer instead."""
    calls = []
    ctx = _ctx(is_reload=True)

    _run_strategy_attempt(_RecordingStrategy(calls), _result(), ctx)

    assert calls == ["load"]
    ctx.adapter.begin_streaming_reload.assert_called_once()
    ctx.adapter.end_streaming_reload.assert_called_once()


def test_layerwise_reload_is_left_even_when_the_attempt_fails():
    """A fallback must not start against a half-materialized model."""
    calls = []
    ctx = _ctx(is_reload=True)

    with pytest.raises(StrategyFailed):
        _run_strategy_attempt(_RecordingStrategy(calls, fail_in="load"), _result(), ctx)

    ctx.adapter.end_streaming_reload.assert_called_once()


@pytest.mark.parametrize("phase", ["prepare", "finalize"])
def test_a_phase_failure_is_reported_as_mutated(phase):
    """Both phases write to the model, so a fallback needs a reinit first."""
    calls = []

    with pytest.raises(StrategyFailed, match=f"{phase} blew up") as exc:
        _run_strategy_attempt(_RecordingStrategy(calls, fail_in=phase), _result(), _ctx())

    assert exc.value.mutated is True


def test_registration_follows_the_phases():
    """Discovery only means anything once the model is in its final layout."""
    calls = []
    strategy = _RecordingStrategy(calls)
    ctx = _ctx()

    with (
        patch("modelexpress.load_strategy.register_tensors") as register,
        patch("modelexpress.load_strategy.publish_source_if_supported"),
    ):
        register.side_effect = lambda *a, **k: calls.append("register")
        execute_load_strategies(MagicMock(), ctx, [strategy])

    assert calls == ["prepare", "load", "finalize", "register"]


def test_a_strategy_that_registers_during_load_is_not_registered_again():
    """RDMA already registered its target buffers to receive into them."""
    calls = []
    strategy = _RecordingStrategy(calls)
    strategy.registers_tensors_during_load = True

    with (
        patch("modelexpress.load_strategy.register_tensors") as register,
        patch("modelexpress.load_strategy.publish_source_if_supported"),
    ):
        execute_load_strategies(MagicMock(), _ctx(), [strategy])

    register.assert_not_called()
