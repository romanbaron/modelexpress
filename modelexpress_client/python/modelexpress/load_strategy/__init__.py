# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
load_strategy: prioritized chain of model loading strategies.

Detects the environment and builds an ordered list of eligible loaders.
MxModelLoader iterates the chain until one succeeds.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch.nn as nn

from modelexpress.tracing import tracer

from ..adapter import StrategyFailed, StrategyRecoveryError, UnsupportedCapability
from .base import (
    LoadContext,
    LoadResult,
    LoadStrategy,
    SourceTransferError,
    clear_exception_tracebacks,
    publish_source_if_supported,
    register_tensors,
    publish_metadata,
    unpublish_metadata,
)

__all__ = [
    "LoadContext",
    "LoadResult",
    "LoadStrategy",
    "LoadStrategyChain",
    "execute_load_strategies",
    "run_load_strategy_chain",
    "SourceTransferError",
    "register_tensors",
    "publish_metadata",
    "unpublish_metadata",
]

logger = logging.getLogger("modelexpress.load_strategy")


class LoadStrategyChain:
    """Prioritized chain of model loading strategies.

    Detects the environment, builds an ordered list of eligible loaders,
    and runs them until one succeeds.
    """

    @staticmethod
    def run(model: nn.Module, ctx: LoadContext) -> nn.Module:
        """Build the chain and execute strategies until one succeeds.

        Strategies return LoadResult on success. Expected misses raise
        StrategyFailed; mutated failures trigger adapter re-initialization
        before the next strategy runs. Unexpected exceptions are rolled back
        and treated as fallback to preserve the existing chain behavior.

        Returns the (possibly re-initialized) model on success.
        Raises RuntimeError if no strategy succeeds.
        """
        from .rdma_strategy import RdmaStrategy
        from .server_cache_strategy import ServerCacheStrategy
        from .instant_tensor_strategy import InstantTensorStrategy
        from .model_streamer_strategy import ModelStreamerStrategy
        from .gds_strategy import GdsStrategy
        from .default_strategy import DefaultStrategy

        all_strategies: list[LoadStrategy] = [
            RdmaStrategy(),
            ServerCacheStrategy(),
            InstantTensorStrategy(),
            ModelStreamerStrategy(),
            GdsStrategy(),
            DefaultStrategy(),
        ]
        return execute_load_strategies(model, ctx, all_strategies)

    @staticmethod
    def _reinit_for_retry(
        result: LoadResult,
        ctx: LoadContext,
        strategy: LoadStrategy,
    ) -> LoadResult:
        if ctx.adapter is None:
            raise RuntimeError(
                f"[Worker {ctx.global_rank}] Strategy '{strategy.name}' mutated "
                "the model but no adapter can reinitialize it"
            )
        try:
            return ctx.adapter.reinit_for_retry(result)
        except UnsupportedCapability as exc:
            raise RuntimeError(
                f"[Worker {ctx.global_rank}] Strategy '{strategy.name}' mutated "
                "the model but adapter does not support retry reinitialization"
            ) from exc


def _run_phase(phase, result: LoadResult, ctx: LoadContext) -> None:
    """Run one phase, reporting any failure as a mutated one.

    Both phases write into the model, so a strategy that falls through after
    one failed leaves the next needing a reinitialized model -- the same
    contract these calls carried when they lived inside load().
    """
    try:
        phase(result, ctx)
    except StrategyFailed:
        raise
    except Exception as e:
        raise StrategyFailed(str(e), mutated=True) from e


@contextmanager
def _strategy_phases(
    strategy: LoadStrategy,
    result: LoadResult,
    ctx: LoadContext,
):
    """Wrap one attempt in the phases its situation calls for.

    A strategy declares only how it delivers weights; which phases wrap it is
    decided here, so no strategy has to branch on whether this is a cold load
    or a reload.

    Cold load: the strategy's own prepare() gets the model into the state it
    needs, and finalize() converts the raw weights it loaded into their final
    runtime layout. finalize() sits after the yield rather than in a finally,
    because a failed attempt has nothing to bring to a final layout.

    Reload: the model is already in that layout and already has real storage,
    so neither applies. Strategies that hand weights to the engine through its
    own load_weights() callbacks run inside the engine's layerwise reload
    instead, which materializes and processes each layer as its weights
    arrive; that one is left in a finally, so a fallback never starts against
    a model still deferred to meta. Strategies that write straight into the
    existing buffers (RDMA, GDS) receive weights the source already processed
    and need no phase at all.

    Every phase runs through the adapter, so without one there is nothing to
    wrap either way. The phases work on ``result`` in place: LoadResult is the
    stable envelope the chain holds for the whole attempt, which is why
    reinit_for_retry copies a replacement's state back into it rather than
    handing one out.
    """
    if ctx.adapter is None:
        yield
        return

    if ctx.is_reload:
        if not strategy.delivers_via_load_weights:
            yield
            return

        ctx.adapter.begin_streaming_reload(result)
        try:
            yield
        finally:
            ctx.adapter.end_streaming_reload(result)
        return

    _run_phase(strategy.prepare, result, ctx)
    yield
    _run_phase(strategy.finalize, result, ctx)


def _run_strategy_attempt(
    strategy: LoadStrategy,
    result: LoadResult,
    ctx: LoadContext,
) -> LoadResult:
    """Run one attempt from end to end: its phases, and the load between them."""
    with _strategy_phases(strategy, result, ctx):
        return strategy.load(result, ctx)


def execute_load_strategies(
    model: nn.Module,
    ctx: LoadContext,
    strategies: list[LoadStrategy],
) -> nn.Module:
    """Execute an ordered policy using the common fallback lifecycle."""
    eligible = [strategy for strategy in strategies if strategy.is_available(ctx)]
    logger.info(f"Eligible loaders: {[strategy.name for strategy in eligible]}")

    result = LoadResult(value=model, model=model)
    with tracer.start_as_current_span("Load model") as span:
        span.set_attribute("model_name", ctx.identity.model_name)
        span.set_attribute("global_rank", ctx.global_rank)
        span.set_attribute("eligible_strategies", [s.name for s in eligible])

        for strategy in eligible:
            logger.info(f"[Worker {ctx.global_rank}] Trying strategy: {strategy.name}")
            try:
                with _strategy_phases(strategy, result, ctx):
                    result = strategy.load(result, ctx)
                # Discovery has to see the model in its final layout, which is
                # only true once the phases above are done. RDMA is the
                # exception: its target buffers must already be registered for
                # the source to write into, so it registers mid-transfer.
                if (
                    not strategy.registers_tensors_during_load
                    and result.model_for_publish is not None
                ):
                    register_tensors(result, ctx)
                publish_source_if_supported(result, ctx)
                span.set_attribute("weight_loading_strategy", strategy.name)
                return result.value
            except StrategyRecoveryError:
                strategy.rollback(ctx)
                raise
            except StrategyFailed as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Strategy {strategy.name} failed, "
                    f"trying next: {e}"
                )
                strategy.rollback(ctx)
                if e.mutated:
                    clear_exception_tracebacks(e)
                    result = LoadStrategyChain._reinit_for_retry(result, ctx, strategy)
                continue
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Strategy {strategy.name} "
                    f"raised unexpected error, trying next: {e}"
                )
                strategy.rollback(ctx)

    raise RuntimeError(
        f"[Worker {ctx.global_rank}] No loading strategy succeeded "
        f"for model '{ctx.identity.model_name}'"
    )


def run_load_strategy_chain(model: nn.Module, ctx: LoadContext) -> nn.Module:
    """Dispatch to the configured engine-neutral loading policy."""
    from .. import envs

    chain = envs.MX_LOAD_STRATEGY_CHAIN
    if chain == "INFERENCE":
        return LoadStrategyChain.run(model, ctx)
    if chain == "RL":
        from modelexpress_rl.inference.load_strategy import RLLoadStrategyChain

        return RLLoadStrategyChain.run(model, ctx)
    raise ValueError(
        "MX_LOAD_STRATEGY_CHAIN must be 'INFERENCE' or 'RL', "
        f"got {chain!r}"
    )
