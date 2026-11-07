"""The compiled forward without ModelRunner.forward's envelope.

A spec round issues two to three forwards; at bs=1 the host launches, not the
chip, bound the round, and the envelope (profiling spans, expert and indexer
capturers, graph-runner checks, the eager runner's batch copy) does nothing for
them. This runs the same preparation the eager runner does and calls the
compiled model forward directly.
"""

from __future__ import annotations

import contextlib

from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner import EagerRunner
from sglang.srt.runtime_context import get_exec, get_global_dwdp_manager, get_parallel
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer
from sglang_tpu.sampler import select_logits_output_mode


def lean_forward_allowed(runner: ModelRunner) -> bool:
    """Whether ModelRunner.forward's envelope is a no-op for this runner."""
    return (
        runner.canary_manager is None
        and runner.hisparse_coordinator is None
        and runner.eplb_manager is None
        and runner.lora_manager is None
        and not runner.enable_elastic_ep
        and runner.decode_cuda_graph_runner is None
        # The target's prefill "graph runner" is the eager runner when graphs are off.
        and (
            runner.prefill_cuda_graph_runner is None
            or isinstance(runner.prefill_cuda_graph_runner, EagerRunner)
        )
        and runner.msprobe_debugger is None
        and get_global_experts_capturer() is None
        and get_global_indexer_capturer() is None
        and get_global_dwdp_manager() is None
        and get_exec().moe.elastic_ep_backend is None
        and get_parallel().attn_dcp_size == 1
        and not dumper.may_enable
    )


def lean_forward(
    runner: ModelRunner, forward_batch: ForwardBatch
) -> LogitsProcessorOutput:
    """One TARGET_VERIFY forward at the batch bucket, prepared as the eager runner
    prepares an extend: DP/attn-TP padding, the model's batch prep, attention
    metadata, then the compiled forward."""
    assert forward_batch.forward_mode.is_target_verify(), forward_batch.forward_mode
    runner.forward_pass_id += 1
    ctx = (
        contextlib.nullcontext()
        if has_forward_context()
        else forward_context(ForwardContext(attn_backend=runner.attn_backend))
    )
    with ctx:
        runner._prepare_eager_forward_batch(forward_batch)
        runner.model.prepare_forward_batch(forward_batch)
        runner.attn_backend.init_forward_metadata(forward_batch)
        forward_batch.attn_cp_metadata = None
        forward_batch.logits_output_mode = select_logits_output_mode(
            forward_batch, runner.model
        )
        kwargs = runner._extend_forward_kwargs(forward_batch, None)
        return runner.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch, **kwargs
        )
