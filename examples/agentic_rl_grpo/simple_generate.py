"""Simple custom generate function for agentic RL GRPO with easy datasets.

Plug into slime via ``--custom-generate-function-path``::

    python train.py \\
        --custom-generate-function-path examples.agentic_rl_grpo.simple_generate.simple_grpo_generate \\
        --custom-rm-path examples.agentic_rl_grpo.simple_reward.simple_outcome_reward \\
        ...

与完整的 ``agentic_grpo_generate`` 不同，此版本:

1. 只使用 ``sglang_loop`` 模式（本地 subprocess sandbox，无需 Docker/E2B）。
2. 不启动 Hermes/ClaudeCode harness，不启动 adapter HTTP server。
3. 直接使用 SGLang /generate + token-level logprobs。
4. Reward = 纯 outcome reward（通过 dataset_adapter.evaluate_task()）。

支持的数据集:
  - simple_shell: shell command tasks (terminal-bench style)
  - simple_math: math problems with Python tool
  - simple_code: coding tasks with test-based verification
  - alfworld: text-based household environment
  - terminal_bench: existing terminal-bench tasks (via existing adapter)
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

CONFIG = {
    "eval_timeout_sec": int(os.environ.get("SIMPLE_EVAL_TIMEOUT_SEC", "300")),
    "rollout_guard_sec": int(os.environ.get("SIMPLE_ROLLOUT_GUARD_SEC", "0")) or (
        int(os.environ.get("AGENT_MAX_TURNS", "10")) * 60 + 300
    ),
    "max_turns": int(os.environ.get("AGENT_MAX_TURNS", "10")),
}

# Benchmarks that use the text-based simple agent loop
_TEXT_ENV_BENCHMARKS = frozenset({"alfworld"})

# Benchmarks that use the existing tool-calling agent loop
_TOOL_BENCHMARKS = frozenset({
    "terminal_bench", "simple_shell", "simple_math", "simple_code",
})


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------


async def simple_grpo_generate(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any] | None = None,
    evaluation: bool = False,
) -> list[Sample]:
    """Simple GRPO custom generate function for easy datasets.

    Called by slime's rollout loop for each sample.

    Args:
        args: Slime training arguments.
        sample: Input sample with ``prompt`` and ``metadata``.
        sampling_params: SGLang sampling overrides.
        evaluation: If True, generate eval-only.

    Returns:
        ``list[Sample]`` with reward and logprobs ready for GRPO training.
    """
    import asyncio

    # 1. Resolve benchmark and dataset adapter
    metadata = sample.metadata or {}
    benchmark = metadata.get("benchmark", "")
    if not benchmark:
        benchmark = _auto_detect_benchmark(sample)
        metadata["benchmark"] = benchmark

    from examples.agentic_rl_datasets import get_adapter, import_all

    import_all()
    dataset_adapter = get_adapter(benchmark)

    # 2. Ensure sglang_loop mode
    agent_mode = os.environ.get("SLIME_AGENT_MODE", "sglang_loop")
    if agent_mode != "sglang_loop":
        logger.warning(
            "[simple_grpo] SLIME_AGENT_MODE=%s, but simple datasets only "
            "support sglang_loop. Forcing sglang_loop mode.",
            agent_mode,
        )

    # 3. Create local sandbox
    from examples.agentic_rl.sandbox import SubprocessSandbox

    workdir = metadata.get("workdir", "/home/agent")
    instance_id = metadata.get("instance_id") or metadata.get("task_id", "unknown")

    t0 = time.time()
    try:
        async with asyncio.timeout(CONFIG["rollout_guard_sec"]):
            async with SubprocessSandbox(workdir) as sb:
                # 3a. Setup task environment
                logger.info(
                    "[simple_grpo] Setting up %s/%s", benchmark, instance_id,
                )
                await dataset_adapter.setup_task(sb, metadata)

                # 3b. Run agent loop
                logger.info(
                    "[simple_grpo] Running agent loop for %s/%s",
                    benchmark, instance_id,
                )
                if benchmark in _TEXT_ENV_BENCHMARKS:
                    # Text-based environment: use simple agent loop
                    from examples.agentic_rl_grpo.simple_agent_loop import (
                        run_simple_agent_loop,
                    )

                    trajectory, segment_samples = await run_simple_agent_loop(
                        args, sample, sampling_params or {},
                        dataset_adapter, metadata,
                        max_turns=metadata.get("max_turns", CONFIG["max_turns"]),
                        workdir=workdir,
                    )
                else:
                    # Tool-calling mode: use existing agent loop
                    from examples.agentic_rl.agent_loop import run_agent_loop

                    trajectory, segment_samples = await run_agent_loop(
                        args, sample, sb, sampling_params or {},
                        max_turns=metadata.get("max_turns", CONFIG["max_turns"]),
                        workdir=workdir,
                    )

                # 3c. Evaluate task
                logger.info(
                    "[simple_grpo] Evaluating %s/%s", benchmark, instance_id,
                )
                try:
                    task_reward = await dataset_adapter.evaluate_task(
                        sb, metadata, timeout_sec=CONFIG["eval_timeout_sec"],
                    )
                except Exception:
                    logger.exception(
                        "Task evaluation failed for %s/%s",
                        benchmark, instance_id,
                    )
                    task_reward = 0.0

                if evaluation:
                    if segment_samples:
                        segment_samples[-1].reward = task_reward
                    else:
                        sample.reward = task_reward
                        segment_samples = [sample]
                    logger.info(
                        "[simple_grpo] %s/%s eval: reward=%.3f elapsed=%.1fs",
                        benchmark, instance_id, task_reward, time.time() - t0,
                    )
                    return segment_samples

                # 3d. Stamp reward on all segment samples
                if not segment_samples:
                    logger.warning(
                        "[simple_grpo] %s/%s: no segment samples, returning "
                        "empty result",
                        benchmark, instance_id,
                    )
                    return _abort_result(sample, "no_segments", instance_id)

                for seg in segment_samples:
                    seg.reward = task_reward
                    if seg.metadata is None:
                        seg.metadata = {}
                    seg.metadata["benchmark"] = benchmark
                    seg.metadata["task_eval_reward"] = task_reward
                    seg.metadata["trajectory_length"] = len(trajectory)

                logger.info(
                    "[simple_grpo] %s/%s: reward=%.3f turns=%d "
                    "segments=%d elapsed=%.1fs",
                    benchmark, instance_id, task_reward,
                    len(trajectory), len(segment_samples),
                    time.time() - t0,
                )
                return segment_samples

    except asyncio.TimeoutError:
        logger.warning(
            "[simple_grpo] %s/%s: wall-clock timeout after %.1fs",
            benchmark, instance_id, time.time() - t0,
        )
        return _abort_result(sample, "wall_clock_timeout", instance_id)
    except Exception:
        logger.warning(
            "[simple_grpo] %s/%s: rollout failed:\n%s",
            benchmark, instance_id, traceback.format_exc(),
        )
        return _abort_result(
            sample, f"exception:{traceback.format_exc()[:200]}", instance_id,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_detect_benchmark(sample: Sample) -> str:
    """Guess the benchmark from metadata fields."""
    md = sample.metadata or {}
    if md.get("task_type", "").startswith("pick_"):
        return "alfworld"
    if md.get("task_type") in ("math", "gsm8k", "simple_math"):
        return "simple_math"
    if md.get("test_cases") is not None or md.get("task_type") == "code":
        return "simple_code"
    if md.get("check_command") is not None:
        return "terminal_bench"
    if md.get("setup_commands") is not None:
        return "simple_shell"
    return "simple_shell"


def _abort_result(
    sample: Sample, reason: str, instance_id: str,
) -> list[Sample]:
    """Return an aborted sample when rollout fails."""
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED
    sample.metadata = {
        **(sample.metadata or {}),
        "abort_reason": reason,
        "instance_id": instance_id,
    }
    logger.warning("[simple_grpo] %s aborted: %s", instance_id, reason)
    return [sample]
