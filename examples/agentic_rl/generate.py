"""Custom generate function for agentic RL with online DPO.

This module provides the ``agentic_generate`` entry point, designed to be used
with slime's ``--custom-generate-function-path``::

    python train.py --custom-generate-function-path examples.agentic_rl.generate.agentic_generate ...

For each prompt, the function generates TWO agent trajectories (chosen/rejected),
evaluates both via the benchmark's task reward, and returns a pair of ``Sample``
objects where the higher-reward trajectory is marked "chosen".
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from typing import Any

from slime.utils.types import Sample

from examples.agentic_rl.agent_loop import run_agent_loop
from examples.agentic_rl.sandbox import create_sandbox

logger = logging.getLogger(__name__)


async def agentic_generate(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any] | None = None,
    evaluation: bool = False,
) -> list[Sample]:
    """Custom generate function for agentic RL.

    Called by slime's rollout loop via ``--custom-generate-function-path``.
    For each input sample:
      1. Determines which benchmark adapter to use (from ``sample.metadata``).
      2. Prepares a sandbox (setup task environment).
      3. Generates TWO independent agent trajectories with different seeds.
      4. Evaluates both against the task reward.
      5. Returns a ``list[Sample]`` with the pair (chosen + rejected) marked
         via ``train_metadata``.

    Args:
        args: Slime training arguments.
        sample: Input prompt wrapped in a ``Sample``.
        sampling_params: SGLang sampling overrides.
        evaluation: If True, generate only one trajectory (no DPO pair).

    Returns:
        List of ``Sample`` objects:
        - ``len=2`` for training (chosen + rejected), same ``group_index``
        - ``len=1`` for evaluation
    """
    # 1. Resolve benchmark adapter
    metadata = sample.metadata or {}
    benchmark = metadata.get("benchmark", "")
    if not benchmark:
        logger.warning("No benchmark specified in sample.metadata, trying auto-detection")
        benchmark = _auto_detect_benchmark(sample)

    adapter = _get_adapter(benchmark)

    # 2. How many trajectories to generate
    num_trajs = 1 if evaluation else 2

    # 3. Generate N trajectories with different random seeds
    tasks = []
    for i in range(num_trajs):
        seed = (getattr(args, "seed", 42) + i) % (2**31)
        tasks.append(
            _generate_one_trajectory(
                args, deepcopy(sample), sampling_params, adapter, seed=seed,
            )
        )

    results: list[tuple[list[dict], list[Sample], float]] = await asyncio.gather(*tasks)

    # 4. Collect all samples with rewards
    all_samples: list[Sample] = []
    for _trajectory, trajectory_samples, reward in results:
        # Assign reward to the last segment sample (which contains full tokens)
        if trajectory_samples:
            final_sample = trajectory_samples[-1]
            final_sample.reward = reward
            # Mark metadata
            if final_sample.train_metadata is None:
                final_sample.train_metadata = {}
            final_sample.train_metadata["benchmark"] = benchmark
            final_sample.train_metadata["adapter"] = adapter.name if hasattr(adapter, "name") else benchmark
            all_samples.append(final_sample)

    # 5. For training, mark chosen/rejected
    if not evaluation and len(all_samples) == 2:
        traj_a, traj_b = all_samples
        # Higher reward = chosen
        if (traj_a.reward or 0.0) >= (traj_b.reward or 0.0):
            chosen, rejected = traj_a, traj_b
        else:
            chosen, rejected = traj_b, traj_a

        chosen.train_metadata["is_chosen"] = True
        chosen.train_metadata["pair_index"] = 0
        rejected.train_metadata["is_chosen"] = False
        rejected.train_metadata["pair_index"] = 0

        # Ensure both share the same group_index so the DPO loss can pair them
        if chosen.group_index is None and rejected.group_index is None:
            chosen.group_index = sample.index or 0
            rejected.group_index = sample.index or 0

        logger.info(
            "DPO pair: prompt=%s chosen_reward=%.3f rejected_reward=%.3f",
            str(sample.prompt)[:80], chosen.reward or 0.0, rejected.reward or 0.0,
        )
        return [chosen, rejected]

    return all_samples


async def _generate_one_trajectory(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any] | None,
    adapter: Any,
    *,
    seed: int,
) -> tuple[list[dict], list[Sample], float]:
    """Generate one agent trajectory and evaluate it."""
    import random
    import numpy as np

    # Set deterministic seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    metadata = sample.metadata or {}
    workdir = metadata.get("workdir", "/home/agent")

    async with create_sandbox(args) as sandbox:
        # 1. Setup task environment
        logger.info("Setting up task (seed=%d) ...", seed)
        await adapter.setup_task(sandbox, metadata)

        # 2. Run agent loop
        logger.info("Running agent loop (seed=%d) ...", seed)
        trajectory, segment_samples = await run_agent_loop(
            args, sample, sandbox, sampling_params or {},
            max_turns=metadata.get("max_turns", 10),
            workdir=workdir,
        )

        # 3. Evaluate task (rule-based)
        logger.info("Evaluating task (seed=%d) ...", seed)
        reward = await adapter.evaluate_task(sandbox, metadata)

        # 4. LLM-judge (if enabled)
        if getattr(args, "llm_judge", False):
            logger.info("LLM-judge (seed=%d) ...", seed)
            try:
                llm_reward = await adapter.llm_judge(
                    trajectory, metadata, args,
                )
                if llm_reward is not None:
                    from examples.agentic_rl.llm_judge import combine_rewards

                    llm_weight = getattr(args, "llm_judge_weight", 0.5)
                    combined = combine_rewards(reward, llm_reward, llm_weight)
                    logger.info(
                        "LLM-judge: rule=%.3f llm=%.3f combined=%.3f (w=%.2f)",
                        reward, llm_reward, combined, llm_weight,
                    )
                    reward = combined
                else:
                    logger.info("LLM-judge returned None (adapter does not support it)")
            except Exception:
                logger.exception("LLM-judge failed, using rule-based reward only")

    return trajectory, segment_samples, reward


def _auto_detect_benchmark(sample: Sample) -> str:
    """Guess the benchmark from metadata fields when not explicitly set."""
    md = sample.metadata or {}
    if md.get("instance_id") or md.get("repo"):
        # R2E-Gym has FAIL_TO_PASS / test_patch as flat keys;
        # SWE-bench typically wraps them inside remote_env_info.
        if md.get("FAIL_TO_PASS") is not None or (
            isinstance(md.get("test_patch"), str)
            and md.get("test_patch", "").startswith("diff")
        ):
            return "r2e_gym"
        return "swe_gym_lite"
    if md.get("task_type") in (
        "os", "db", "kg", "dcg", "ltp", "hh", "ws", "wb",
    ):
        return "agent_bench"
    if md.get("env") in ("retail", "airline"):
        return "tau_bench"
    if md.get("check_command") is not None:
        return "terminal_bench"
    if md.get("check_script") is not None:
        return "cli_gym"
    if md.get("api_spec"):
        return "api_bank"
    logger.warning("Could not auto-detect benchmark from metadata=%s", md)
    return "swe_gym_lite"


def _get_adapter(benchmark: str) -> Any:
    """Lazy-import and return the adapter for *benchmark*."""
    from examples.agentic_rl_datasets import get_adapter, import_all
    import_all()
    return get_adapter(benchmark)
