"""Reward functions for agentic RL.

Provides the ``agentic_reward`` entry point for ``--custom-rm-path``,
dispatching to the correct benchmark adapter's ``evaluate_task``.

Since rewards are computed inside ``agentic_generate`` (in the generate step),
this module's function is a NO-OP passthrough — the reward is already set
on the ``Sample`` by the time it reaches the reward model hub.

If you use ``--n-samples-per-prompt 2`` with the default rollout loop (instead
of the custom generate function), this reward function can compute task rewards
independently.
"""

from __future__ import annotations

import logging
from typing import Any

from slime.utils.types import Sample

from examples.agentic_rl_datasets import get_adapter, import_all

logger = logging.getLogger(__name__)


async def agentic_reward(args: Any, sample: Sample) -> float:
    """Reward function for ``--custom-rm-path``.

    If the sample already has a reward (set by ``agentic_generate``), return it
    as-is.  Otherwise, resolve the benchmark adapter and evaluate.

    This allows the reward to be set either:
    - During generation (preferred, inside the sandbox after trajectory is done)
    - During the reward model pass (fallback, when using default generate)
    """
    # If the generate function already set the reward, use it
    if sample.reward is not None:
        return float(sample.reward)

    # Fallback: need to evaluate from scratch (requires sandbox)
    metadata = sample.metadata or {}
    benchmark = metadata.get("benchmark", "")
    if not benchmark:
        logger.warning("No benchmark in metadata and no reward set; returning 0")
        return 0.0

    import_all()
    adapter = get_adapter(benchmark)

    # For evaluation we need a sandbox — this path is only hit when NOT using
    # the custom generate function
    from examples.agentic_rl.sandbox import create_sandbox

    async with create_sandbox(args) as sandbox:
        await adapter.setup_task(sandbox, metadata)
        reward = await adapter.evaluate_task(sandbox, metadata)

    return reward
