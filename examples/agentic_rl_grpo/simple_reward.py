"""Simple outcome reward functions for agentic RL GRPO.

与完整的多维 reward 系统不同，这里只使用简单的 outcome reward:

- **Terminal 任务** (terminal_bench, simple_shell): rule-based verifier
  (check_command exit code / expected output matching) → 0 or 1
- **Math 任务** (simple_math): answer matching against ground truth → 0 or 1
- **Code 任务** (simple_code): test pass rate → [0, 1]
- **ALFWorld** (alfworld): task success check → 0 or 1
- **开放性问题** (open_qa): LLM judge score → [0, 1]

Reward 计算在 generate 阶段完成（通过 dataset_adapter.evaluate_task()），
此模块提供 ``--custom-rm-path`` 所需的 pass-through 函数。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def simple_outcome_reward(args: Any, sample: Any) -> float:
    """Simple outcome reward pass-through for ``--custom-rm-path``.

    由于 outcome reward 已在 generate 阶段通过 ``dataset_adapter.evaluate_task()``
    计算并写入 ``sample.reward``，此函数直接返回已计算好的 reward。

    如果是 evaluation 或 fallback 路径（sample.reward 未设置），返回 0。

    Args:
        args: Slime training args.
        sample: A ``Sample`` object, should already have ``.reward`` set.

    Returns:
        Scalar reward in [0, 1].
    """
    if sample.reward is not None:
        try:
            return float(sample.reward)
        except (TypeError, ValueError):
            logger.warning("sample.reward is not a float: %r", sample.reward)
            return 0.0

    # Fallback: try to extract reward from metadata
    metadata = sample.metadata or {}
    task_eval_reward = metadata.get("task_eval_reward")
    if task_eval_reward is not None:
        try:
            return float(task_eval_reward)
        except (TypeError, ValueError):
            pass

    logger.warning(
        "No reward on sample %s; returning 0",
        getattr(sample, "index", "?"),
    )
    return 0.0
