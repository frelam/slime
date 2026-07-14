"""Multi-dimensional reward composer for agentic RL (GRPO/PPO).

**This reward is currently registered for general tasks only** (terminal_bench,
cli_gym, tau_bench, api_bank, agent_bench, etc.).

Combines verifier scores (rule-based dimensions 4.2, 4.3, 4.4, 4.7) with
reward model scores (LLM-judged dimensions 4.1, 4.5, 4.6) into a single
scalar reward in [0, 1] via weighted sum.

SWE tasks (swe_gym_lite, r2e_gym) use the raw task evaluation reward
(test pass rate) directly — multi-dimensional reward rules for SWE will
be added later.

Weights are configurable via ``--reward-weights`` JSON argument. Defaults
implement the user's specification:

===========  ==============================  ======  ========
Dimension    Name                            Weight  Source
===========  ==============================  ======  ========
4.1          Answer correctness              0.51    RM
4.2          Format compliance               0.15    Verifier
4.3          Tool call param correctness     0.10    Verifier
4.4          Tool call retry behavior        0.05    Verifier
4.5          Planning quality                0.075   RM
4.6          Hallucination                   0.075   RM
4.7          Tool call count penalty         0.05    Verifier
===========  ==============================  ======  ========
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "correctness": 0.51,
    "format": 0.15,
    "tool_params": 0.10,
    "retry": 0.05,
    "planning": 0.075,
    "hallucination": 0.075,
    "tool_count": 0.05,
}


def get_weights(args: Any) -> dict[str, float]:
    """Resolve reward dimension weights from args or defaults.

    Args:
        args: Slime training args. Reads ``--reward-weights`` if set.

    Returns:
        Dict mapping dimension name → weight.
    """
    raw = getattr(args, "reward_weights", None)
    if raw is None:
        return dict(DEFAULT_WEIGHTS)

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid --reward-weights JSON, using defaults: %r", raw)
            return dict(DEFAULT_WEIGHTS)

    if not isinstance(raw, dict):
        logger.warning("--reward-weights is not a dict, using defaults: %r", raw)
        return dict(DEFAULT_WEIGHTS)

    # Merge with defaults so partial overrides work
    merged = dict(DEFAULT_WEIGHTS)
    for key in DEFAULT_WEIGHTS:
        if key in raw:
            merged[key] = float(raw[key])

    # Normalize to sum to 1.0
    total = sum(merged.values())
    if total > 0:
        merged = {k: v / total for k, v in merged.items()}

    return merged


# ---------------------------------------------------------------------------
# Reward breakdown
# ---------------------------------------------------------------------------


@dataclass
class RewardBreakdown:
    """All seven dimension scores plus the weighted total."""

    total: float
    correctness: float
    format_compliance: float
    tool_param_correctness: float
    tool_retry: float
    planning_quality: float
    hallucination: float
    tool_count_penalty: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Return a flat dict suitable for W&B logging."""
        return {
            "reward/total": self.total,
            "reward/correctness": self.correctness,
            "reward/format_compliance": self.format_compliance,
            "reward/tool_param_correctness": self.tool_param_correctness,
            "reward/tool_retry": self.tool_retry,
            "reward/planning_quality": self.planning_quality,
            "reward/hallucination": self.hallucination,
            "reward/tool_count_penalty": self.tool_count_penalty,
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def compute_multi_dimensional_reward(
    args: Any,
    trajectory: list[dict[str, Any]],
    task_description: str,
    task_type: str,
    *,
    task_eval_reward: float | None = None,
) -> RewardBreakdown:
    """Compute the multi-dimensional reward for one agent trajectory.

    Args:
        args: Slime training args.
        trajectory: Normalized trajectory (list of turn/observation dicts).
        task_description: The original task prompt / problem statement.
        task_type: Benchmark name, e.g. ``"terminal_bench"``, ``"swe_gym_lite"``.
        task_eval_reward: Task execution reward from ``adapter.evaluate_task()``
            (e.g., test pass rate for SWE, check command result for terminal).
            Used as fallback for correctness when RM is unavailable.

    Returns:
        ``RewardBreakdown`` with all dimension scores and the weighted total.
    """
    from examples.agentic_rl_grpo.reward_model import call_reward_model
    from examples.agentic_rl_grpo.verifier import compute_verifier_scores

    weights = get_weights(args)

    # ---- RM dimensions (4.1, 4.5, 4.6) ----

    rm_result = await call_reward_model(
        args, task_type, trajectory, task_description,
    )

    correctness = rm_result.correctness
    planning = rm_result.planning
    hallucination = rm_result.hallucination

    # If RM is unavailable (neutral scores) and we have a task eval reward,
    # override the correctness dimension with the task eval result.
    if (
        task_eval_reward is not None
        and rm_result.reason == "RM unavailable — neutral scores"
    ):
        correctness = 1.0 if task_eval_reward >= 0.5 else 0.0

    # ---- Verifier dimensions (4.2, 4.3, 4.4, 4.7) ----

    answer_correct = correctness >= 0.5
    verifier_scores = compute_verifier_scores(
        trajectory, answer_correct=answer_correct, task_type=task_type,
    )

    # ---- Weighted sum ----

    total = (
        weights["correctness"] * correctness
        + weights["format"] * verifier_scores["format_compliance"]
        + weights["tool_params"] * verifier_scores["tool_param_correctness"]
        + weights["retry"] * verifier_scores["tool_retry"]
        + weights["planning"] * planning
        + weights["hallucination"] * hallucination
        + weights["tool_count"] * verifier_scores["tool_count_penalty"]
    )

    total = max(0.0, min(1.0, total))

    breakdown = RewardBreakdown(
        total=total,
        correctness=correctness,
        format_compliance=verifier_scores["format_compliance"],
        tool_param_correctness=verifier_scores["tool_param_correctness"],
        tool_retry=verifier_scores["tool_retry"],
        planning_quality=planning,
        hallucination=hallucination,
        tool_count_penalty=verifier_scores["tool_count_penalty"],
        details={
            "weights": weights,
            "rm_reason": rm_result.reason,
            "task_eval_reward": task_eval_reward,
        },
    )

    logger.info(
        "Multi-dim reward [%s]: total=%.3f "
        "correctness=%.0f format=%.0f params=%.0f retry=%.1f "
        "planning=%.2f halluc=%.0f count_pen=%.3f",
        task_type,
        total,
        correctness,
        verifier_scores["format_compliance"],
        verifier_scores["tool_param_correctness"],
        verifier_scores["tool_retry"],
        planning,
        hallucination,
        verifier_scores["tool_count_penalty"],
    )

    return breakdown


# ---------------------------------------------------------------------------
# Custom RM path adapter (for --custom-rm-path)
# ---------------------------------------------------------------------------


async def agentic_grpo_reward(args: Any, sample: Any) -> float:
    """Reward function for ``--custom-rm-path``.

    Since the multi-dimensional reward is computed during generation
    (inside ``agentic_grpo_generate``), this function acts as a pass-through:
    if the sample already has a reward set, it returns it as-is.

    If the reward is not set (fallback path), this attempts to compute it
    from the sample's trajectory metadata. This is a secondary path used
    when the custom generate function is NOT the one setting rewards.

    Args:
        args: Slime training args.
        sample: A ``Sample`` object that may already have ``.reward`` set.

    Returns:
        Scalar reward in [0, 1].
    """
    # If reward already set during generation, pass through
    if sample.reward is not None:
        try:
            return float(sample.reward)
        except (TypeError, ValueError):
            logger.warning("sample.reward is not a float: %r", sample.reward)
            return 0.0

    # Fallback: if sample has trajectory metadata, compute reward from it
    metadata = sample.metadata or {}
    trajectory = metadata.get("trajectory")
    task_type = metadata.get("benchmark", "")
    task_desc = sample.prompt if isinstance(sample.prompt, str) else ""

    if trajectory and task_type:
        logger.info("Computing reward post-hoc for %s", task_type)
        breakdown = await compute_multi_dimensional_reward(
            args, trajectory, task_desc, task_type,
        )
        return breakdown.total

    logger.warning(
        "No reward and no trajectory metadata on sample %s; returning 0",
        getattr(sample, "index", "?"),
    )
    return 0.0
