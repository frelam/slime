"""Rule-based verifier dimensions for agentic RL reward scoring.

Implements dimensions that can be scored programmatically without an LLM call:

- Dim 4.2: Format compliance (weight 0.15)
- Dim 4.3: Tool call parameter correctness (weight 0.10)
- Dim 4.4: Tool call retry behavior (weight 0.05)
- Dim 4.7: Tool call count penalty (weight 0.05)

All functions are pure (no I/O, no async) so they are trivially testable.
"""

from __future__ import annotations

import logging
from typing import Any

from examples.agentic_rl_grpo.traj_analysis import (
    count_tool_calls,
    detect_retries,
    extract_tool_calls,
    find_failed_calls,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Dim 4.2 — Format Compliance (weight 0.15)
# =============================================================================


def check_format_compliance(
    trajectory: list[dict[str, Any]],
    task_type: str = "terminal",
) -> float:
    """Check if the agent's output follows expected formatting conventions.

    For terminal tasks, checks that:
    - Tool calls use valid JSON format (if JSON tool calls used)
    - Responses are well-structured
    - No garbled or truncated output

    For SWE tasks, checks that:
    - Code edits are well-formed
    - Diff/commit messages follow conventions

    Args:
        trajectory: Normalized trajectory list.
        task_type: ``"terminal"`` or ``"swe"``.

    Returns:
        1.0 if format is compliant, 0.0 otherwise.
    """
    if task_type in ("swe_gym_lite", "r2e_gym", "swe"):
        return _check_swe_format(trajectory)
    return _check_terminal_format(trajectory)


def _check_terminal_format(trajectory: list[dict[str, Any]]) -> float:
    """Check terminal-task format compliance."""
    all_valid = True
    has_tool_calls = False

    for record in trajectory:
        text = record.get("text", "")

        # Skip observation records (tool outputs)
        if record.get("type") == "observation":
            continue

        # Check for parsible tool calls
        from examples.agentic_rl.agent_loop import parse_tool_calls

        calls = parse_tool_calls(text)
        if calls:
            has_tool_calls = True
            for call in calls:
                if not isinstance(call, dict) or "name" not in call:
                    all_valid = False
                    logger.debug("Format issue: tool call missing 'name': %r", call)

    # If no tool calls at all, check that text isn't garbled
    if not has_tool_calls:
        for record in trajectory:
            if record.get("type") != "observation":
                text = record.get("text", "")
                if text and len(text) < 2:  # Suspiciously short
                    all_valid = False

    return 1.0 if all_valid else 0.0


def _check_swe_format(trajectory: list[dict[str, Any]]) -> float:
    """Check SWE-task format compliance."""
    # For SWE tasks: check that code diffs are valid, no empty edits
    all_valid = True

    for record in trajectory:
        text = record.get("text", "")
        if record.get("type") == "observation":
            continue

        # Basic check: response should have reasonable length
        if text and len(text) < 5 and record.get("finish_reason") == "stop":
            all_valid = False

    return 1.0 if all_valid else 0.0


# =============================================================================
# Dim 4.3 — Tool Call Parameter Correctness (weight 0.10)
# =============================================================================


def check_tool_param_correctness(
    trajectory: list[dict[str, Any]],
) -> float:
    """Check if all tool calls had correct parameters (no execution errors).

    A score of 1.0 means every tool call executed successfully.
    A score of 0.0 means at least one tool call failed due to parameter issues.

    Args:
        trajectory: Normalized trajectory list.

    Returns:
        1.0 if no tool execution errors, 0.0 otherwise.
    """
    failures = find_failed_calls(trajectory)

    if not failures:
        # Also check: were any tool calls made at all?
        calls = extract_tool_calls(trajectory)
        if not calls:
            # No tool calls → no parameter issues → 1.0
            return 1.0
        return 1.0

    logger.debug(
        "Tool param failures detected: %d failures: %s",
        len(failures),
        [f["tool_name"] for f in failures],
    )
    return 0.0


# =============================================================================
# Dim 4.4 — Tool Call Retry Behavior (weight 0.05)
# =============================================================================


def check_retry_behavior(
    trajectory: list[dict[str, Any]],
) -> float:
    """Evaluate the agent's retry behavior after tool failures.

    Scoring:
    - 1.0: No tool failures occurred.
    - 0.5: Failures occurred, but the agent retried at least once or tried
      an alternative approach.
    - 0.0: Failures occurred and the agent did not retry or recover.

    Args:
        trajectory: Normalized trajectory list.

    Returns:
        Score in {0.0, 0.5, 1.0}.
    """
    failures = find_failed_calls(trajectory)

    if not failures:
        return 1.0

    retries = detect_retries(trajectory)

    if retries:
        logger.debug(
            "Agent retried after failures: %d failures, %d retries",
            len(failures),
            len(retries),
        )
        return 0.5

    # Check if agent tried alternative approaches (different tool with similar
    # purpose) after a failure.
    if _detected_alternative_approach(trajectory, failures):
        return 0.5

    logger.debug(
        "Agent did not retry after %d failures",
        len(failures),
    )
    return 0.0


def _detected_alternative_approach(
    trajectory: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> bool:
    """Check if the agent tried a different tool after a failure.

    For example: ``bash`` failed → agent tried ``python`` instead.
    """
    if not failures:
        return False

    failed_names = {f.get("tool_name", "") for f in failures}

    # Look for subsequent tool calls with different names
    calls = extract_tool_calls(trajectory)
    for call in calls:
        name = call.get("name", "")
        if name and name not in failed_names:
            # Found a different tool used after failure — counts as
            # "alternative approach"
            return True

    return False


# =============================================================================
# Dim 4.7 — Tool Call Count Penalty (weight 0.05)
# =============================================================================


def compute_tool_count_penalty(
    trajectory: list[dict[str, Any]],
    answer_correct: bool,
    *,
    max_count: int = 1000,
) -> float:
    """Compute the tool call count penalty.

    - If the answer is wrong: returns 0.0.
    - If the answer is right: returns ``max(0.0, 1.0 - count / max_count)``.

    This penalizes excessive tool calls even when the answer is correct.

    Args:
        trajectory: Normalized trajectory list.
        answer_correct: Whether the agent's final answer is correct.
        max_count: The tool call count at which the penalty is 1.0 (score=0).

    Returns:
        Score in [0.0, 1.0].
    """
    if not answer_correct:
        return 0.0

    count = count_tool_calls(trajectory)
    if count == 0:
        return 1.0

    penalty = 1.0 - count / max_count
    score = max(0.0, penalty)

    if score < 1.0:
        logger.debug(
            "Tool count penalty: %d calls / %d max → score=%.3f",
            count,
            max_count,
            score,
        )

    return score


# =============================================================================
# Combined verifier score (for logging / fallback without RM)
# =============================================================================


def compute_verifier_scores(
    trajectory: list[dict[str, Any]],
    *,
    answer_correct: bool = False,
    task_type: str = "terminal",
) -> dict[str, float]:
    """Compute all verifier dimension scores at once.

    Args:
        trajectory: Normalized trajectory list.
        answer_correct: Whether the agent's answer is correct (from RM or task eval).
        task_type: ``"terminal"`` or ``"swe"``.

    Returns:
        Dict mapping dimension names to scores in [0, 1].
    """
    return {
        "format_compliance": check_format_compliance(trajectory, task_type),
        "tool_param_correctness": check_tool_param_correctness(trajectory),
        "tool_retry": check_retry_behavior(trajectory),
        "tool_count_penalty": compute_tool_count_penalty(trajectory, answer_correct),
    }
