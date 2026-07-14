"""Trajectory analysis utilities for agentic RL reward scoring.

Provides pure functions to extract structured data from agent trajectories
for the multi-dimensional reward verifier. Works with both:

1. **SGLang agent loop** trajectories (from ``agent_loop.py``):
   List of turn dicts with ``turn``, ``text``, ``type``, ``tool_call`` keys.

2. **Claude Code / Hermes** trajectories (from adapter):
   List of message dicts with at least ``text`` and optionally ``type``,
   ``tool_call`` keys. Constructed by formatting the adapter's session history.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool call extraction
# ---------------------------------------------------------------------------


def extract_tool_calls(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract all tool call dicts from the trajectory.

    For SGLang agent-loop trajectories, tool calls are stored in observation
    records with a ``tool_call`` key. For adapter trajectories, tool calls
    are parsed from the ``text`` field of assistant turns.

    Args:
        trajectory: List of turn/observation dicts.

    Returns:
        List of tool call dicts with at least ``name`` and ``arguments`` keys.
    """
    calls: list[dict[str, Any]] = []

    for record in trajectory:
        # Direct tool_call entry (agent_loop observation format)
        if "tool_call" in record:
            tc = record["tool_call"]
            if isinstance(tc, dict) and "name" in tc:
                calls.append(tc)
            continue

        # Parse tool calls from text
        text = record.get("text", "")
        if text:
            from examples.agentic_rl.agent_loop import parse_tool_calls

            parsed = parse_tool_calls(text)
            calls.extend(parsed)

    return calls


def count_tool_calls(trajectory: list[dict[str, Any]]) -> int:
    """Count total tool calls in the trajectory.

    Args:
        trajectory: List of turn/observation dicts.

    Returns:
        Number of tool calls made by the agent.
    """
    return len(extract_tool_calls(trajectory))


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------


def find_failed_calls(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find tool calls that resulted in execution errors.

    Detects failures by examining observation records for:
    - Non-zero exit codes
    - stderr output
    - Error keywords in observation text
    - Explicit failure markers

    Args:
        trajectory: List of turn/observation dicts.

    Returns:
        List of dicts with keys: ``tool_name``, ``arguments``, ``error``,
        ``observation``.
    """
    failures: list[dict[str, Any]] = []

    for i, record in enumerate(trajectory):
        if record.get("type") != "observation":
            continue

        text = record.get("text", "")
        tool_call = record.get("tool_call", {})

        if _is_failure(text):
            # Find the preceding tool call
            tc_name = tool_call.get("name", "unknown") if isinstance(tool_call, dict) else "unknown"
            tc_args = tool_call.get("arguments", {}) if isinstance(tool_call, dict) else {}

            failures.append({
                "tool_name": tc_name,
                "arguments": tc_args,
                "error": _extract_error(text),
                "observation": text,
                "record_index": i,
            })

    return failures


def _is_failure(text: str) -> bool:
    """Heuristic: does this observation text indicate a tool execution failure?"""
    if not text:
        return False

    # Explicit error markers
    error_markers = [
        "[Error]",
        "[ERROR]",
        "Error:",
        "error:",
        "Traceback (most recent call last)",
        "SyntaxError",
        "NameError",
        "TypeError",
        "ValueError",
        "KeyError",
        "IndexError",
        "FileNotFoundError",
        "PermissionError",
        "ModuleNotFoundError",
        "ImportError",
        "command not found",
        "No such file or directory",
        "Permission denied",
        "exit status 1",
        "exit status 2",
        "FAILED",
        "Aborted",
    ]
    for marker in error_markers:
        if marker in text:
            return True

    return False


def _extract_error(text: str) -> str:
    """Extract a concise error message from observation text."""
    lines = text.strip().split("\n")
    error_lines = []
    in_traceback = False
    for line in lines:
        stripped = line.strip()
        if "Traceback" in stripped:
            in_traceback = True
            error_lines.append(stripped)
            continue
        if in_traceback:
            if stripped.startswith(("File ", "  File ", "    ")) or stripped == "":
                error_lines.append(stripped)
                continue
            # Last line of traceback is the actual error
            if any(
                err in stripped
                for err in ("Error:", "Error ", "Exception:", "KeyboardInterrupt")
            ):
                error_lines.append(stripped)
            in_traceback = False
            continue
        if any(
            marker in stripped
            for marker in ("[Error]", "Error:", "error:", "FAILED")
        ):
            error_lines.append(stripped)

    if error_lines:
        return "\n".join(error_lines[-5:])  # Last 5 error lines
    return text[:500]


# ---------------------------------------------------------------------------
# Retry detection
# ---------------------------------------------------------------------------


def detect_retries(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect retry patterns after tool call failures.

    A "retry" is when the agent calls the same tool (by name) after a
    previous call to that tool produced an error.

    Args:
        trajectory: List of turn/observation dicts.

    Returns:
        List of dicts with keys: ``tool_name``, ``failed_at`` (record index),
        ``retry_at`` (record index), ``retry_successful`` (bool).
    """
    retries: list[dict[str, Any]] = []
    failed_tools: dict[str, int] = {}  # tool_name → record_index of failure

    for i, record in enumerate(trajectory):
        if record.get("type") != "observation":
            continue

        text = record.get("text", "")
        tool_call = record.get("tool_call", {})

        if not isinstance(tool_call, dict):
            continue

        tool_name = tool_call.get("name", "")

        if _is_failure(text):
            # Record the failure
            if tool_name and tool_name not in failed_tools:
                failed_tools[tool_name] = i
        else:
            # Successful execution — check if it was a retry
            if tool_name and tool_name in failed_tools:
                retries.append({
                    "tool_name": tool_name,
                    "failed_at": failed_tools[tool_name],
                    "retry_at": i,
                    "retry_successful": True,
                })
                del failed_tools[tool_name]

    return retries


# ---------------------------------------------------------------------------
# Formatting for RM
# ---------------------------------------------------------------------------


def format_for_rm(
    trajectory: list[dict[str, Any]],
    task_description: str,
) -> str:
    """Format a trajectory as a readable string for the reward model.

    Args:
        trajectory: List of turn/observation dicts.
        task_description: The original task description / prompt.

    Returns:
        Formatted markdown string suitable as RM user prompt.
    """
    parts: list[str] = []
    parts.append("## Task Description\n")
    parts.append(task_description)
    parts.append("\n## Agent Trajectory\n")

    for i, record in enumerate(trajectory):
        turn = record.get("turn", i)
        rtype = record.get("type", "turn")
        text = record.get("text", "")

        if rtype == "observation":
            tool_call = record.get("tool_call", {})
            tc_name = tool_call.get("name", "unknown") if isinstance(tool_call, dict) else "unknown"
            parts.append(f"\n### Turn {turn} — Tool Result: `{tc_name}`\n")
            parts.append(f"```\n{text[:2000]}\n```\n")
        else:
            finish = record.get("finish_reason", "")
            parts.append(f"\n### Turn {turn} — Agent Response ({finish})\n")
            parts.append(f"{text[:3000]}\n")

    return "\n".join(parts)


def extract_final_answer(trajectory: list[dict[str, Any]]) -> str | None:
    """Extract the agent's final answer from the trajectory.

    Looks for:
    1. The last text response from the agent (non-observation turn)
    2. Content after ``[FINISH]`` markers in observation records

    Args:
        trajectory: List of turn/observation dicts.

    Returns:
        The final answer string, or None if not found.
    """
    # Look for [FINISH] marker first (agent_loop convention)
    for record in reversed(trajectory):
        text = record.get("text", "")
        if "[FINISH]" in text:
            idx = text.index("[FINISH]")
            return text[idx + len("[FINISH]"):].strip()

    # Fallback: return the last agent response text
    for record in reversed(trajectory):
        if record.get("type") != "observation":
            text = record.get("text", "")
            if text.strip():
                return text.strip()

    return None


# ---------------------------------------------------------------------------
# Utility: normalize trajectory from different sources
# ---------------------------------------------------------------------------


def normalize_trajectory(
    raw: list[dict[str, Any]],
    source: str = "agent_loop",
) -> list[dict[str, Any]]:
    """Normalize a trajectory from different sources into the standard format.

    Standard format keys per record:
        - ``turn`` (int): turn index
        - ``text`` (str): the text content
        - ``type`` (str): ``"turn"`` for agent responses, ``"observation"``
          for tool execution results
        - ``tool_call`` (dict | None): parsed tool call if applicable
        - ``finish_reason`` (str | None): ``"stop"``, ``"length"``, ``"abort"``

    Args:
        raw: Raw trajectory from agent_loop, adapter, or custom source.
        source: Source format name (``"agent_loop"``, ``"adapter"``, ``"auto"``).

    Returns:
        Normalized list of dicts.
    """
    if source == "agent_loop":
        # Already in standard format
        return raw

    if source == "adapter":
        # Claude Code / Hermes adapter format: list of message dicts with
        # role + content
        normalized: list[dict[str, Any]] = []
        for i, msg in enumerate(raw):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Flatten Anthropic content blocks
                from slime.agent.adapters.common import flatten_content
                content = flatten_content(content)

            normalized.append({
                "turn": i,
                "text": str(content),
                "type": "observation" if role == "tool" else "turn",
                "tool_call": msg.get("tool_call"),
                "finish_reason": msg.get("finish_reason", ""),
            })
        return normalized

    if source == "auto":
        # Auto-detect: if first record has "turn" key, assume agent_loop
        if raw and "turn" in raw[0]:
            return normalize_trajectory(raw, source="agent_loop")
        return normalize_trajectory(raw, source="adapter")

    logger.warning("Unknown trajectory source %r, returning raw", source)
    return raw
