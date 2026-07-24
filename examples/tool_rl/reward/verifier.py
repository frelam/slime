"""Rule-based verifier for Qwen3-4B tool-use RL — Dim 2 + Dim 3.

Parses Qwen's XML tool call format:

.. code-block:: xml

    <tool_call>
    <function=function_name>
    <parameter=param_name>
    value
    </parameter>
    </function>
    </tool_call>

Dimensions
----------
- **Dim 2 (weight 0.20)**: Format compliance — rule verifier
  Scoring:
    1. All tool_calls after reasoning content → +0.6
    2. Each tool_call preceded by <think> → +0.4 × 1/N
    3. No tools used → 1.0

- **Dim 3 (weight 0.20)**: Tool call format correctness — rule verifier
  Scoring (N = total tool calls):
    1. Tool name correct + no undeclared tools → +1/N × 0.5
    2. Param name correct + no undeclared params → +1/N × 0.3
    3. Param type correct → +1/N × 0.2
    4. No tools used → 1.0
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# Regex patterns — Qwen XML format
# ============================================================================

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# Qwen XML tool call: <tool_call>...<function=NAME>...</function>...</tool_call>
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE,
)
_FUNCTION_NAME_RE = re.compile(r"<function=(\w[\w.]*)>")
_PARAM_RE = re.compile(
    r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL,
)

# Fallback: JSON format tool calls
_TOOL_CALL_JSON_RE = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}\s*\}',
    re.DOTALL,
)


# ============================================================================
# Tool call parsing — Qwen XML format
# ============================================================================


def parse_qwen_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse Qwen XML tool calls from text.

    Returns:
        List of ``{"name": str, "arguments": dict}``.
    """
    calls: list[dict[str, Any]] = []

    for tc_match in _TOOL_CALL_BLOCK_RE.finditer(text):
        block = tc_match.group(1)
        func_match = _FUNCTION_NAME_RE.search(block)
        if not func_match:
            continue
        func_name = func_match.group(1)

        args: dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(block):
            pname = pm.group(1)
            pval = pm.group(2).strip()
            try:
                pval = json.loads(pval)
            except (json.JSONDecodeError, TypeError):
                pass
            args[pname] = pval

        calls.append({"name": func_name, "arguments": args})

    # Fallback: JSON format
    if not calls:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            try:
                obj = json.loads(m.group(0))
                if "name" in obj and obj not in calls:
                    calls.append(obj)
            except json.JSONDecodeError:
                pass

    return calls


# ============================================================================
# Dim 2 — Format Compliance (weight 0.20)
# ============================================================================


def check_format_compliance(
    trajectory: list[dict[str, Any]],
    *,
    available_tools: list[dict[str, Any]] | None = None,
) -> float:
    """Check <think>...<tool_call> format compliance.

    Scoring:
      1. All tool_calls after reasoning content → +0.6
      2. Each tool_call preceded by <think> → +0.4 × count/N
      3. No tools → 0.0 if tools are defined, 1.0 if no tools at all

    Args:
        trajectory: Normalized trajectory.
        available_tools: Tool definitions. If non-empty and no calls, score 0.

    Returns:
        Score in [0.0, 1.0].
    """
    all_text = _get_agent_text(trajectory)
    n_calls = len(_TOOL_CALL_BLOCK_RE.findall(all_text))
    n_calls += len(_TOOL_CALL_JSON_RE.findall(all_text))

    if n_calls == 0:
        if available_tools:
            logger.debug("[dim2] No tool calls but tools available → 0.0")
            return 0.0
        logger.debug("[dim2] No tool calls, no tools defined → 1.0")
        return 1.0

    score = 0.0

    # Rule 1: All tool calls after </think> → +0.6
    if _all_calls_after_think(all_text):
        score += 0.6
        logger.debug("[dim2] All calls after think → +0.6")

    # Rule 2: Each tool call preceded by <think> → +0.4 × count/N
    preceded = _count_preceded_by_think(all_text, n_calls)
    if preceded > 0:
        bonus = 0.4 * preceded / n_calls
        score += bonus
        logger.debug("[dim2] %d/%d calls preceded by think → +%.3f", preceded, n_calls, bonus)

    return max(0.0, min(1.0, score))


def _get_agent_text(trajectory: list[dict[str, Any]]) -> str:
    parts = [r.get("text", "") for r in trajectory if r.get("type") != "observation"]
    return "\n".join(parts)


def _all_calls_after_think(text: str) -> bool:
    """Check all <tool_call> blocks are after the last </think>."""
    last_end = 0
    for m in re.finditer(r"</think>", text, re.IGNORECASE):
        last_end = m.end()
    if last_end == 0:
        return False
    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        if m.start() < last_end:
            return False
    for m in _TOOL_CALL_JSON_RE.finditer(text):
        if m.start() < last_end:
            return False
    return True


def _count_preceded_by_think(text: str, total: int) -> int:
    """Count how many tool calls have </think> before them."""
    think_ends = [m.end() for m in re.finditer(r"</think>", text, re.IGNORECASE)]
    if not think_ends:
        return 0

    call_starts = []
    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        call_starts.append(m.start())
    for m in _TOOL_CALL_JSON_RE.finditer(text):
        call_starts.append(m.start())
    call_starts.sort()

    count = 0
    ti = 0
    for cs in call_starts:
        while ti < len(think_ends) - 1 and think_ends[ti + 1] < cs:
            ti += 1
        if think_ends[ti] < cs:
            count += 1
    return count


# ============================================================================
# Dim 3 — Tool Call Format Correctness (weight 0.20)
# ============================================================================


def check_tool_call_format(
    trajectory: list[dict[str, Any]],
    available_tools: list[dict[str, Any]] | None = None,
) -> float:
    """Check tool call name, param name, param type correctness.

    Scoring (N = total tool calls):
      1. Name correct + not undeclared → +1/N × 0.5
      2. Param name correct + not undeclared → +1/N × 0.3
      3. Param type correct → +1/N × 0.2
      4. No tools → 0.0 if tools are defined, 1.0 if no tools at all

    Args:
        trajectory: Normalized trajectory.
        available_tools: Tool definitions. If non-empty and no calls, score 0.

    Returns:
        Score in [0.0, 1.0].
    """
    all_text = _get_agent_text(trajectory)
    parsed = parse_qwen_tool_calls(all_text)

    if not parsed:
        if available_tools:
            logger.debug("[dim3] No tool calls but tools available → 0.0")
            return 0.0
        logger.debug("[dim3] No tool calls, no tools defined → 1.0")
        return 1.0

    n = len(parsed)
    available = available_tools or []

    tool_names, tool_params = _build_tool_index(available)

    name_acc = 0.0
    pname_acc = 0.0
    ptype_acc = 0.0

    for call in parsed:
        cname = call.get("name", "")
        cargs = call.get("arguments", {}) or {}

        # 1. Name correctness
        if cname and cname in tool_names:
            name_acc += 1.0
        elif cname:
            logger.debug("[dim3] Unknown tool: %r", cname)

        # 2. Param name + 3. Param type
        if cname in tool_params:
            declared = tool_params[cname]
            declared_names = set(declared.keys())

            if declared_names and cargs:
                # Param name: fraction of declared params present + no extra
                matched = sum(1 for k in cargs if k in declared_names)
                extra = sum(1 for k in cargs if k not in declared_names)
                pname_acc += matched / max(len(declared_names), len(cargs))
                if extra:
                    logger.debug("[dim3] Extra params for %r: %s",
                                 cname, [k for k in cargs if k not in declared_names])

                # Param type
                ptype_acc += _check_types(cargs, declared)
            elif not declared_names:
                pname_acc += 1.0
                ptype_acc += 1.0
            elif not cargs:
                # No args but params declared → half credit for name
                pname_acc += 0.5

    # Normalize by N
    score = (
        (name_acc / n) * 0.5
        + (pname_acc / n) * 0.3
        + (ptype_acc / n) * 0.2
    )

    logger.debug("[dim3] N=%d name=%.3f pname=%.3f ptype=%.3f → %.3f",
                 n, name_acc / n * 0.5, pname_acc / n * 0.3, ptype_acc / n * 0.2, score)

    return max(0.0, min(1.0, score))


_TYPE_MAP = {
    "string": str, "str": str,
    "integer": int, "int": int,
    "number": (int, float), "float": float,
    "boolean": bool, "bool": bool,
    "array": list, "list": list,
    "object": dict, "dict": dict,
}


def _check_types(
    args: dict[str, Any],
    declared: dict[str, dict],
) -> float:
    """Fraction of args with correct types."""
    correct = 0
    for k, v in args.items():
        if k not in declared:
            continue
        dtype = declared[k].get("type", "")
        expected = _TYPE_MAP.get(dtype.lower()) if dtype else None
        if expected is None or isinstance(v, expected):
            correct += 1
        else:
            logger.debug("[dim3] Type mismatch: %s=%s (expected %s, got %s)",
                         k, v, dtype, type(v).__name__)
    return correct / max(len(args), 1)


# ============================================================================
# Tool call correctness — per-call verdict (for token-level loss masking)
# ============================================================================


def _build_tool_index(
    available_tools: list[dict[str, Any]] | None,
) -> tuple[set[str], dict[str, dict[str, dict]]]:
    """Build tool name set and param index from available_tools.

    Args:
        available_tools: Tool definitions from the dataset metadata.

    Returns:
        Tuple of ``(tool_names, tool_params)`` where ``tool_params`` maps
        tool name → param name → param info dict.
    """
    tool_names: set[str] = set()
    tool_params: dict[str, dict[str, dict]] = {}
    for tool in (available_tools or []):
        name = tool.get("name", "")
        if not name:
            continue
        tool_names.add(name)
        params = tool.get("parameters", {})
        props = params.get("properties", params) if isinstance(params, dict) else {}
        if isinstance(props, dict):
            if props and isinstance(next(iter(props.values()), None), dict):
                tool_params[name] = props
    return tool_names, tool_params


def _is_tool_call_correct(
    call: dict[str, Any],
    tool_names: set[str],
    tool_params: dict[str, dict[str, dict]],
) -> bool:
    """Check whether a single parsed tool call is fully correct.

    A tool call is correct when ALL of:
    1. Function name exists in ``tool_names``
    2. All parameter names are declared for that function
    3. No extra/undeclared parameter names
    4. All parameter values match declared types

    If ``tool_names`` is empty (no tool definitions available), returns
    ``True`` (cannot verify — assume correct).
    """
    cname = call.get("name", "")
    cargs = call.get("arguments", {}) or {}

    if not tool_names:
        return True  # No tool definitions to check against

    # 1. Name correctness
    if not cname or cname not in tool_names:
        return False

    # 2-4. Parameter correctness
    if cname not in tool_params:
        # Tool has no declared params — any args are wrong
        return not cargs

    declared = tool_params[cname]
    declared_names = set(declared.keys())

    if not declared_names:
        return not cargs  # No declared params, no args expected

    if not cargs:
        # Tool expects params but none given
        return False

    # Check for extra/undeclared params
    for k in cargs:
        if k not in declared_names:
            return False

    # Check param types
    for k, v in cargs.items():
        if k not in declared:
            continue
        dtype = declared[k].get("type", "")
        expected = _TYPE_MAP.get(dtype.lower()) if dtype else None
        if expected is not None and not isinstance(v, expected):
            return False

    return True


def get_incorrect_tool_call_spans(
    text: str,
    available_tools: list[dict[str, Any]] | None = None,
) -> list[tuple[int, int]]:
    """Return ``(start_char, end_char)`` spans of incorrect tool call blocks.

    Parses Qwen XML ``<tool_call>...</tool_call>`` blocks from *text* and checks
    each one against *available_tools*.  Blocks with wrong function name, wrong
    parameter names, undeclared parameters, or wrong parameter types are
    collected.

    Args:
        text: Raw assistant response containing zero or more tool call blocks.
        available_tools: Tool definitions.  If empty or ``None``, all tool calls
            are treated as correct.

    Returns:
        List of ``(start_char, end_char)`` tuples for incorrect tool call
        blocks.  Empty if all tool calls are correct or none exist.
    """
    tool_names, tool_params = _build_tool_index(available_tools)

    incorrect_spans: list[tuple[int, int]] = []

    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        block_text = match.group(1)  # content inside <tool_call>...</tool_call>
        func_match = _FUNCTION_NAME_RE.search(block_text)

        call: dict[str, Any] = {"name": "", "arguments": {}}
        if func_match:
            call["name"] = func_match.group(1)

        for pm in _PARAM_RE.finditer(block_text):
            pname = pm.group(1)
            pval = pm.group(2).strip()
            try:
                pval = json.loads(pval)
            except (json.JSONDecodeError, TypeError):
                pass
            call["arguments"][pname] = pval

        if not _is_tool_call_correct(call, tool_names, tool_params):
            incorrect_spans.append((match.start(), match.end()))
            logger.debug(
                "[mask] Incorrect tool call: name=%r span=(%d, %d)",
                call.get("name"), match.start(), match.end(),
            )

    return incorrect_spans


# ============================================================================
# Combined verifier (Dim 2 + Dim 3 only)
# ============================================================================


def compute_verifier_scores(
    trajectory: list[dict[str, Any]],
    *,
    available_tools: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Compute Dim 2 + Dim 3 verifier scores.

    Args:
        trajectory: Normalized trajectory.
        available_tools: Tool definitions for Dim 3 format check.

    Returns:
        ``{"format_compliance": float, "tool_call_format": float}``.
    """
    return {
        "format_compliance": check_format_compliance(trajectory, available_tools=available_tools),
        "tool_call_format": check_tool_call_format(trajectory, available_tools),
    }
