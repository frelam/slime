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
    1. All tool_calls after a single, non-empty <think> → +0.6
    2. Each tool_call preceded by a non-empty <think> → +0.4 × 1/N
    3. No calls → 1.0 when no tools are expected (label says no tools
       needed, or no tools are defined), otherwise 0.0

  The <think> block must appear exactly once with non-empty content for
  full credit — an empty or repeated <think> is not valid reasoning.
  Empty <tool_call> blocks (no <function=...> header) are not counted as
  tool calls.

- **Dim 3 (weight 0.20)**: Tool call format correctness — rule verifier
  Scoring (N = total tool calls):
    1. Tool name correct + no undeclared tools → +1/N × 0.5
    2. Param name correct + no undeclared params → +1/N × 0.3
    3. Param type correct → +1/N × 0.2
    4. No calls → 1.0 when no tools are expected (label says no tools
       needed, or no tools are defined), otherwise 0.0
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

# A raw ``<think>`` opener — used to detect an unclosed think block.
_OPENING_RE = re.compile(r"<think>", re.IGNORECASE)


def _check_strict_format(text: str) -> bool:
    """Validate the top-level layout of a response's think block.

    The Qwen3 chat template makes the assistant emit ``<think>...</think>``
    followed by a response and/or ``<tool_call>``.  ``<think>``/``</think>``
    are not special tokens (``special: False``), so they survive SGLang
    decoding and appear verbatim in the rollout text.

    Rules:
      1. ``<think>`` tags must be paired — an unclosed ``<think>`` opener
         with no ``</think>`` is invalid.
      2. More than one complete ``<think>...</think>`` block is invalid
         (ambiguous layout).
      3. Something (response text / tool_call) must follow the think block —
         emitting reasoning and halting without an answer is the
         "think-then-stop" collapse.
      4. Otherwise valid: no think block at all, or exactly one complete
         ``<think>...</think>`` block.
    """
    matches = list(_THINK_RE.finditer(text))
    # Every opener must have a matching closer — an unclosed <think> is a
    # think-then-stop collapse.
    if len(_OPENING_RE.findall(text)) != len(matches):
        return False
    if len(matches) > 1:
        return False
    if len(matches) == 1:
        # A response or tool_call must follow the think block.
        return text[matches[0].end():].strip() != ""
    return True


# Qwen XML tool call: <tool_call>...<function=NAME>...</function>...</tool_call>
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE,
)
_FUNCTION_NAME_RE = re.compile(r"<function=(\w[\w.]*)>")
_PARAM_RE = re.compile(
    r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL,
)
# Inline JSON style: <tool_call>\n"name": NAME, "arguments": {...}\n</tool_call>
_INLINE_CALL_RE = re.compile(
    r'^\s*"name"\s*:\s*"([\w.]*)",\s*"arguments"\s*:\s*(\{.*\})\s*$',
    re.DOTALL,
)

# Fallback: JSON format tool calls — bracket-matching is more robust than
# regex because it handles nested objects/arrays inside arguments.
def _extract_json_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract ``{"name": …, "arguments": {…}}`` objects from text.

    Uses bracket-depth tracking so nested parameter values (objects,
    arrays of objects, etc.) are handled correctly.
    """
    results: list[dict[str, Any]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except (json.JSONDecodeError, TypeError):
                    start = -1
                    continue
                if isinstance(obj, dict) and "name" in obj:
                    results.append(obj)
                start = -1
    return results


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
        inline_match = _INLINE_CALL_RE.search(block)
        if not func_match and not inline_match:
            continue
        func_name = func_match.group(1) if func_match else inline_match.group(1)

        args: dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(block):
            pname = pm.group(1)
            pval = pm.group(2).strip()
            try:
                pval = json.loads(pval)
            except (json.JSONDecodeError, TypeError):
                pass
            args[pname] = pval

        if inline_match and not args:
            try:
                args = json.loads(inline_match.group(2))
            except (json.JSONDecodeError, TypeError):
                args = {}

        calls.append({"name": func_name, "arguments": args})

    # Fallback: JSON format
    if not calls:
        for obj in _extract_json_tool_calls(text):
            if obj not in calls:
                calls.append(obj)

    return calls


# ============================================================================
# Dim 2 — Format Compliance (weight 0.20)
# ============================================================================


def check_format_compliance(
    trajectory: list[dict[str, Any]],
    *,
    available_tools: list[dict[str, Any]] | None = None,
    expects_no_tools: bool = False,
) -> float:
    """Check <think>...<tool_call> format compliance.

    Scoring:
      1. All tool_calls after a single, non-empty <think> block → +0.6
      2. Each tool_call preceded by a non-empty <think> → +0.4 × count/N
      3. No calls → 1.0 when no tools are expected (label says no tools
         needed, or no tools are defined), otherwise 0.0 if tools are defined

    Full credit requires the think block to appear exactly once with
    non-empty content — an empty or repeated ``<think>`` is not valid
    reasoning and forfeits the 0.6 portion.  Empty ``<tool_call>`` blocks
    (those without a ``<function=...>`` header) are not counted as tool
    calls, matching the other dimensions' parsers.

    Args:
        trajectory: Normalized trajectory.
        available_tools: Tool definitions. If non-empty and no calls, score 0.
        expects_no_tools: Label says no tools are needed for this task.

    Returns:
        Score in [0.0, 1.0].
    """
    all_text = _get_agent_text(trajectory)

    # Strict gate: the response must be a well-formed thinking→response block
    # with something (response text / tool_call) after it. Emitting an
    # unclosed ``<think>`` (think-then-stop collapse), more than one think
    # block, or a think block with nothing after it is a format violation —
    # even when it makes no tool calls.
    if not _check_strict_format(all_text):
        logger.debug("[dim2] Strict thinking→response format broken → 0.0")
        return 0.0

    n_calls = len(_xml_tool_call_spans(all_text))
    n_calls += len(_find_json_tool_call_spans(all_text))

    if n_calls == 0:
        if expects_no_tools or not available_tools:
            reason = "label says no tools needed" if expects_no_tools else "no tools defined"
            logger.debug("[dim2] No tool calls and %s → 1.0", reason)
            return 1.0
        logger.debug("[dim2] No tool calls but tools available → 0.0")
        return 0.0

    score = 0.0

    # Rule 1: All tool calls after a single, non-empty think → +0.6
    if _all_calls_after_think(all_text):
        score += 0.6
        logger.debug("[dim2] All calls after a single non-empty think → +0.6")
    else:
        logger.debug("[dim2] No valid single think (empty/missing/repeated) → no +0.6")

    # Rule 2: Each tool call preceded by a non-empty <think> → +0.4 × count/N
    preceded = _count_preceded_by_think(all_text, n_calls)
    if preceded > 0:
        bonus = 0.4 * preceded / n_calls
        score += bonus
        logger.debug("[dim2] %d/%d calls preceded by think → +%.3f", preceded, n_calls, bonus)

    return max(0.0, min(1.0, score))


def _get_agent_text(trajectory: list[dict[str, Any]]) -> str:
    parts = [r.get("text", "") for r in trajectory if r.get("type") != "observation"]
    return "\n".join(parts)


def _find_json_tool_call_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans of ``{"name": …}`` objects in *text*.

    Uses bracket-depth tracking — robust against nested arguments.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append((start, i + 1))
                start = -1
    return spans


def _xml_tool_call_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans of *valid* ``<tool_call>`` blocks.

    A block is valid only when it declares a function via ``<function=...>``.
    Empty blocks (``<tool_call></tool_call>``) or blocks without a function
    header parse to no call in :func:`parse_qwen_tool_calls`, so they are
    excluded here to keep the format-score call count consistent with the
    other dimensions.
    """
    spans: list[tuple[int, int]] = []
    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        if _FUNCTION_NAME_RE.search(m.group(1)) or _INLINE_CALL_RE.search(m.group(1)):
            spans.append((m.start(), m.end()))
    return spans


def _all_calls_after_think(text: str) -> bool:
    """Check all tool calls are after a single, non-empty <think> block.

    Full credit requires exactly one ``<think>...</think>`` with non-empty
    content — empty or repeated think blocks are not valid reasoning.
    """
    matches = list(_THINK_RE.finditer(text))
    if len(matches) != 1 or not matches[0].group(1).strip():
        return False
    think_end = matches[0].end()
    for start, _end in _xml_tool_call_spans(text):
        if start < think_end:
            return False
    for start, _end in _find_json_tool_call_spans(text):
        if start < think_end:
            return False
    return True


def _count_preceded_by_think(text: str, total: int) -> int:
    """Count how many tool calls have a non-empty <think> before them."""
    think_ends = [
        m.end() for m in _THINK_RE.finditer(text) if m.group(1).strip() != ""
    ]
    if not think_ends:
        return 0

    call_starts = [start for start, end in _xml_tool_call_spans(text)]
    call_starts.extend(start for start, end in _find_json_tool_call_spans(text))
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
    expects_no_tools: bool = False,
) -> float:
    """Check tool call name, param name, param type correctness.

    Scoring (N = total tool calls):
      1. Name correct + not undeclared → +1/N × 0.5
      2. Param name correct + not undeclared → +1/N × 0.3
      3. Param type correct → +1/N × 0.2
      4. No calls → 1.0 when no tools are expected (label says no tools
         needed, or no tools are defined), otherwise 0.0 if tools are defined

    Args:
        trajectory: Normalized trajectory.
        available_tools: Tool definitions. If non-empty and no calls, score 0.
        expects_no_tools: Label says no tools are needed for this task.

    Returns:
        Score in [0.0, 1.0].
    """
    all_text = _get_agent_text(trajectory)

    # Same strict gate: the no-call 1.0 branch must not reward a
    # think-then-stop collapse either.
    if not _check_strict_format(all_text):
        logger.debug("[dim3] Strict thinking→response format broken → 0.0")
        return 0.0

    parsed = parse_qwen_tool_calls(all_text)

    if not parsed:
        if expects_no_tools or not available_tools:
            reason = "label says no tools needed" if expects_no_tools else "no tools defined"
            logger.debug("[dim3] No tool calls and %s → 1.0", reason)
            return 1.0
        logger.debug("[dim3] No tool calls but tools available → 0.0")
        return 0.0

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
                # Param name: fraction of provided params that are declared.
                # Missing optional schema params are not penalized — the
                # label match already covers whether a param was required.
                matched = sum(1 for k in cargs if k in declared_names)
                extra = sum(1 for k in cargs if k not in declared_names)
                pname_acc += matched / max(len(cargs), 1)
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


# ============================================================================
# Label-based tool call correctness matching (for ground-truth-labeled data)
# ============================================================================
#
# When the dataset provides ground-truth tool call labels, we match the
# model output against them with order-independent (bipartite) matching.
#
# Scoring per sample (within the tool_correctness dimension):
#   - Tool name match    → 0.5  (binary: matched or not, per label call)
#   - Parameter content  → 0.5  (fractional: how well param values match)
#
# Multiple tool calls in the output are matched to label calls by tool name
# first, then by best param similarity — order is irrelevant.


def _values_match(v1: Any, v2: Any) -> bool:
    """Check value equality with some fuzziness for strings.

    A JSON-encoded string (e.g. a label parameter stored as
    ``'["session1", ...]'``) is treated as equivalent to the decoded
    structured value (``["session1", ...]``).
    """
    v1 = _unwrap_json_string(v1)
    v2 = _unwrap_json_string(v2)

    if v1 is v2 or type(v1) == type(v2) and v1 == v2:
        return True
    if v1 is None or v2 is None:
        return False
    # bool is a subclass of int — True == 1 and False == 0 in Python,
    # but they are semantically different for tool-call parameters.
    if isinstance(v1, bool) != isinstance(v2, bool):
        return False
    if isinstance(v1, str) and isinstance(v2, str):
        return v1.strip().lower() == v2.strip().lower()
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return abs(float(v1) - float(v2)) < 1e-6
    if isinstance(v1, dict) and isinstance(v2, dict):
        return v1.keys() == v2.keys() and all(_values_match(v1[k], v2[k]) for k in v1)
    if isinstance(v1, list) and isinstance(v2, list) and len(v1) == len(v2):
        return all(_values_match(a, b) for a, b in zip(v1, v2))
    return False


def _unwrap_json_string(value: Any) -> Any:
    """Decode a string that wraps a JSON array/object into the structured value."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
        if isinstance(decoded, (dict, list)):
            return decoded
    return value


def _param_content_score(
    label_args: dict[str, Any],
    output_args: dict[str, Any],
) -> float:
    """Score how well output parameter values match ground truth.

    Returns a score in [0, 1]:
      - Fraction of label params whose values match the output.
      - Extra output params (not in label) incur a 50 % penalty on the
        remainder so that spurious parameters don't go free.
    """
    if not label_args and not output_args:
        return 1.0
    if not label_args:
        # Label expects no params but output provided some → hallucinated args
        return 0.0

    correct = sum(
        1 for k, lv in label_args.items()
        if k in output_args and _values_match(lv, output_args[k])
    )
    # Penalty for extra/undeclared params in output
    extra = sum(1 for k in output_args if k not in label_args)
    penalty = 0.5 * extra / max(len(label_args) + extra, 1)
    return max(0.0, correct / len(label_args) - penalty)


def _format_call(call: dict[str, Any]) -> str:
    """Format a tool call as ``name(key=val, ...)`` for logging."""
    name = call.get("name", "?")
    args = call.get("arguments", {}) or {}
    if args:
        params = ", ".join(
            "%s=%s" % (k, json.dumps(v, ensure_ascii=False))
            for k, v in args.items()
        )
        return "%s(%s)" % (name, params)
    return name + "()"


def _guess_penalty(n_emitted: int) -> float:
    """Penalty for blind tool guessing: ``-0.1`` per emitted call, capped at ``-1.0``.

    Applied when the model emits tool calls that hit **none** of the ground
    truth labels (or when the label says no tools are needed but the model
    still called tools). Returns the magnitude (a non-negative float); callers
    negate it.
    """
    return min(0.1 * n_emitted, 1.0)


def match_tool_calls_against_label(
    output_calls: list[dict[str, Any]],
    label_calls: list[dict[str, Any]],
) -> tuple[float, float]:
    """Order-independent matching of tool calls against ground truth labels.

    Scoring uses **Jaccard-like** normalisation so both missed tools and
    extra/spurious calls are penalised:

        name_score   = matched / (M + N - matched)
        param_score = sum(matched_pair_scores) / (M + N - matched)

    where
        M = len(output_calls)
        N = len(label_calls)
        matched = number of label calls that found a partner by tool name.

    If **no** label call is matched (``matched == 0``) — the model guessed
    tool(s) that hit none of the ground truth labels — each emitted call is
    penalized to discourage blind guessing: ``name_score`` and
    ``param_score`` are both set to ``-min(0.1 * M, 1.0)``. This covers both
    "label requires tools but nothing matched" and "label requires no tools
    but the model still called tools". When both are empty, the score stays
    ``(1.0, 1.0)`` (correct "no tools" response).

    Args:
        output_calls: Parsed tool calls from the model output
            (``[{"name": …, "arguments": {…}}]``).
        label_calls: Ground truth tool calls from the dataset
            (``[{"name": …, "arguments": {…}}]``).

    Returns:
        Tuple ``(name_score, param_score)`` each in ``[0.0, 1.0]``.
    """
    # ── Log extracted calls ──────────────────────────────
    if output_calls:
        logger.info("[tool_rl] Model calls (%d):", len(output_calls))
        for i, c in enumerate(output_calls):
            logger.info("[tool_rl]   [%d] %s", i + 1, _format_call(c))
    else:
        logger.info("[tool_rl] Model calls: (none)")

    if label_calls:
        logger.info("[tool_rl] Label calls (%d):", len(label_calls))
        for i, c in enumerate(label_calls):
            logger.info("[tool_rl]   [%d] %s", i + 1, _format_call(c))
    else:
        logger.info("[tool_rl] Label calls: (none / no tools needed)")

    if not label_calls:
        if not output_calls:
            logger.info("[tool_rl] Match: both empty → 1.0")
            return (1.0, 1.0)
        # Label says "no tools needed" but the model still guessed tools
        penalty = _guess_penalty(len(output_calls))
        logger.info(
            "[tool_rl] Mismatch: label expects no tools, but model called %d tool(s) → penalty %.2f",
            len(output_calls), -penalty,
        )
        return (-penalty, -penalty)

    matched_indices: set[int] = set()
    pair_param_scores: list[float] = []

    for l_call in label_calls:
        l_name = l_call.get("name", "")
        l_args = l_call.get("arguments", {}) or {}
        best_param_score = -1.0  # start below 0 so even zero-param matches count
        best_idx = -1

        for oi, o_call in enumerate(output_calls):
            if oi in matched_indices:
                continue
            if o_call.get("name", "") != l_name:
                continue
            o_args = o_call.get("arguments", {}) or {}
            ps = _param_content_score(l_args, o_args)
            if ps > best_param_score:
                best_param_score = ps
                best_idx = oi

        if best_idx >= 0:
            matched_indices.add(best_idx)
            pair_param_scores.append(best_param_score)

    matched = len(pair_param_scores)

    # No label call hit — the model guessed tool(s) that missed entirely.
    if matched == 0:
        penalty = _guess_penalty(len(output_calls))
        logger.info(
            "[tool_rl] No label call matched (%d output vs %d label) → penalty %.2f",
            len(output_calls), len(label_calls), -penalty,
        )
        return (-penalty, -penalty)

    m = len(output_calls)
    n = len(label_calls)
    union = m + n - matched  # Jaccard denominator

    if union == 0:
        return (1.0, 1.0)

    name_score = matched / union
    param_score = sum(pair_param_scores) / union

    # ── Log unmatched calls ───────────────────────────────
    unmatched_output = [
        (i, _format_call(output_calls[i]))
        for i in range(m) if i not in matched_indices
    ]
    if unmatched_output:
        logger.info("[tool_rl] Unmatched output (%d):", len(unmatched_output))
        for idx, call_str in unmatched_output:
            logger.info("[tool_rl]   [#%d] %s", idx + 1, call_str)

    logger.info(
        "[tool_rl] Match result: name=%.3f param=%.3f (matched %d/%d label calls)",
        name_score, param_score, matched, n,
    )

    return (name_score, param_score)


def parse_ground_truth_calls(
    ground_truth: Any,
) -> list[dict[str, Any]]:
    """Normalise ground truth into ``[{"name": …, "arguments": {…}}]``.

    Handles formats from various data sources:
      - ``list[dict]`` with "name"/"arguments" keys (the canonical format).
      - A JSON string containing such a list.
      - A single dict (one tool call).
    """
    if not ground_truth:
        return []
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(ground_truth, dict):
        if "name" in ground_truth:
            ground_truth = [ground_truth]
        else:
            return []
    if not isinstance(ground_truth, list):
        return []

    normalised: list[dict[str, Any]] = []
    for item in ground_truth:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("function", "")
        if not name:
            continue
        args = item.get("arguments") or item.get("parameters") or {}
        if not isinstance(args, dict):
            args = {}
        normalised.append({"name": str(name), "arguments": args})
    return normalised


def compute_verifier_scores(
    trajectory: list[dict[str, Any]],
    *,
    available_tools: list[dict[str, Any]] | None = None,
    expects_no_tools: bool = False,
) -> dict[str, float]:
    """Compute Dim 2 + Dim 3 verifier scores.

    Args:
        trajectory: Normalized trajectory.
        available_tools: Tool definitions for Dim 3 format check.
        expects_no_tools: Label says no tools are needed for this task.

    Returns:
        ``{"format_compliance": float, "tool_call_format": float}``.
    """
    return {
        "format_compliance": check_format_compliance(
            trajectory,
            available_tools=available_tools,
            expects_no_tools=expects_no_tools,
        ),
        "tool_call_format": check_tool_call_format(
            trajectory,
            available_tools,
            expects_no_tools=expects_no_tools,
        ),
    }
