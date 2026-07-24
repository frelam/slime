"""Tool RL reward composer — 3 dimensions with rule-first design.

Reward Dimensions
-----------------
==============  ==============================  ======  =============
Dim             Name                            Weight  Source
==============  ==============================  ======  =============
Dim 1           工具调用正确性 (Tool Correctness)  0.60    Label match
                                                         or RM v2
Dim 2           回复格式合规 (Format)              0.20    Verifier
Dim 3           工具调用格式 (Tool Call Format)    0.20    Verifier
==============  ==============================  ======  =============

Dim 1 has two modes:

**Label mode** (dataset provides ground truth tool calls):
  Rule-based matching, order-independent.
  - Tool name match  → 0.5  (binary per label call)
  - Param content    → 0.5  (value match per label param)

**RM mode** (no ground truth):
  LLM judge scores two sub-dimensions, same weights:
  - Tool name correctness (0.0-1.0)
  - Parameter content correctness (0.0-1.0)

Dim 2 — Verifier (format):
  0.6 if all tool_calls after reasoning + 0.4 × count/N for think before each call

Dim 3 — Verifier (tool call format):
  1/N × 0.5 name + 1/N × 0.3 param_name + 1/N × 0.2 param_type per call
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# Weights
# ============================================================================

DEFAULT_WEIGHTS: dict[str, float] = {
    "tool_correctness": 0.60,  # replaces old planning(0.40) + hallucination(0.20)
    "format": 0.20,
    "tool_call": 0.20,
}

# Backward-compat mapping: old → new dimension keys
_OLD_TO_NEW: dict[str, str] = {
    "planning": "tool_correctness",
    "hallucination": "tool_correctness",
}


def get_weights(args: Any) -> dict[str, float]:
    """Resolve weights from ``--reward-weights`` JSON, else defaults."""
    raw = getattr(args, "reward_weights", None)
    defaults = dict(DEFAULT_WEIGHTS)

    if raw is None:
        return defaults
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid --reward-weights JSON: %r", raw)
            return defaults
    if not isinstance(raw, dict):
        return defaults

    # Map old dimension keys to new (e.g. planning → tool_correctness)
    for old_k, new_k in _OLD_TO_NEW.items():
        if old_k in raw:
            if new_k not in defaults:
                defaults[new_k] = 0.0
            defaults[new_k] += float(raw.pop(old_k))

    for k in defaults:
        if k in raw:
            defaults[k] = float(raw[k])
    total = sum(defaults.values())
    if total > 0:
        defaults = {k: v / total for k, v in defaults.items()}
    return defaults


# ============================================================================
# Reward breakdown
# ============================================================================


@dataclass
class ToolRLRewardBreakdown:
    total: float
    tool_correctness: float       # 0-1 composite: name 0.5 + param content 0.5
    name_score: float             # tool name match sub-score
    param_content_score: float    # parameter content match sub-score
    format_compliance: float      # Dim 2 — verifier
    tool_call_format: float       # Dim 3 — verifier
    source: str = "label"         # "label" or "rm"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return {
            "reward/total": self.total,
            "reward/tool_correctness": self.tool_correctness,
            "reward/name_score": self.name_score,
            "reward/param_content_score": self.param_content_score,
            "reward/format_compliance": self.format_compliance,
            "reward/tool_call_format": self.tool_call_format,
        }


# ============================================================================
# Main entry point
# ============================================================================


async def compute_tool_rl_reward(
    args: Any,
    trajectory: list[dict[str, Any]],
    task_description: str,
    *,
    available_tools: list[dict[str, Any]] | None = None,
    ground_truth_label: str = "",
    ground_truth_calls: list[dict[str, Any]] | None = None,
) -> ToolRLRewardBreakdown:
    """Compute the 3-dim tool RL reward.

    Two modes of operation:

    **Label mode** (``ground_truth_calls`` is non-empty):
      Dim 1 (tool_correctness) is scored by rule-based, order-independent
      matching of the model's tool calls against the ground truth labels.
      Tool name match = 0.5, parameter content match = 0.5 within the dim.

    **RM mode** (no ``ground_truth_calls``):
      Dim 1 is scored by an LLM judge (RM) along the same two sub-dimensions:
      tool name correctness and parameter content correctness.

    Dim 2 (format) and Dim 3 (tool_call_format) are always rule-based
    verifiers (see ``verifier.py``).

    Args:
        args: Slime training args.
        trajectory: Normalized trajectory from generate function.
        task_description: The task prompt (for RM context).
        available_tools: Tool definitions for Dim 3 verification.
        ground_truth_label: Ground truth string (for RM reference).
        ground_truth_calls: Structured ground truth tool calls
            (``[{"name": …, "arguments": {…}}]``) for rule-based matching.

    Returns:
        ``ToolRLRewardBreakdown`` with all dimension scores.
    """
    from examples.tool_rl.reward.verifier import (
        compute_verifier_scores,
        match_tool_calls_against_label,
        parse_ground_truth_calls,
        parse_qwen_tool_calls,
    )

    weights = get_weights(args)

    # ── Dim 2 + Dim 3: Verifier (rule-based, always on) ──
    verifier = compute_verifier_scores(trajectory, available_tools=available_tools)
    format_score = verifier["format_compliance"]
    tool_call_score = verifier["tool_call_format"]

    # Parse the model's tool calls from trajectory text
    all_text = _get_agent_text(trajectory)
    output_calls = parse_qwen_tool_calls(all_text)

    # ── Dim 1: Tool Call Correctness ────────────────────
    parsed_gt = parse_ground_truth_calls(ground_truth_calls)

    if parsed_gt:
        # ── Label mode: rule-based matching ──
        source = "label"
        name_score, param_score = match_tool_calls_against_label(
            output_calls, parsed_gt,
        )
        tool_correctness = 0.5 * name_score + 0.5 * param_score

        details: dict[str, Any] = {
            "source": "label",
            "n_label_calls": len(parsed_gt),
            "n_output_calls": len(output_calls),
        }
    else:
        # ── RM mode: LLM judge (structured 2-dim scoring) ──
        source = "rm"
        if _is_garbled_output(trajectory):
            name_score = 0.0
            param_score = 0.0
            tool_correctness = 0.0
            details = {
                "source": "rm",
                "name_reason": "Output is garbled — floor score",
                "param_reason": "Output is garbled — floor score",
            }
        else:
            rm = await _call_rm_v2(
                args, trajectory, task_description, ground_truth_label,
            )
            name_score = rm["tool_name_score"]
            param_score = rm["param_content_score"]
            tool_correctness = 0.5 * name_score + 0.5 * param_score
            details = {
                "source": "rm",
                "name_reason": rm.get("name_reason", ""),
                "param_reason": rm.get("param_reason", ""),
            }

    # ── Weighted sum ────────────────────────────────────
    total = (
        weights["tool_correctness"] * tool_correctness
        + weights["format"] * format_score
        + weights["tool_call"] * tool_call_score
    )
    total = max(0.0, min(1.0, total))

    breakdown = ToolRLRewardBreakdown(
        total=total,
        tool_correctness=tool_correctness,
        name_score=name_score,
        param_content_score=param_score,
        format_compliance=format_score,
        tool_call_format=tool_call_score,
        source=source,
        details=details,
    )

    logger.info(
        "Tool RL: total=%.3f correctness=%.3f(name=%.3f+param=%.3f) "
        "format=%.3f tool_call=%.3f src=%s",
        total, tool_correctness, name_score, param_score,
        format_score, tool_call_score, source,
    )
    return breakdown


def _get_agent_text(trajectory: list[dict[str, Any]]) -> str:
    """Extract assistant-generated text from a trajectory."""
    return "\n".join(
        r.get("text", "") for r in trajectory if r.get("type") != "observation"
    )


def _is_garbled_output(trajectory: list[dict[str, Any]]) -> bool:
    """Heuristic: check if the assistant output is garbled/gibberish.

    Early untrained models often produce repetitive token sequences that the
    RM (especially when pointing at the same SGLang endpoint) will incorrectly
    score as reasonable.  We detect this via character and trigram diversity
    and skip the RM call entirely.
    """
    text = " ".join(
        msg.get("content", "") for msg in trajectory if msg.get("role") == "assistant"
    )
    # Strip XML tags to get the raw reasoning/text content
    text = re.sub(r"<[^>]+>", "", text).strip()
    if len(text) < 20:
        return False  # too short to reliably detect

    # Character-set diversity — garbled text reuses very few characters
    char_diversity = len(set(text)) / max(1, len(text))

    # Trigram diversity — high repetition of 3-char windows
    trigrams = [text[i : i + 3] for i in range(len(text) - 2)]
    trigram_diversity = len(set(trigrams)) / len(trigrams) if trigrams else 1.0

    return char_diversity < 0.15 and trigram_diversity < 0.15


# ============================================================================
# RM call — Dim 1 + Dim 4
# ============================================================================


async def _call_rm_v2(
    args: Any,
    trajectory: list[dict[str, Any]],
    task_description: str,
    ground_truth_label: str,
    *,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Call RM for tool name + param content scoring (structured 2-dim).

    The RM evaluates two clear sub-dimensions:
      1. Tool name correctness (0.0-1.0): whether the right tools were
         selected for the task.
      2. Parameter content correctness (0.0-1.0): whether parameter
         values are reasonable / not fabricated.

    API key from ``RM_API_KEY`` env var (never CLI).

    Returns:
        ``{"tool_name_score": float, "param_content_score": float,
          "name_reason": str, "param_reason": str}``
    """
    import aiohttp
    import asyncio

    # 1. System prompt
    prompt_dir = getattr(args, "rm_system_prompt_dir",
                         "examples/tool_rl/reward/prompts")
    system_prompt = _load_prompt("tool_rl", prompt_dir)

    # 2. User message
    traj_text = _format_traj(trajectory)
    user = (
        "## Task Description\n\n"
        f"{task_description[:5000]}\n\n"
        "## Agent Trajectory\n\n"
        f"{traj_text}\n\n"
    )
    if ground_truth_label:
        user += f"## Ground Truth (Reference)\n\n{ground_truth_label[:2000]}\n\n"
    user += "Output your evaluation as a JSON object."

    # 3. Endpoint
    rm_type = getattr(args, "rm_model_type", None) or os.environ.get("RM_MODEL_TYPE", "sglang")
    endpoint = getattr(args, "rm_model_endpoint", None) or os.environ.get("RM_MODEL_ENDPOINT", None)
    if not endpoint:
        ip = getattr(args, "sglang_router_ip", "127.0.0.1")
        port = getattr(args, "sglang_router_port", 30000)
        endpoint = f"http://{ip}:{port}/v1/chat/completions"

    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    if rm_type == "deepseek":
        api_key = os.environ.get("RM_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if not (getattr(args, "rm_model_endpoint", None) or os.environ.get("RM_MODEL_ENDPOINT", None)):
            endpoint = "https://api.deepseek.com/v1/chat/completions"
            payload["model"] = "deepseek-chat"

    # 4. Call with retry
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    endpoint, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        t = await resp.text()
                        raise RuntimeError(f"RM {resp.status}: {t[:300]}")
                    data = await resp.json()
                    content = (data.get("choices", [{}])[0]
                               .get("message", {}).get("content", ""))
                    result = _parse_rm_v2(content)
                    if result:
                        logger.info("RMv2: name=%.3f param=%.3f",
                                    result["tool_name_score"],
                                    result["param_content_score"])
                        return result
                    last_err = f"parse: {content[:200]}"
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                logger.warning("RM %d/%d: %s", attempt + 1, max_retries + 1, e)
                await asyncio.sleep(1.0 * (attempt + 1))

    logger.error("RM failed after %d retries: %s — aborting sample", max_retries + 1, last_err)
    raise RuntimeError(f"RM failed: {last_err}")


def _parse_rm_v2(text: str) -> dict | None:
    """Parse RM v2 JSON response.

    Expected format::

        {"tool_name_score": 0.8, "param_content_score": 0.6,
         "name_reason": "...", "param_reason": "..."}
    """
    text = text.strip()
    cands: list[str] = []
    if text.startswith("{"):
        cands.append(text)
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        cands.append(m.group(1).strip())
    for m in re.finditer(
        r'\{[^{}]*"tool_name_score"[^{}]*(?"param_content_score"[^{}]*)*\}',
        text, re.DOTALL,
    ):
        cands.append(m.group(0))

    for c in cands:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "tool_name_score" in obj:
            return {
                "tool_name_score": max(0.0, min(1.0, float(obj.get("tool_name_score", 0.5)))),
                "param_content_score": max(0.0, min(1.0, float(obj.get("param_content_score", 0.5)))),
                "name_reason": str(obj.get("name_reason", "")),
                "param_reason": str(obj.get("param_reason", "")),
            }
    logger.warning("Could not parse RMv2: %r", text[:500])
    return None


# ============================================================================
# Helpers
# ============================================================================

_prompt_cache: dict[str, str] = {}


def _load_prompt(task_type: str, prompt_dir: str) -> str:
    cache_key = f"{prompt_dir}/{task_type}"
    if cache_key in _prompt_cache:
        return _prompt_cache[cache_key]
    p = Path(prompt_dir) / f"{task_type}.md"
    if p.exists():
        content = p.read_text(encoding="utf-8")
        _prompt_cache[cache_key] = content
        return content
    fallback = (
        "You are an expert evaluator for AI agent tool-use trajectories.\n\n"
        "Evaluate on two dimensions:\n"
        "1. Tool Name Correctness (0.0-1.0): Did the agent select the correct "
        "tools for the task? 1.0 = perfectly appropriate, 0.0 = completely wrong.\n"
        "2. Parameter Content Correctness (0.0-1.0): Are the parameter values "
        "reasonable and correct? Penalize fabrication / hallucination.\n\n"
        'Respond ONLY with JSON: {"tool_name_score": <float>, '
        '"param_content_score": <float>, '
        '"name_reason": "...", "param_reason": "..."}'
    )
    _prompt_cache[cache_key] = fallback
    return fallback


def _format_traj(trajectory: list[dict]) -> str:
    parts = []
    for rec in trajectory:
        turn = rec.get("turn", 0)
        rtype = rec.get("type", "turn")
        text = rec.get("text", "")
        if rtype == "observation":
            tc = rec.get("tool_call", {})
            name = tc.get("name", "?") if isinstance(tc, dict) else "?"
            parts.append(f"\n### Turn {turn} — Tool: `{name}`\n```\n{text[:2000]}\n```\n")
        else:
            fin = rec.get("finish_reason", "")
            parts.append(f"\n### Turn {turn} — Agent ({fin})\n{text[:3000]}\n")
    return "\n".join(parts) if parts else "(empty)"


# ============================================================================
# --custom-rm-path adapter
# ============================================================================


async def tool_rl_reward(args: Any, sample: Any) -> float | list[float]:
    """Reward for ``--custom-rm-path``. Pass-through from generate phase.

    NOTE: ``batched_async_rm`` passes a ``list[Sample]`` to custom RM
    functions when ``--custom-rm-path`` is set.  We expect the reward was
    already computed during generation (``tool_rl_grpo_generate``), so
    we simply extract it from each sample.
    """
    if isinstance(sample, list):
        # Batched mode: list[Sample] — reward already set in generate
        rewards = []
        for s in sample:
            r = _extract_reward(s)
            rewards.append(r)
        return rewards

    return _extract_reward(sample)


def _extract_reward(sample: Any) -> float:
    """Extract the pre-computed reward from a sample."""
    if sample.reward is not None:
        try:
            return float(sample.reward)
        except (TypeError, ValueError):
            pass

    metadata = sample.metadata or {}
    traj = metadata.get("trajectory")
    if traj:
        tools = metadata.get("tools", [])
        desc = sample.prompt if isinstance(sample.prompt, str) else ""
        gt_label = sample.label or ""
        gt_calls = metadata.get("ground_truth", None)
        bd = asyncio.run(
            compute_tool_rl_reward(
                None, traj, desc,
                available_tools=tools,
                ground_truth_label=gt_label,
                ground_truth_calls=gt_calls,
            )
        )
        return bd.total

    return 0.0
