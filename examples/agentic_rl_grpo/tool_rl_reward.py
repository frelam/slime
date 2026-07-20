"""Tool RL reward composer — strict 4-dimensions per user spec.

Dimensions
----------
===========  ==============================  ======  ========
Dim          Name                            Weight  Source
===========  ==============================  ======  ========
Dim 1        思考与规划质量 (Planning)           0.40    RM
Dim 2        回复格式合规 (Format)               0.20    Verifier
Dim 3        工具调用格式 (Tool Call Format)     0.20    Verifier
Dim 4        臆想检测 (Hallucination)            0.20    RM
===========  ==============================  ======  ========

Dim 1 — RM scored (planning):
  优 (Excellent):  1.0 — correct understanding + reasoning + planning + tools
  良 (Good):       0.6 — correct understanding + planning, wrong tools/deps
  合格 (Adequate): 0.3 — correct understanding, planning+tool errors
  差 (Poor):      -0.2 — wrong understanding, no tools when needed, too cautious

Dim 2 — Verifier (format):
  0.6 if all tool_calls after reasoning + 0.4 × count/N for think before each call

Dim 3 — Verifier (tool call format):
  1/N × 0.5 name + 1/N × 0.3 param_name + 1/N × 0.2 param_type per call

Dim 4 — RM scored (hallucination):
  0 = fabricated info, 1 = fact-based / humble
"""

from __future__ import annotations

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
    "planning": 0.40,
    "format": 0.20,
    "tool_call": 0.20,
    "hallucination": 0.20,
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
    planning: float
    format_compliance: float
    tool_call_format: float
    hallucination: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return {
            "reward/total": self.total,
            "reward/planning": self.planning,
            "reward/format_compliance": self.format_compliance,
            "reward/tool_call_format": self.tool_call_format,
            "reward/hallucination": self.hallucination,
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
) -> ToolRLRewardBreakdown:
    """Compute the 4-dim tool RL reward.

    Args:
        args: Slime training args.
        trajectory: Normalized trajectory from generate function.
        task_description: The task prompt (for RM context).
        available_tools: Tool definitions for Dim 3 verification.
        ground_truth_label: Ground truth string (for RM reference).

    Returns:
        ``ToolRLRewardBreakdown`` with all 4 dimension scores.
    """
    from examples.agentic_rl_grpo.tool_rl_verifier import compute_verifier_scores

    weights = get_weights(args)

    # ---- Dim 2 + Dim 3: Verifier ----
    verifier = compute_verifier_scores(trajectory, available_tools=available_tools)
    format_score = verifier["format_compliance"]
    tool_call_score = verifier["tool_call_format"]

    # ---- Dim 1 + Dim 4: RM ----
    rm = await _call_rm(args, trajectory, task_description, ground_truth_label)

    planning_raw = rm["planning_score"]   # 1.0 / 0.6 / 0.3 / -0.2
    halluc_raw = rm["hallucination_score"]  # 0 or 1

    # Normalize planning from [-0.2, 1.0] → [0.0, 1.0]
    planning_norm = max(0.0, (planning_raw + 0.2) / 1.2)

    # ---- Weighted sum ----
    total = (
        weights["planning"] * planning_norm
        + weights["format"] * format_score
        + weights["tool_call"] * tool_call_score
        + weights["hallucination"] * halluc_raw
    )
    total = max(0.0, min(1.0, total))

    breakdown = ToolRLRewardBreakdown(
        total=total,
        planning=planning_raw,
        format_compliance=format_score,
        tool_call_format=tool_call_score,
        hallucination=halluc_raw,
        details={
            "weights": weights,
            "planning_reason": rm.get("planning_reason", ""),
            "hallucination_reason": rm.get("hallucination_reason", ""),
        },
    )

    logger.info(
        "Tool RL: total=%.3f planning=%.1f(n=%.3f) format=%.3f "
        "tool_call=%.3f halluc=%.0f",
        total, planning_raw, planning_norm, format_score,
        tool_call_score, halluc_raw,
    )
    return breakdown


# ============================================================================
# RM call — Dim 1 + Dim 4
# ============================================================================


async def _call_rm(
    args: Any,
    trajectory: list[dict[str, Any]],
    task_description: str,
    ground_truth_label: str,
    *,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Call RM for planning (Dim 1) + hallucination (Dim 4) scores.

    API key from ``RM_API_KEY`` env var (never CLI).
    """
    import aiohttp
    import asyncio

    # 1. System prompt
    prompt_dir = getattr(args, "rm_system_prompt_dir",
                         "examples/agentic_rl_grpo/prompts")
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
    rm_type = getattr(args, "rm_model_type", "sglang") or "sglang"
    endpoint = getattr(args, "rm_model_endpoint", None)
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
        if not getattr(args, "rm_model_endpoint", None):
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
                    result = _parse_rm(content)
                    if result:
                        logger.info("RM: planning=%.1f halluc=%.0f",
                                    result["planning_score"],
                                    result["hallucination_score"])
                        return result
                    last_err = f"parse: {content[:200]}"
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                logger.warning("RM %d/%d: %s", attempt + 1, max_retries + 1, e)
                await asyncio.sleep(1.0 * (attempt + 1))

    logger.error("RM failed: %s — neutral", last_err)
    return {
        "planning_score": 0.5,
        "hallucination_score": 1.0,
        "planning_reason": "RM unavailable",
        "hallucination_reason": "RM unavailable",
    }


def _parse_rm(text: str) -> dict | None:
    """Parse RM JSON response."""
    text = text.strip()
    cands = []
    if text.startswith("{"):
        cands.append(text)
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        cands.append(m.group(1).strip())
    for m in re.finditer(
        r'\{[^{}]*"planning_score"[^{}]*"hallucination_score"[^{}]*\}',
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
        if "planning_score" in obj:
            return {
                "planning_score": float(obj.get("planning_score", 0.5)),
                "hallucination_score": float(obj.get("hallucination_score", 1.0)),
                "planning_reason": str(obj.get("planning_reason", "")),
                "hallucination_reason": str(obj.get("hallucination_reason", "")),
            }
    logger.warning("Could not parse RM: %r", text[:500])
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
        "1. Planning Score: 1.0/0.6/0.3/-0.2\n"
        "2. Hallucination Score: 0 (fabricated) / 1 (fact-based)\n\n"
        'Respond ONLY with JSON: {"planning_score": <float>, "hallucination_score": <0|1>, "planning_reason": "...", "hallucination_reason": "..."}'
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


async def tool_rl_reward(args: Any, sample: Any) -> float:
    """Reward for ``--custom-rm-path``. Pass-through from generate phase."""
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
        bd = await compute_tool_rl_reward(
            args, traj, desc,
            available_tools=tools,
            ground_truth_label=gt_label,
        )
        return bd.total

    return 0.0
