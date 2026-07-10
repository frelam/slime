"""LLM-judge utilities for agentic RL reward scoring.

Provides helpers to call SGLang (or any OpenAI-compatible endpoint) as a judge
for trajectory evaluation. Each benchmark adapter supplies its own system prompt
via ``llm_judge()``, and this module handles the HTTP call, response parsing,
and retry logic.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default judge prompt that asks for a score in [0, 1] with structured output.
DEFAULT_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for AI agent trajectories. Your task is to assess \
how well the agent completed the given objective.

Read the agent's conversation history and output a JSON object with exactly \
this structure:

{"score": <float between 0 and 1>, "reason": "<brief explanation>"}

- score=1.0: task fully completed correctly
- score=0.5: partial progress or minor errors
- score=0.0: completely wrong or no progress
"""


def build_judge_messages(
    system_prompt: str,
    task_description: str,
    trajectory: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the message list for an LLM-judge call.

    Args:
        system_prompt: Benchmark-specific judging instructions.
        task_description: The original task / problem statement.
        trajectory: List of turn dicts, each with at least ``"role"`` and
            ``"content"`` keys.

    Returns:
        List of message dicts ready for an OpenAI-compatible chat completion.
    """
    # Format trajectory as readable conversation
    conv_parts: list[str] = []
    for turn in trajectory:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if content:
            conv_parts.append(f"[{role}]: {content}")
    conversation = "\n\n".join(conv_parts)

    user_prompt = f"""## Task Description

{task_description}

## Agent Trajectory

{conversation}

## Instructions

Evaluate the agent's performance. Output your verdict as a JSON object:
{{"score": <0.0–1.0>, "reason": "<explanation>"}}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_judge_response(text: str) -> float | None:
    """Extract the numeric score from an LLM-judge response.

    Handles several common formats:
    - Pure JSON: ``{"score": 0.8, "reason": "..."}``
    - JSON block: `` ```json {...} ``` ``
    - Bare number fallback: scans for a float in [0, 1].

    Returns:
        Score in [0, 1] or None if parsing failed.
    """
    text = text.strip()

    # Try to find a JSON block first
    candidates: list[str] = []

    # Inline JSON object
    if text.startswith("{"):
        candidates.append(text)
    else:
        # Extract from markdown code fences
        import re
        for match in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
            candidates.append(match.group(1).strip())

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "score" in obj:
            score = obj["score"]
            if isinstance(score, (int, float)):
                return float(min(max(score, 0.0), 1.0))

    # Fallback: look for a standalone float
    import re
    for match in re.finditer(r"\b([01](?:\.\d+)?|0?\.\d+)\b", text):
        val = float(match.group(1))
        if 0.0 <= val <= 1.0:
            return val

    logger.warning("Could not parse judge score from response: %r", text[:500])
    return None


async def call_llm_judge(
    args: Any,
    messages: list[dict[str, str]],
    *,
    max_retries: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> float | None:
    """Call the SGLang endpoint as an LLM judge.

    Uses the same SGLang router / engine that serves the rollout model.
    The judge model is the current actor model (self-judge) unless overridden
    by ``--llm-judge-model``.

    Args:
        args: Slime training arguments (must have sglang_router_ip/port or
            per-model host info).
        messages: Chat-format message list.
        max_retries: Number of retries on transient failures.
        temperature: Sampling temperature (0.0 for deterministic judging).
        max_tokens: Max output tokens.

    Returns:
        Score in [0, 1] or None on failure.
    """
    import aiohttp

    # Resolve the SGLang endpoint
    router_ip = getattr(args, "sglang_router_ip", "127.0.0.1")
    router_port = getattr(args, "sglang_router_port", 30000)
    url = f"http://{router_ip}:{router_port}/v1/chat/completions"

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(
                            f"Judge API returned {resp.status}: {text[:200]}"
                        )
                    data = await resp.json()
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    score = parse_judge_response(content)
                    if score is not None:
                        return score
                    last_error = f"parse failed: {content[:200]}"
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                logger.warning(
                    "LLM-judge attempt %d/%d failed: %s",
                    attempt + 1, max_retries + 1, exc,
                )
                import asyncio
                await asyncio.sleep(1.0 * (attempt + 1))

    logger.error("LLM-judge failed after %d retries: %s", max_retries + 1, last_error)
    return None


def combine_rewards(
    rule_reward: float,
    llm_reward: float | None,
    llm_weight: float,
) -> float:
    """Combine rule-based and LLM-judge rewards.

    Args:
        rule_reward: Rule-based reward in [0, 1].
        llm_reward: LLM-judge reward in [0, 1] or None.
        llm_weight: Weight for LLM-judge (0.0 = all rule, 1.0 = all LLM).

    Returns:
        Combined reward in [0, 1].
    """
    if llm_reward is None:
        return rule_reward
    return (1.0 - llm_weight) * rule_reward + llm_weight * llm_reward
