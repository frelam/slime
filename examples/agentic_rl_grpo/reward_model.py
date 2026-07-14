"""Reward Model (RM) API client for agentic RL.

Calls an external RM (Qwen3.5-9B via SGLang, or DeepSeek API) to score
the subjective dimensions of an agent trajectory:

- Dim 4.1: Answer correctness / reasonableness (0 or 1)
- Dim 4.5: Planning quality (0.0–1.0)
- Dim 4.6: Hallucination (0 or 1)

The RM system prompt is loaded from a per-task-type ``.md`` file, so users
can tune the judging criteria without touching code.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RMResult:
    """Structured output from the reward model."""

    correctness: float  # 0.0 or 1.0
    planning: float  # 0.0–1.0
    hallucination: float  # 0.0 or 1.0
    reason: str = ""

    @classmethod
    def neutral(cls) -> RMResult:
        """Return a neutral result (used when RM is unavailable)."""
        return cls(correctness=0.5, planning=0.5, hallucination=0.5, reason="RM unavailable — neutral scores")


# ---------------------------------------------------------------------------
# System prompt loading
# ---------------------------------------------------------------------------

# Module-level cache so we only read each .md file once.
_prompt_cache: dict[str, str] = {}


def load_system_prompt(task_type: str, prompt_dir: str) -> str:
    """Load the RM system prompt for *task_type* from *prompt_dir*.

    Caches loaded prompts in memory. If the file is not found, returns
    a generic default prompt.

    Args:
        task_type: Benchmark name, e.g. ``"terminal_bench"``, ``"swe_gym_lite"``.
        prompt_dir: Path to the directory containing ``{task_type}.md`` files.

    Returns:
        The system prompt string (markdown).
    """
    cache_key = f"{prompt_dir}/{task_type}"
    if cache_key in _prompt_cache:
        return _prompt_cache[cache_key]

    prompt_path = Path(prompt_dir) / f"{task_type}.md"
    if prompt_path.exists():
        content = prompt_path.read_text(encoding="utf-8")
        _prompt_cache[cache_key] = content
        logger.info("Loaded RM system prompt from %s", prompt_path)
        return content

    # Fallback: generic prompt
    logger.warning(
        "RM system prompt not found at %s, using default",
        prompt_path,
    )
    fallback = _default_system_prompt(task_type)
    _prompt_cache[cache_key] = fallback
    return fallback


def _default_system_prompt(task_type: str) -> str:
    """Generate a generic fallback system prompt."""
    return f"""You are an expert evaluator for AI agent trajectories in {task_type} tasks.

Evaluate the trajectory on three dimensions:

1. **Correctness** (0 or 1): Did the agent accomplish the task correctly?
2. **Planning Quality** (0.0 to 1.0): How well did the agent plan its approach?
3. **Hallucination** (0 or 1): Did the agent make false claims or reference non-existent things? (1 = no hallucination)

Respond ONLY with a JSON object:
{{"correctness": <0|1>, "planning": <0.0-1.0>, "hallucination": <0|1>, "reason": "<brief>"}}"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_rm_response(text: str) -> RMResult | None:
    """Extract scores from an RM's JSON response.

    Handles:
    - Pure JSON: ``{"correctness": 1, "planning": 0.8, ...}``
    - Code-fenced JSON: `` ```json {...} ``` ``
    - Inline JSON embedded in explanatory text.

    Args:
        text: Raw RM response text.

    Returns:
        ``RMResult`` if parsing succeeded, ``None`` otherwise.
    """
    text = text.strip()

    candidates: list[str] = []

    # Try inline JSON first
    if text.startswith("{"):
        candidates.append(text)

    # Try code-fence extraction
    for match in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        candidates.append(match.group(1).strip())

    # Also try to find JSON-like objects anywhere
    for match in re.finditer(r"\{[^{}]*\"correctness\"[^{}]*\}", text, re.DOTALL):
        if match.group(0) not in candidates:
            candidates.append(match.group(0))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue

        correctness = obj.get("correctness")
        planning = obj.get("planning") or obj.get("planning_quality")
        hallucination = obj.get("hallucination")

        if correctness is None and planning is None and hallucination is None:
            continue

        return RMResult(
            correctness=float(correctness) if correctness is not None else 0.5,
            planning=float(planning) if planning is not None else 0.5,
            hallucination=float(hallucination) if hallucination is not None else 0.5,
            reason=str(obj.get("reason", "")),
        )

    logger.warning("Could not parse RM response: %r", text[:500])
    return None


# ---------------------------------------------------------------------------
# RM API call
# ---------------------------------------------------------------------------


async def call_reward_model(
    args: Any,
    task_type: str,
    trajectory: list[dict[str, Any]],
    task_description: str,
    *,
    max_retries: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> RMResult:
    """Call the reward model and return dimension scores.

    Args:
        args: Slime training arguments. Uses:
            - ``rm_model_endpoint``: URL for RM endpoint.
            - ``rm_model_type``: ``"sglang"`` or ``"deepseek"``.
            - ``rm_api_key``: API key for DeepSeek.
            - ``rm_system_prompt_dir``: Directory with ``.md`` prompt files.
            - ``sglang_router_ip`` / ``sglang_router_port``: Fallback if no
              endpoint set.
        task_type: Benchmark name (e.g. ``"terminal_bench"``).
        trajectory: Normalized trajectory list.
        task_description: The task prompt / problem statement.
        max_retries: Retries on transient failure.
        temperature: Sampling temperature (0 = deterministic).
        max_tokens: Max output tokens.

    Returns:
        ``RMResult`` with scores (or neutral scores on failure).
    """
    import aiohttp

    # 1. Load system prompt
    prompt_dir = getattr(args, "rm_system_prompt_dir", "examples/agentic_rl_grpo/prompts")
    system_prompt = load_system_prompt(task_type, prompt_dir)

    # 2. Build messages
    from examples.agentic_rl_grpo.traj_analysis import format_for_rm

    formatted_traj = format_for_rm(trajectory, task_description)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Please evaluate the following agent trajectory:\n\n"
                + formatted_traj
                + "\n\nOutput your evaluation as a JSON object."
            ),
        },
    ]

    # 3. Resolve endpoint
    rm_type = getattr(args, "rm_model_type", "sglang") or "sglang"
    endpoint = getattr(args, "rm_model_endpoint", None)

    if not endpoint:
        router_ip = getattr(args, "sglang_router_ip", "127.0.0.1")
        router_port = getattr(args, "sglang_router_port", 30000)
        endpoint = f"http://{router_ip}:{router_port}/v1/chat/completions"

    # 4. Build payload
    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if rm_type == "deepseek":
        api_key = getattr(args, "rm_api_key", None)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # DeepSeek uses https://api.deepseek.com/v1/chat/completions
        if not getattr(args, "rm_model_endpoint", None):
            endpoint = "https://api.deepseek.com/v1/chat/completions"
            payload["model"] = "deepseek-chat"

    # 5. Call with retry
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(
                            f"RM API returned {resp.status}: {text[:300]}"
                        )
                    data = await resp.json()
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    result = parse_rm_response(content)
                    if result is not None:
                        logger.info(
                            "RM scores for %s: correctness=%.0f planning=%.2f hallucination=%.0f",
                            task_type,
                            result.correctness,
                            result.planning,
                            result.hallucination,
                        )
                        return result
                    last_error = f"parse failed: {content[:200]}"
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                logger.warning(
                    "RM attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                import asyncio

                await asyncio.sleep(1.0 * (attempt + 1))

    logger.error(
        "RM call failed after %d retries: %s. Using neutral scores.",
        max_retries + 1,
        last_error,
    )
    return RMResult.neutral()
