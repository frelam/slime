"""Custom generate function for single-turn tool-use RL GRPO — Qwen3-4B.

Plug into slime via ``--custom-generate-function-path``::

    python train.py \\
        --custom-generate-function-path examples.agentic_rl_grpo.tool_rl_generate.tool_rl_grpo_generate \\
        --custom-rm-path examples.agentic_rl_grpo.tool_rl_reward.tool_rl_reward \\
        --input-key messages \\
        --tool-key tools \\
        --apply-chat-template \\
        ...

Flow (single-turn)
------------------
1. Prompt is already formatted by Qwen chat template (via ``--apply-chat-template``).
2. Single SGLang generate → model outputs ``<think>...</think>``
   followed by ``<tool_call>...</tool_call>`` (Qwen XML format).
3. Parse response into pseudo-trajectory for verifier/RM consumption.
4. Call ``compute_tool_rl_reward()`` → 4-dim weighted reward:
   - Dim 1 (0.40): Planning — RM scored (优/良/合格/差)
   - Dim 2 (0.20): Format — Verifier (<think> + <tool_call> format)
   - Dim 3 (0.20): Tool Call — Verifier (name + param name + param type)
   - Dim 4 (0.20): Hallucination — RM scored (0/1)
5. Return ``list[Sample]`` with logprobs + reward.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

CONFIG = {
    "rollout_guard_sec": int(os.environ.get("TOOL_RL_ROLLOUT_GUARD_SEC", "300")),
}

# ============================================================================
# Qwen format regex
# ============================================================================

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE,
)
_FUNCTION_NAME_RE = re.compile(r"<function=(\w[\w.]*)>")


# ============================================================================
# SGLang helper
# ============================================================================


async def _call_sglang(
    router_ip: str,
    router_port: int,
    input_ids: list[int],
    sampling_params: dict[str, Any] | None = None,
    return_logprob: bool = True,
) -> dict[str, Any]:
    import aiohttp

    url = f"http://{router_ip}:{router_port}/generate"
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": sampling_params or {},
        "return_logprob": return_logprob,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()

    meta = result.get("meta_info") or {}
    finish_reason = meta.get("finish_reason", {})
    return {
        "output_ids": result.get("output_ids", []),
        "text": result.get("text", ""),
        "finish_reason": (
            finish_reason.get("type", "stop")
            if isinstance(finish_reason, dict)
            else str(finish_reason)
        ),
        "meta_info": meta,
        "logprobs": [
            float(lp[0]) if isinstance(lp, (list, tuple)) else float(lp)
            for lp in meta.get("output_token_logprobs", [])
        ] if return_logprob else [],
    }


# ============================================================================
# Response → pseudo-trajectory (for verifier/RM)
# ============================================================================


def _response_to_trajectory(text: str) -> list[dict[str, Any]]:
    """Convert a single response into a trajectory for verifier consumption.

    Parses Qwen XML format::

        <think>reasoning</think>
        <tool_call><function=X><parameter=Y>val</parameter></function></tool_call>

    Produces:
        [{"turn": 0, "text": "<think>...</think>\n\n<tool_call>...</tool_call>",
          "finish_reason": "stop",
          "tool_calls": [{"name": "X", "arguments": {"Y": "val"}}]}]
    """
    # Parse tool calls for the verifier
    tool_calls = []
    for tc_match in _TOOL_CALL_BLOCK_RE.finditer(text):
        block = tc_match.group(1)
        func_match = _FUNCTION_NAME_RE.search(block)
        if not func_match:
            continue
        args = {}
        for pm in re.finditer(
            r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", block, re.DOTALL,
        ):
            pval = pm.group(2).strip()
            try:
                pval = json.loads(pval)
            except (json.JSONDecodeError, TypeError):
                pass
            args[pm.group(1)] = pval
        tool_calls.append({"name": func_match.group(1), "arguments": args})

    return [{
        "turn": 0,
        "text": text,
        "finish_reason": "stop",
        "type": "turn",
        "tool_calls_parsed": tool_calls,
    }]


# ============================================================================
# Main generate function
# ============================================================================


async def tool_rl_grpo_generate(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any] | None = None,
    evaluation: bool = False,
) -> list[Sample]:
    """Single-turn tool-use GRPO generate for Qwen3-4B.

    Args:
        args: Slime training args.
        sample: Input sample — prompt is already chat-template-formatted.
        sampling_params: SGLang sampling overrides.
        evaluation: If True, eval-only.

    Returns:
        ``list[Sample]`` with logprobs + scalar reward.
    """
    import asyncio

    metadata = sample.metadata or {}
    available_tools = metadata.get("tools", [])
    task_id = metadata.get("task_id", "unknown")
    t0 = time.time()

    try:
        async with asyncio.timeout(CONFIG["rollout_guard_sec"]):
            # 1. Tokenize
            router_ip = getattr(args, "sglang_router_ip", "127.0.0.1")
            router_port = getattr(args, "sglang_router_port", 30000)

            prompt_text = (
                sample.prompt if isinstance(sample.prompt, str)
                else str(sample.prompt)
            )

            tokenizer = getattr(args, "tokenizer", None)
            if sample.tokens:
                input_ids = list(sample.tokens)
            elif tokenizer:
                input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            else:
                from slime.utils.processing_utils import load_tokenizer
                tok = load_tokenizer(
                    getattr(args, "hf_checkpoint", ""),
                    trust_remote_code=True,
                )
                input_ids = tok.encode(prompt_text, add_special_tokens=False)

            # 2. Single-turn SGLang generate
            max_resp = getattr(args, "rollout_max_response_len", 4096)
            max_ctx = getattr(args, "rollout_max_context_len", 32768)
            remaining = min(max_resp, max_ctx - len(input_ids))

            sampling = {**(sampling_params or {})}
            sampling["max_new_tokens"] = min(
                sampling.get("max_new_tokens", remaining), remaining,
            )
            # Qwen uses <|im_end|> as EOS — SGLang will stop on it
            if "stop" not in sampling:
                sampling["stop"] = ["<|im_end|>"]

            resp = await _call_sglang(
                router_ip, router_port,
                input_ids=input_ids,
                sampling_params=sampling,
                return_logprob=True,
            )

            output_ids = resp.get("output_ids", [])
            output_text = resp.get("text", "")
            logprobs = resp.get("logprobs", [])
            response_len = len(output_ids) if output_ids else max(len(output_text), 1)

            # 3. Build trajectory
            trajectory = _response_to_trajectory(output_text)

            # 4. Compute 4-dim reward
            ground_truth_label = sample.label or ""

            from examples.agentic_rl_grpo.tool_rl_reward import (
                compute_tool_rl_reward,
            )

            breakdown = await compute_tool_rl_reward(
                args, trajectory, prompt_text,
                available_tools=available_tools,
                ground_truth_label=ground_truth_label,
            )
            reward = breakdown.total

            # 5. Build output Sample
            full_tokens = list(input_ids) + (output_ids if output_ids else [])
            full_logprobs = [0.0] * len(input_ids) + (
                logprobs if logprobs else [0.0] * response_len
            )
            full_loss_mask = [0] * len(input_ids) + [2] * response_len

            result = Sample(
                index=sample.index,
                group_index=sample.group_index,
                rollout_id=getattr(sample, "rollout_id", None),
                prompt=prompt_text,
                tokens=full_tokens,
                response=output_text,
                response_length=response_len,
                loss_mask=full_loss_mask,
                rollout_log_probs=full_logprobs,
                reward=reward,
                status="completed",
                metadata={
                    **(metadata),
                    "reward_breakdown": breakdown.to_dict(),
                    "reward_details": breakdown.details,
                    "trajectory": trajectory,
                    "task_id": task_id,
                },
            )

            logger.info(
                "[tool_rl] %s: reward=%.3f planning=%.1f format=%.3f "
                "tool_call=%.3f halluc=%.0f len=%d %.1fs",
                task_id, reward, breakdown.planning,
                breakdown.format_compliance, breakdown.tool_call_format,
                breakdown.hallucination, response_len, time.time() - t0,
            )

            if evaluation:
                result.remove_sample = True
            return [result]

    except asyncio.TimeoutError:
        logger.warning("[tool_rl] %s: timeout %.1fs", task_id, time.time() - t0)
        return _abort(sample, "timeout", task_id)
    except Exception:
        logger.warning("[tool_rl] %s: failed\n%s", task_id, traceback.format_exc())
        return _abort(sample, f"err:{traceback.format_exc()[:200]}", task_id)


def _abort(sample: Sample, reason: str, task_id: str) -> list[Sample]:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason, "task_id": task_id}
    return [sample]
