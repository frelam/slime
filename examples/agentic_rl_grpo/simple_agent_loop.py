"""Simple multi-turn agent loop for text-based environments (ALFWorld, WebShop, etc.).

与 ``examples/agentic_rl/agent_loop.py`` 不同，这个 agent loop 不要求模型输出
结构化的 tool call JSON。模型直接输出文本，由环境（dataset adapter）解析并执行。

Supports two interaction modes:
  1. **text mode**: 模型输出纯文本 action → 环境返回 observation → 循环
  2. **tool mode**: 模型输出 ``{"name": "env", "arguments": {"action": "..."}}``
     → 环境执行 action → 返回 observation

两种模式由 dataset adapter 的 ``interaction_mode`` 属性决定。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SGLang client helpers
# ---------------------------------------------------------------------------

async def call_sglang(
    router_ip: str,
    router_port: int,
    input_ids: list[int],
    sampling_params: dict[str, Any] | None = None,
    return_logprob: bool = True,
) -> dict[str, Any]:
    """Call the SGLang router's ``/generate`` endpoint.

    Returns a dict with keys:
      - ``output_ids`` (list[int])
      - ``text`` (str)
      - ``logprobs`` (list[float])
      - ``finish_reason`` (str)
      - ``meta_info`` (dict)
    """
    import aiohttp

    url = f"http://{router_ip}:{router_port}/generate"
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": sampling_params or {},
        "return_logprob": return_logprob,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()

    meta = result.get("meta_info") or {}
    finish_reason = meta.get("finish_reason", {})

    output: dict[str, Any] = {
        "output_ids": result.get("output_ids", []),
        "text": result.get("text", ""),
        "finish_reason": (
            finish_reason.get("type", "stop")
            if isinstance(finish_reason, dict)
            else str(finish_reason)
        ),
        "meta_info": meta,
    }

    if return_logprob:
        raw_logprobs = meta.get("output_token_logprobs", [])
        output["logprobs"] = [
            float(lp[0]) if isinstance(lp, (list, tuple)) else float(lp)
            for lp in raw_logprobs
        ]
    else:
        output["logprobs"] = []

    return output


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

# 匹配 tool call JSON: {"name": "env", "arguments": {"action": "..."}}
_TOOL_CALL_RE = re.compile(
    r'\{[^{}]*"name"\s*:\s*"(?:env|alfworld|webshop|finish|answer|submit)"[^{}]*\}',
    re.DOTALL,
)


def extract_action(text: str) -> str:
    """Extract the agent's action from response text.

    Priority:
      1. Tool call JSON ``{"name": "env", "arguments": {"action": "..."}}``
      2. Tool call JSON ``{"name": "finish", "arguments": {"answer": "..."}}``
      3. ``<action>...</action>`` XML tag
      4. Last non-empty line of text (text-mode environment)
    """
    # Try tool call format first
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(0))
            name = obj.get("name", "")
            args = obj.get("arguments", {}) or {}
            if name in ("env", "alfworld", "webshop"):
                return args.get("action", "") or args.get("command", "")
            elif name in ("finish", "answer", "submit"):
                answer = args.get("answer", "") or args.get("output", "")
                return f"__FINISH__{answer}"
        except json.JSONDecodeError:
            pass

    # Try XML format
    xml_match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if xml_match:
        return xml_match.group(1).strip()

    # Fallback: return the last non-empty line as the raw action
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return lines[-1]

    return text.strip()


def extract_finish_answer(text: str) -> str | None:
    """Extract final answer from a finish/answer/submit response.

    Returns the answer string, or None if no finish signal found.
    """
    # Check for explicit finish markers
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(0))
            if obj.get("name", "") in ("finish", "answer", "submit"):
                args = obj.get("arguments", {}) or {}
                return args.get("answer", "") or args.get("output", "") or ""
        except json.JSONDecodeError:
            pass

    # Check for text markers
    for marker in ["[FINISH]", "DONE:", "ANSWER:", "Final answer:"]:
        if marker in text:
            idx = text.index(marker)
            return text[idx + len(marker):].strip()

    return None


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


async def run_simple_agent_loop(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    dataset_adapter: Any,
    metadata: dict[str, Any],
    *,
    max_turns: int = 10,
    workdir: str = "/home/agent",
) -> tuple[list[dict[str, Any]], list[Sample]]:
    """Simple multi-turn agent loop for text-based environments.

    模型输出文本（action 或 tool call），环境执行并返回 observation。
    支持 dataset adapter 的 ``env_step()`` 和 ``env_done()`` 方法。

    Args:
        args: slime training args.
        sample: Input sample.
        sampling_params: SGLang sampling overrides.
        dataset_adapter: The dataset adapter managing the environment.
        metadata: Task metadata.
        max_turns: Maximum turns before forcing stop.
        workdir: Working directory inside the sandbox.

    Returns:
        ``(trajectory, samples)`` where:
        - ``trajectory`` is the full conversation history.
        - ``samples`` is ``list[Sample]`` for GRPO training.
    """
    router_ip = getattr(args, "sglang_router_ip", "127.0.0.1")
    router_port = getattr(args, "sglang_router_port", 30000)

    prompt_text = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)

    # Tokenize prompt
    tokenizer = getattr(args, "tokenizer", None)
    if sample.tokens:
        all_input_ids = list(sample.tokens)
    elif tokenizer:
        all_input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    else:
        from slime.utils.processing_utils import load_tokenizer

        tok = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        all_input_ids = tok.encode(prompt_text, add_special_tokens=False)

    trajectory: list[dict[str, Any]] = []
    segment_samples: list[Sample] = []
    accumulated_tokens = list(all_input_ids)
    accumulated_log_probs: list[float] = [0.0] * len(all_input_ids)
    accumulated_loss_mask: list[int] = [0] * len(all_input_ids)

    # Current environment observation (set by dataset adapter during setup)
    current_obs = metadata.get("initial_observation", "")

    for turn_idx in range(max_turns):
        # 1. Build prompt with current observation
        if turn_idx == 0 and current_obs:
            obs_prefix = _format_observation_prompt(current_obs)
            if tokenizer:
                obs_ids = tokenizer.encode(obs_prefix, add_special_tokens=False)
            else:
                obs_ids = list(obs_prefix.encode("utf-8"))
            accumulated_tokens.extend(obs_ids)
            accumulated_log_probs.extend([0.0] * len(obs_ids))
            accumulated_loss_mask.extend([0] * len(obs_ids))

        # 2. Calculate remaining context budget
        max_context = getattr(args, "rollout_max_context_len", 32768)
        max_response = getattr(args, "rollout_max_response_len", 4096)
        remaining_budget = min(max_response, max_context - len(accumulated_tokens))
        if remaining_budget <= 0:
            logger.warning("Context budget exhausted at turn %d", turn_idx)
            break

        # 3. SGLang generate
        turn_sampling = {**(sampling_params or {})}
        turn_sampling["max_new_tokens"] = min(
            turn_sampling.get("max_new_tokens", remaining_budget),
            remaining_budget,
        )
        if "stop" not in turn_sampling:
            turn_sampling["stop"] = ["<|eot|>", "</s>", "<|im_end|>", "\nObservation:"]

        resp = await call_sglang(
            router_ip,
            router_port,
            input_ids=accumulated_tokens,
            sampling_params=turn_sampling,
            return_logprob=True,
        )

        output_ids = resp.get("output_ids", [])
        output_text = resp.get("text", "")
        logprobs = resp.get("logprobs", [])
        finish_reason = resp.get("finish_reason", "stop")

        # 4. Record turn
        turn_record = {
            "turn": turn_idx,
            "text": output_text,
            "logprobs": logprobs,
            "finish_reason": finish_reason,
        }
        trajectory.append(turn_record)

        # 5. Append to accumulated tensors
        response_len = len(output_ids)
        accumulated_tokens.extend(output_ids)
        accumulated_log_probs.extend(logprobs)
        accumulated_loss_mask.extend([2 if turn_idx == 0 else 1] * response_len)

        # 6. Create segment Sample
        segment = Sample(
            index=sample.index,
            group_index=sample.group_index,
            rollout_id=getattr(sample, "rollout_id", None),
            prompt=prompt_text,
            tokens=list(accumulated_tokens),
            response=output_text,
            response_length=response_len,
            loss_mask=list(accumulated_loss_mask),
            rollout_log_probs=list(accumulated_log_probs),
            status="completed",
        )
        segment_samples.append(segment)

        # 7. Parse action and execute in environment
        action = extract_action(output_text)

        # Check for finish signal
        if action.startswith("__FINISH__"):
            answer = action[len("__FINISH__"):]
            trajectory.append({
                "turn": turn_idx,
                "type": "finish",
                "text": answer,
            })
            break

        # Check if model stopped without tool call (natural finish)
        if finish_reason == "stop" and not action:
            finish_answer = extract_finish_answer(output_text)
            if finish_answer is not None:
                trajectory.append({
                    "turn": turn_idx,
                    "type": "finish",
                    "text": finish_answer,
                })
            break

        # 8. Execute action in environment
        if hasattr(dataset_adapter, "env_step"):
            observation, done = await dataset_adapter.env_step(
                action, metadata
            )
        else:
            # Fallback: bash execution (for terminal tasks)
            observation = await _bash_step(action, metadata)
            done = False

        # Record observation
        trajectory.append({
            "turn": turn_idx,
            "type": "observation",
            "text": observation,
            "action": action,
        })

        if done:
            break

        # 9. Append observation as non-trainable context
        obs_text = f"\nObservation: {observation}\n"
        if tokenizer:
            obs_ids = tokenizer.encode(obs_text, add_special_tokens=False)
        else:
            obs_ids = list(obs_text.encode("utf-8"))
        accumulated_tokens.extend(obs_ids)
        accumulated_log_probs.extend([0.0] * len(obs_ids))
        accumulated_loss_mask.extend([0] * len(obs_ids))

    return trajectory, segment_samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_observation_prompt(observation: str) -> str:
    """Format the initial observation as part of the prompt."""
    if not observation:
        return ""
    return f"\nCurrent environment:\n{observation}\n\nWhat is your next action?\n"


async def _bash_step(action: str, metadata: dict[str, Any]) -> tuple[str, bool]:
    """Execute action as a bash command in the current sandbox.

    Used as fallback when dataset adapter doesn't provide env_step().

    Security note: ``action`` originates from the model output and is
    inherently a shell command — this is the intended design for terminal
    agent tasks.  The action runs in an ephemeral subprocess sandbox with
    a minimal environment (PATH + HOME only) and a validated workdir.
    """
    import asyncio
    import os
    import re
    import shlex
    import subprocess
    import tempfile

    workdir = metadata.get("workdir", tempfile.gettempdir())

    # Validate workdir is a safe path (prevent traversal)
    allowed_prefixes = ("/home/agent", "/tmp/", "/home/charles/workspace")
    resolved = os.path.normpath(os.path.realpath(workdir))
    if not any(resolved.startswith(p) for p in allowed_prefixes):
        resolved = tempfile.mkdtemp(prefix="slime_sandbox_")

    # Minimal env: only what a terminal command legitimately needs
    safe_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": resolved,
        "USER": "agent",
        "PWD": resolved,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }

    proc = await asyncio.create_subprocess_shell(
        action,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=resolved,
        env=safe_env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return "Error: Command timed out after 120s", False

    out = (stdout or b"").decode("utf-8", errors="replace")
    err = (stderr or b"").decode("utf-8", errors="replace")
    result = out
    if err:
        result += f"\n[stderr]\n{err}"
    return result[-4096:], False
