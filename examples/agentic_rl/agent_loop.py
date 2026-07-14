"""Multi-turn agent loop: SGLang generate ↔ tool execution in sandbox.

The core loop:
  1. Construct the next turn's prompt (system prompt + conversation history)
  2. Call SGLang ``/generate`` with ``return_logprob=True``
  3. Parse the response — extract text, tool calls, finish reason
  4. If tool call → execute in sandbox, append observation, go to 1
  5. If finish (``stop`` / ``length``) → break, return trajectory
"""

from __future__ import annotations

import json
import logging
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
    """Call the SGLang router's ``/generate`` endpoint with token-level input.

    Returns a dict with keys:
      - ``output_ids`` (list[int])
      - ``text`` (str)
      - ``logprobs`` (list[float])  — only when ``return_logprob=True``
      - ``finish_reason`` (str)     — ``"stop"`` | ``"length"`` | ``"abort"``
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
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            resp.raise_for_status()
            result = await resp.json()

    meta = result.get("meta_info") or {}
    finish_reason = meta.get("finish_reason", {})

    output = {
        "output_ids": result.get("output_ids", []),
        "text": result.get("text", ""),
        "finish_reason": finish_reason.get("type", "stop") if isinstance(finish_reason, dict) else str(finish_reason),
        "meta_info": meta,
    }

    if return_logprob:
        # output_token_logprobs is a list of [logprob, token_id, token_str, ...]
        raw_logprobs = meta.get("output_token_logprobs", [])
        output["logprobs"] = [float(lp[0]) if isinstance(lp, (list, tuple)) else float(lp) for lp in raw_logprobs]
    else:
        output["logprobs"] = []

    return output


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool call JSON from the model output.

    Supports multiple formats:
      - ```json { "name": "...", "arguments": {...} } ```  (code fence)
      - ``<tool_call>{"name":"...","arguments":{...}}</tool_call>`` (XML)
      - Bare JSON in a line starting with ``Action:`` or ``{"name":``

    Returns a list of ``{"name": str, "arguments": dict}`` dicts.
    """
    import re

    calls = []

    # Format 1: code-fenced JSON
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        try:
            obj = json.loads(match.group(1))
            if "name" in obj:
                calls.append(obj)
        except json.JSONDecodeError:
            pass

    # Format 2: XML-style <tool_call>
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            obj = json.loads(match.group(1))
            if "name" in obj:
                calls.append(obj)
        except json.JSONDecodeError:
            pass

    # Format 3: Action: prefix + JSON
    for match in re.finditer(r"Action:\s*(\{.*?\})", text, re.DOTALL):
        try:
            obj = json.loads(match.group(1))
            if "name" in obj:
                calls.append(obj)
        except json.JSONDecodeError:
            pass

    return calls


async def execute_tool_call(
    call: dict[str, Any],
    sandbox: Any,
    workdir: str = "/home/agent",
) -> str:
    """Execute a single tool call in the sandbox and return the observation."""
    name = call.get("name", "")
    arguments = call.get("arguments", {}) or {}

    # Common tool types
    if name in ("bash", "shell", "execute_command", "run"):
        cmd = arguments.get("command") or arguments.get("cmd", "")
        ec, stdout, stderr = await sandbox.exec(
            cmd, user="agent", check=False, timeout=120,
        )
        result = stdout or ""
        if stderr:
            result += f"\n[stderr]\n{stderr}"
        result = result[-4096:]  # truncate

    elif name in ("python", "execute_python", "ipython"):
        code = arguments.get("code") or arguments.get("python", "")
        # Write to temp file and execute
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=workdir)
        try:
            tmp.write(code)
            tmp.flush()
            ec, stdout, stderr = await sandbox.exec(
                f"cd {workdir} && python3 {tmp.name}",
                user="agent", check=False, timeout=120,
            )
            result = stdout or ""
            if stderr:
                result += f"\n[stderr]\n{stderr}"
            result = result[-4096:]
        finally:
            tmp.close()

    elif name in ("write_file", "file_write", "edit"):
        path = arguments.get("path", "")
        content = arguments.get("content", "")
        await sandbox.exec(
            f"mkdir -p {workdir}/{os.path.dirname(path)}",
            user="agent", check=False, timeout=10,
        )
        # Write via sandbox (base64 to avoid shell quoting issues)
        import base64
        encoded = base64.b64encode(content.encode() if isinstance(content, str) else content).decode()
        await sandbox.exec(
            f"echo {encoded} | base64 -d > {workdir}/{path}",
            user="agent", check=False, timeout=30,
        )
        result = f"Wrote {len(content)} bytes to {path}"

    elif name in ("read_file", "file_read"):
        path = arguments.get("path", "")
        _, content, _ = await sandbox.exec(
            f"cat {workdir}/{path}",
            user="agent", check=False, timeout=10,
        )
        result = content or ""

    elif name in ("finish", "answer", "submit"):
        result = arguments.get("answer", "") or arguments.get("output", "")
        # Signal the loop to stop
        result = f"[FINISH] {result}"

    else:
        logger.warning("Unknown tool call: %s with args %s", name, arguments)
        result = f"[Error] Unknown tool: {name}"

    return result


# ---------------------------------------------------------------------------
# Multi-turn agent loop
# ---------------------------------------------------------------------------

async def run_agent_loop(
    args: Any,
    sample: Sample,
    sandbox: Any,
    sampling_params: dict[str, Any],
    *,
    max_turns: int = 10,
    workdir: str = "/home/agent",
) -> tuple[list[dict[str, Any]], list[Sample]]:
    """Multi-turn agent loop.

    Args:
        args: slime training args (must have ``sglang_router_ip``, ``sglang_router_port``).
        sample: Input sample with ``prompt`` and optionally ``metadata``.
        sandbox: Sandbox instance for tool execution.
        sampling_params: SGLang sampling overrides.
        max_turns: Maximum turns before forcing stop.
        workdir: Working directory inside the sandbox.

    Returns:
        ``(trajectory, samples)`` where:
        - ``trajectory`` is the full conversation ``list[dict]`` (for logging / eval).
        - ``samples`` is ``list[Sample]`` containing one flattened trajectory
          per turn with loss masks.
    """
    router_ip = getattr(args, "sglang_router_ip", "127.0.0.1")
    router_port = getattr(args, "sglang_router_port", 30000)

    # Build system prompt + conversation history
    prompt_text = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)

    # Tokenized prompt (pre-computed or tokenize now)
    tokenizer = getattr(args, "tokenizer", None)
    if sample.tokens:
        all_input_ids = list(sample.tokens)
    elif tokenizer:
        from slime.utils.processing_utils import load_tokenizer
        tok = tokenizer if tokenizer else load_tokenizer(args.hf_checkpoint)
        all_input_ids = tok.encode(prompt_text, add_special_tokens=False)
    else:
        all_input_ids = []

    # Track tokens, logprobs, and loss masks
    trajectory = []  # list of turn dicts
    segment_samples = []
    accumulated_tokens = list(all_input_ids)
    accumulated_log_probs: list[float] = [0.0] * len(all_input_ids)
    accumulated_loss_mask: list[int] = [0] * len(all_input_ids)

    for turn_idx in range(max_turns):
        # 1. Calculate remaining context budget
        max_context = getattr(args, "rollout_max_context_len", 32768)
        max_response = getattr(args, "rollout_max_response_len", 4096)
        remaining_budget = min(max_response, max_context - len(accumulated_tokens))
        if remaining_budget <= 0:
            logger.warning("Context budget exhausted at turn %d", turn_idx)
            break

        # 2. SGLang generate
        turn_sampling = {**(sampling_params or {})}
        turn_sampling["max_new_tokens"] = min(
            turn_sampling.get("max_new_tokens", remaining_budget),
            remaining_budget,
        )
        if "stop" not in turn_sampling:
            turn_sampling["stop"] = ["<|eot|>", "</s>", "<|im_end|>"]

        resp = await call_sglang(
            router_ip, router_port,
            input_ids=accumulated_tokens,
            sampling_params=turn_sampling,
            return_logprob=True,
        )

        output_ids = resp.get("output_ids", [])
        output_text = resp.get("text", "")
        logprobs = resp.get("logprobs", [])
        finish_reason = resp.get("finish_reason", "stop")

        # 3. Record this turn in trajectory
        turn_record = {
            "turn": turn_idx,
            "prompt_length": len(accumulated_tokens),
            "output_ids": output_ids,
            "text": output_text,
            "logprobs": logprobs,
            "finish_reason": finish_reason,
        }
        trajectory.append(turn_record)

        # 4. Append to accumulated tensors
        response_len = len(output_ids)
        accumulated_tokens.extend(output_ids)
        accumulated_log_probs.extend(logprobs)
        accumulated_loss_mask.extend([2 if turn_idx == 0 else 1] * response_len)

        # 5. Create a segment Sample for each turn
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

        # 6. Check finish
        if finish_reason == "stop":
            # Check for finish/answer tool call in the text
            finish_marker = "[FINISH]"
            if finish_marker in output_text:
                break
            # Check for tool calls
            tool_calls = parse_tool_calls(output_text)
            if not tool_calls:
                # No tool call and not stopped for a special reason — assume answer
                break

            # 7. Execute tool calls
            for tc in tool_calls:
                if tc.get("name") in ("finish", "answer", "submit"):
                    break
                observation = await execute_tool_call(tc, sandbox, workdir)

                if observation.startswith("[FINISH]"):
                    break

                # Append observation tokens as non-trainable context
                obs_text = f"\n<observation>\n{observation}\n</observation>\n"
                if tokenizer:
                    obs_ids = tokenizer.encode(obs_text, add_special_tokens=False)
                else:
                    obs_ids = list(obs_text.encode("utf-8"))

                accumulated_tokens.extend(obs_ids)
                accumulated_log_probs.extend([0.0] * len(obs_ids))
                accumulated_loss_mask.extend([0] * len(obs_ids))

                trajectory.append({
                    "turn": turn_idx,
                    "type": "observation",
                    "text": observation,
                    "tool_call": tc,
                })
        elif finish_reason == "length":
            # Hit max tokens — could still be mid-response
            if turn_idx < max_turns - 1:
                continue
            break
        else:
            # "abort" or error
            logger.warning("Turn %d finished with reason: %s", turn_idx, finish_reason)
            break

    return trajectory, segment_samples


import os  # noqa: E402 (needed for execute_tool_call)
