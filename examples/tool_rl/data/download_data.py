#!/usr/bin/env python3
"""Download and prepare tool-use datasets for Qwen3-4B GRPO training.

Downloads APIGen, ToolACE, Hammer, BFCL and converts to slime JSONL format
compatible with Qwen3-4B's chat template and ``--apply-chat-template``.

Output Format (slime JSONL)
---------------------------
.. code-block:: json

    {
      "messages": [
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": "What's the weather in Beijing?"}
      ],
      "tools": [{"name": "get_weather", "description": "...", "parameters": {...}}],
      "label": "Ground truth reference",
      "metadata": {...}
    }

- ``messages`` → ``--input-key messages``, chat template wraps as Qwen format.
- ``tools`` → ``--tool-key tools``, injected into chat template for tool defs.
- The instruction is prepended to the FIRST user message.
- Multi-turn conversations are split into single-turn samples.

Usage
-----
.. code-block:: bash

    python examples/agentic_rl_grpo/download_tool_data.py -o ./data/tool_rl
    python examples/agentic_rl_grpo/download_tool_data.py -o ./data/tool_rl --max-samples 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Instruction — prepended to first user message
# ============================================================================

_INSTRUCTION = (
    "At no point should you assume any information about location, date, "
    "or any other details. Stay humble and honest. "
    "The entire task can be solved through multiple rounds of dialogue, "
    "gathering detailed information step by step — "
    "there is no need to solve everything in one go."
)

_DEFAULT_MAX = 5000
_SEED = 42


# ============================================================================
# Helpers
# ============================================================================

def _normalize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize tool definitions to standard format."""
    result = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        name = t.get("name", "")
        if not name:
            continue
        params = t.get("parameters", {})
        # Qwen chat template expects JSON Schema properties
        if isinstance(params, dict) and "properties" not in params:
            params = {"type": "object", "properties": params}
        result.append({
            "name": name,
            "description": t.get("description", ""),
            "parameters": params,
        })
    return result


def _format_gt(answers: list[dict[str, Any]]) -> str:
    """Format ground truth as readable string."""
    if not answers:
        return ""
    lines = []
    for a in answers:
        name = a.get("name", "")
        args = a.get("arguments", {}) or {}
        if isinstance(args, dict):
            args_str = json.dumps(args, ensure_ascii=False)
        else:
            args_str = str(args)
        lines.append(f"  {name}({args_str})")
    return "Ground truth:\n" + "\n".join(lines)


def _make_meta(source: str, task_id: str, tools: list, gt: Any, **extra) -> dict:
    return {
        "benchmark": "tool_rl",
        "source": source,
        "task_id": task_id,
        "ground_truth": gt if gt else None,
        "has_ground_truth": bool(gt),
        "tools": _normalize_tools(tools),
        "max_turns": 1,
        "workdir": "/home/agent",
        **extra,
    }


def _prepend_instruction(messages: list[dict]) -> list[dict]:
    """Prepend instruction to the first user message in the conversation."""
    for msg in messages:
        if msg.get("role") == "user":
            msg["content"] = (
                f"<instruction>\n{_INSTRUCTION}\n</instruction>\n\n"
                + msg["content"]
            )
            break
    return messages


# ============================================================================
# APIGen loader
# ============================================================================

def load_apigen(max_samples: int) -> list[dict[str, Any]]:
    """Load APIGen — single JSON file via hf_hub_download."""
    logger.info("Loading APIGen (Salesforce/xlam-function-calling-60k)...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("pip install huggingface_hub")
        return []

    try:
        path = hf_hub_download(
            "Salesforce/xlam-function-calling-60k",
            "xlam_function_calling_60k.json",
            repo_type="dataset",
        )
    except Exception as e:
        logger.warning("APIGen download failed: %s", e)
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    for sample in data:
        if len(tasks) >= max_samples:
            break
        query = sample.get("query", "")
        if not query:
            continue

        tools_raw = sample.get("tools", "[]")
        answers_raw = sample.get("answers", "[]")
        try:
            tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
        except json.JSONDecodeError:
            tools = []
        try:
            answers = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
        except json.JSONDecodeError:
            answers = []

        messages = _prepend_instruction([
            {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when needed to answer user queries accurately."},
            {"role": "user", "content": query},
        ])

        tasks.append({
            "messages": messages,
            "tools": _normalize_tools(tools),
            "label": _format_gt(answers),
            "metadata": _make_meta("apigen", f"apigen-{sample.get('id', '?')}", tools, answers),
        })

    logger.info("APIGen: %d samples", len(tasks))
    return tasks


# ============================================================================
# ToolACE loader (multi-turn → split into single-turn)
# ============================================================================

def load_toolace(max_samples: int) -> list[dict[str, Any]]:
    """Load ToolACE — split multi-turn conversations into single-turn samples."""
    logger.info("Loading ToolACE (Team-ACE/ToolACE)...")
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets")
        return []

    try:
        ds = load_dataset("Team-ACE/ToolACE", split="train")
    except Exception as e:
        logger.warning("ToolACE: %s", e)
        return []

    tasks = []
    for i, sample in enumerate(ds):
        if len(tasks) >= max_samples:
            break
        system = sample.get("system", "")
        conversations = sample.get("conversations", [])
        if not conversations:
            continue

        tools = _extract_tools_from_text(system)
        history: list[dict] = []

        for ti, turn in enumerate(conversations):
            if not isinstance(turn, dict):
                continue
            role = turn.get("from", "")
            value = str(turn.get("value", ""))

            if role == "user":
                messages = [{"role": "system", "content": system}]
                for h in history[-8:]:
                    messages.append(dict(h))
                messages.append({"role": "user", "content": value})
                messages = _prepend_instruction(messages)

                assistant_resp = _find_next_assistant(conversations, ti)
                gt_calls = _parse_qwen_tool_calls(assistant_resp)

                tasks.append({
                    "messages": messages,
                    "tools": _normalize_tools(tools),
                    "label": _format_gt(gt_calls) + (
                        f"\nReference:\n{assistant_resp[:1000]}"
                        if assistant_resp else ""
                    ),
                    "metadata": _make_meta(
                        "toolace", f"toolace-{i}-t{ti}", tools, gt_calls,
                        conversation_turn=ti,
                    ),
                })
                history.append({"role": "user", "content": value})

            elif role == "assistant":
                history.append({"role": "assistant", "content": value[:800]})
            elif role == "tool":
                history.append({"role": "tool", "content": value[:500]})

            if len(tasks) >= max_samples:
                break

    logger.info("ToolACE: %d single-turn samples", len(tasks))
    return tasks


def _find_next_assistant(conversations: list, idx: int) -> str:
    for j in range(idx + 1, len(conversations)):
        t = conversations[j]
        if isinstance(t, dict) and t.get("from") == "assistant":
            return str(t.get("value", ""))
    return ""


# ============================================================================
# Hammer loader
# ============================================================================

def load_hammer(max_samples: int) -> list[dict[str, Any]]:
    """Load Hammer irrelevance data."""
    logger.info("Loading Hammer (MadeAgents/xlam-irrelevance-7.5k)...")
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets")
        return []

    try:
        ds = load_dataset("MadeAgents/xlam-irrelevance-7.5k", split="train")
    except Exception as e:
        logger.warning("Hammer: %s", e)
        return []

    tasks = []
    for i, sample in enumerate(ds):
        if len(tasks) >= max_samples:
            break
        query = sample.get("query", "")
        if not query:
            continue
        tools_raw = sample.get("tools", "[]")
        answers_raw = sample.get("answers", "[]")
        try:
            tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
        except json.JSONDecodeError:
            tools = []
        try:
            answers = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
        except json.JSONDecodeError:
            answers = []

        messages = _prepend_instruction([
            {"role": "system", "content": "You are a helpful assistant. Determine if tools are needed for the user's request."},
            {"role": "user", "content": query},
        ])

        tasks.append({
            "messages": messages,
            "tools": _normalize_tools(tools),
            "label": _format_gt(answers),
            "metadata": _make_meta("hammer", f"hammer-{i}", tools, answers,
                                   is_irrelevant=not bool(answers)),
        })

    logger.info("Hammer: %d samples", len(tasks))
    return tasks


# ============================================================================
# BFCL loader
# ============================================================================

def load_bfcl(max_samples: int) -> list[dict[str, Any]]:
    """Load BFCL — split multi-turn, keep single-turn."""
    logger.info("Loading BFCL (gorilla-llm/Berkeley-Function-Calling-Leaderboard)...")
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        logger.error("pip install huggingface_hub")
        return []

    try:
        files = list_repo_files(
            "gorilla-llm/Berkeley-Function-Calling-Leaderboard", repo_type="dataset",
        )
    except Exception as e:
        logger.warning("BFCL: %s", e)
        return []

    json_files = [f for f in files if f.endswith(".json")]
    priority = ["simple", "multiple", "parallel", "multi_turn"]
    json_files.sort(key=lambda f: (not any(p in f.lower() for p in priority), f))

    tasks = []
    for jf in json_files:
        if len(tasks) >= max_samples:
            break
        try:
            path = hf_hub_download(
                "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                jf, repo_type="dataset",
            )
        except Exception:
            continue
        category = jf.replace(".json", "").replace("BFCL_v3_", "")
        with open(path) as f:
            for line in f:
                if len(tasks) >= max_samples:
                    break
                try:
                    raw = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                tasks.extend(_parse_bfcl(raw, category))

    logger.info("BFCL: %d samples", len(tasks))
    return tasks


def _extract_bfcl_turns(question: Any) -> list[list[dict]] | None:
    """Extract turns from BFCL v3 question field.

    BFCL v3 format: ``[[{"role": "user", "content": "..."}], ...]``
    Each outer element is a turn with one or more messages.
    Returns list of messages per turn (list of lists of dicts).
    """
    if isinstance(question, list) and len(question) > 0:
        # BFCL v3: list of turns, each turn is a list of messages
        if isinstance(question[0], list):
            return question
        # Some files have list of dicts (single turn)
        if isinstance(question[0], dict) and "role" in question[0]:
            return [question]
    return None


def _extract_text_from_bfcl_messages(messages: list) -> str:
    """Extract user text from BFCL message list."""
    parts = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
    return "\n".join(parts) if parts else ""


def _parse_bfcl(raw: dict, category: str) -> list[dict]:
    """Parse BFCL sample → single-turn samples."""
    funcs = raw.get("function") or raw.get("functions") or []
    if isinstance(funcs, dict):
        funcs = [funcs]
    tools = _normalize_tools(funcs)
    tid = raw.get("id", f"bfcl-{category}")

    # Skip auxiliary files
    if category.startswith("possible_answer/") or category.startswith("multi_turn_func_doc/"):
        return []

    turns_data = _extract_bfcl_turns(raw.get("question"))
    if turns_data is None:
        return []

    results = []
    history: list[dict] = []

    for ti, turn_msgs in enumerate(turns_data):
        if not isinstance(turn_msgs, list):
            continue
        query_text = _extract_text_from_bfcl_messages(turn_msgs)
        if not query_text:
            continue

        msgs = [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
        ]
        for h in history[-8:]:
            msgs.append(dict(h))
        msgs.append({"role": "user", "content": query_text})
        msgs = _prepend_instruction(msgs)

        # Parse ground truth for this turn
        gt_raw = raw.get("ground_truth") or raw.get("answers") or raw.get("answer") or ""
        if isinstance(gt_raw, list):
            gt_str = "\n".join(str(g) for g in gt_raw if g) if gt_raw else ""
        elif isinstance(gt_raw, str):
            gt_str = gt_raw
        else:
            gt_str = str(gt_raw) if gt_raw else ""
        if gt_str.strip():
            gt = _parse_qwen_tool_calls(gt_str)
        else:
            gt = []

        results.append({
            "messages": msgs,
            "tools": tools,
            "label": _format_gt(gt) + (f"\nReference:\n{gt_str[:800]}" if gt_str.strip() else ""),
            "metadata": _make_meta(
                f"bfcl/{category}", f"{tid}-t{ti}", tools, gt,
                bfcl_category=category,
            ),
        })

        # Add to history for multi-turn context
        history.append({"role": "user", "content": query_text})
        # Look for assistant response in the same turn
        for msg in turn_msgs:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    history.append({"role": "assistant", "content": content[:500]})

    return results


# ============================================================================
# Tool extraction
# ============================================================================

def _extract_tools_from_text(text: str) -> list[dict[str, Any]]:
    """Extract tool definitions from system prompt text."""
    tools = []
    # Try to find the outermost JSON array of tools in the text
    array_match = re.search(r'\[.*\]', text, re.DOTALL)
    if array_match:
        try:
            candidates = json.loads(array_match.group(0))
            if isinstance(candidates, list):
                for c in candidates:
                    if isinstance(c, dict) and "name" in c:
                        tools.append(c)
        except (json.JSONDecodeError, TypeError):
            pass
    # Fallback: try extracting individual objects with nested braces
    if not tools:
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    block = text[start:i+1]
                    try:
                        obj = json.loads(block)
                        if isinstance(obj, dict) and "name" in obj:
                            tools.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1
    return tools


def _parse_qwen_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse Qwen XML tool calls from assistant response.

    Format::

        <tool_call>
        <function=name>
        <parameter=param>
        value
        </parameter>
        </function>
        </tool_call>
    """
    calls = []
    for tc_match in re.finditer(
        r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL | re.IGNORECASE,
    ):
        block = tc_match.group(1)
        # Parse function name
        func_match = re.search(r"<function=(\w[\w.]*)>", block)
        if not func_match:
            continue
        func_name = func_match.group(1)

        # Parse parameters
        args = {}
        for pm in re.finditer(
            r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", block, re.DOTALL,
        ):
            pname = pm.group(1)
            pval = pm.group(2).strip()
            # Try JSON parse for structured values
            try:
                pval = json.loads(pval)
            except (json.JSONDecodeError, TypeError):
                pass
            args[pname] = pval

        calls.append({"name": func_name, "arguments": args})

    # Fallback: JSON tool calls
    if not calls:
        for m in re.finditer(
            r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}\s*\}',
            text, re.DOTALL,
        ):
            try:
                obj = json.loads(m.group(0))
                if "name" in obj:
                    calls.append(obj)
            except json.JSONDecodeError:
                pass

    return calls


# ============================================================================
# Validation
# ============================================================================

def validate_tasks(tasks: list) -> list:
    valid = []
    for t in tasks:
        msgs = t.get("messages", [])
        if not msgs:
            continue
        tools = t.get("tools", t.get("metadata", {}).get("tools", []))
        if not tools:
            continue
        user_content = next(
            (m["content"] for m in msgs if m.get("role") == "user"), "",
        )
        if len(user_content) > 65536:
            continue
        valid.append(t)
    removed = len(tasks) - len(valid)
    if removed:
        logger.info("Filtered %d invalid (%d remaining)", removed, len(valid))
    return valid


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download tool-use datasets for Qwen3-4B RL")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--max-samples", type=int, default=_DEFAULT_MAX)
    parser.add_argument("--seed", type=int, default=_SEED)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = (["apigen", "toolace", "hammer", "bfcl"] if args.datasets == "all"
             else [n.strip() for n in args.datasets.split(",")])

    loaders = {"apigen": load_apigen, "toolace": load_toolace,
               "hammer": load_hammer, "bfcl": load_bfcl}

    all_tasks = []
    for name in names:
        if name not in loaders:
            logger.warning("Unknown %r, available: %s", name, sorted(loaders))
            continue
        tasks = loaders[name](args.max_samples)
        tasks = validate_tasks(tasks)
        if tasks:
            rng = random.Random(args.seed)
            rng.shuffle(tasks)
            out = output_dir / f"{name}_tool_rl.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for t in tasks:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            logger.info("Wrote %d → %s", len(tasks), out)
            all_tasks.extend(tasks)
        else:
            logger.warning("No samples for %s", name)

    if not all_tasks:
        logger.error("No datasets loaded.")
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(all_tasks)
    merged = output_dir / "mixed_tool_rl.jsonl"
    with open(merged, "w", encoding="utf-8") as f:
        for t in all_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    logger.info("Done! %d total → %s", len(all_tasks), merged)
    for src in sorted(set(t["metadata"]["source"] for t in all_tasks)):
        logger.info("  %s: %d", src, sum(1 for t in all_tasks if t["metadata"]["source"] == src))


if __name__ == "__main__":
    main()
