"""Terminal-Bench dataset adapter.

Terminal-Bench contains shell-based tasks where the model must execute the
correct shell commands to accomplish a goal. Evaluation checks whether the
terminal output matches expected patterns.

Native format (JSONL):
    Each line has:
    - ``prompt`` / ``instruction``: natural language task description
    - ``metadata.setup_commands``: shell commands to prepare the environment
    - ``metadata.expected_output`` or ``metadata.check_command``: eval criteria
    - ``metadata.timeout_sec``: optional per-task timeout (default 120)

Example line::

    {"prompt": "Find all files modified in the last 24 hours under /home",
     "metadata": {"setup_commands": [], "check_command": "ls -la /tmp/checkpoint.txt", "expected_exit_code": 0}}
"""

from __future__ import annotations

import json
import logging
from typing import Any

from examples.agentic_rl_datasets import DatasetAdapter, register_adapter
from slime.agent.sandbox import Sandbox, ensure_agent_user

logger = logging.getLogger(__name__)


@register_adapter
class TerminalBenchAdapter(DatasetAdapter):
    """Adapter for Terminal-Bench shell tasks."""

    name = "terminal_bench"

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        tasks = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                metadata = raw.get("metadata") or {}
                prompt = raw.get("prompt") or raw.get("instruction", "")
                tasks.append({
                    "prompt": prompt,
                    "metadata": {
                        "setup_commands": metadata.get("setup_commands", []),
                        "check_command": metadata.get("check_command", ""),
                        "expected_output": metadata.get("expected_output", ""),
                        "expected_exit_code": metadata.get("expected_exit_code", 0),
                        "timeout_sec": metadata.get("timeout_sec", 120),
                        "task_id": metadata.get("task_id", raw.get("id", str(len(tasks)))),
                    },
                    "label": metadata.get("task_id", str(len(tasks))),
                })
        return tasks

    async def setup_task(self, sb: Sandbox, metadata: dict[str, Any]) -> None:
        await ensure_agent_user(sb, "/home/agent")
        setup_cmds = metadata.get("setup_commands", [])
        for cmd in setup_cmds:
            if cmd.strip():
                await sb.exec(cmd, user="agent", check=False, timeout=60)

    async def evaluate_task(
        self, sb: Sandbox, metadata: dict[str, Any], *, timeout_sec: int = 300
    ) -> float:
        check_cmd = metadata.get("check_command", "")
        expected_output = metadata.get("expected_output", "")
        expected_exit_code = metadata.get("expected_exit_code", 0)

        if not check_cmd and not expected_output:
            logger.warning("[terminal_bench] No check_command or expected_output; reward=0")
            return 0.0

        reward = 1.0

        # Check exit code via check_command
        if check_cmd:
            ec, stdout, stderr = await sb.exec(
                check_cmd, user="agent", check=False, timeout=timeout_sec
            )
            if ec != expected_exit_code:
                reward -= 0.5

        # Check expected output substring if specified
        if expected_output:
            # Re-run or use the stdout from the check command if available
            if check_cmd:
                output = (stdout or "") + (stderr or "")
            else:
                # No check_command — run a simple cat or echo to verify
                output = ""
            if expected_output not in output:
                reward -= 0.5

        return max(0.0, reward)

    async def llm_judge(
        self,
        trajectory: list[dict[str, Any]],
        metadata: dict[str, Any],
        args: Any,
    ) -> float | None:
        from examples.agentic_rl.llm_judge import (
            DEFAULT_JUDGE_SYSTEM_PROMPT,
            build_judge_messages,
            call_llm_judge,
        )

        system_prompt = DEFAULT_JUDGE_SYSTEM_PROMPT + (
            "\n\nFor terminal/command-line tasks, evaluate whether the agent:"
            "\n1. Used correct shell commands to solve the problem"
            "\n2. Handled errors and edge cases"
            "\n3. Produced output matching the expected result"
        )
        task_desc = metadata.get("prompt", "")
        messages = build_judge_messages(system_prompt, task_desc, trajectory)
        max_retries = getattr(args, "llm_judge_max_retries", 2)
        return await call_llm_judge(args, messages, max_retries=max_retries)

    async def analyze_trajectory(
        self,
        trajectory: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract terminal-task trajectory statistics for verifier scoring."""
        failed_commands = []
        total_commands = 0
        exit_codes: list[int] = []

        for record in trajectory:
            text = record.get("text", "")
            if record.get("type") == "observation":
                total_commands += 1
                # Check for command failure indicators
                if _is_terminal_failure(text):
                    failed_commands.append({
                        "text": text[:500],
                        "record_index": record.get("turn", len(failed_commands)),
                    })
                # Try to extract exit code
                ec = _extract_exit_code(text)
                if ec is not None:
                    exit_codes.append(ec)

        return {
            "total_commands": total_commands,
            "failed_commands": len(failed_commands),
            "failed_command_details": failed_commands,
            "exit_codes": exit_codes,
            "all_commands_succeeded": len(failed_commands) == 0,
        }


def _is_terminal_failure(text: str) -> bool:
    """Check if a terminal observation indicates command failure."""
    if not text:
        return False
    markers = [
        "command not found",
        "No such file or directory",
        "Permission denied",
        "exit status 1",
        "exit status 2",
        "[Error]",
        "[ERROR]",
        "Error:",
        "cannot access",
        "not found",
    ]
    return any(m in text for m in markers)


def _extract_exit_code(text: str) -> int | None:
    """Extract exit code from terminal output if present."""
    import re
    match = re.search(r"exit (?:status|code)\s*(\d+)", text)
    if match:
        return int(match.group(1))
    return None
