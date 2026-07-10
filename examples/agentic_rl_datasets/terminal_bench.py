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
