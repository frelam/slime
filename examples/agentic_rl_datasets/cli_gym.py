"""CLI-Gym dataset adapter.

CLI-Gym provides interactive CLI environments where the model must navigate
command-line tools to accomplish tasks.  Evaluation checks whether the final
state matches expected criteria.

Native format (JSONL):
    Each line has:
    - ``prompt`` / ``instruction``: task description with available commands
    - ``metadata.setup_script``: shell script to bootstrap the CLI environment
    - ``metadata.expected_state``: dict of check keys → expected values
    - ``metadata.check_script``: shell script that exits 0 when task is solved
    - ``metadata.timeout_sec``: optional per-task timeout (default 180)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from examples.agentic_rl_datasets import DatasetAdapter, register_adapter
from slime.agent.sandbox import Sandbox, ensure_agent_user

logger = logging.getLogger(__name__)


@register_adapter
class CLIGymAdapter(DatasetAdapter):
    """Adapter for CLI-Gym interactive shell tasks."""

    name = "cli_gym"

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
                        "setup_script": metadata.get("setup_script", ""),
                        "check_script": metadata.get("check_script", ""),
                        "expected_state": metadata.get("expected_state", {}),
                        "timeout_sec": metadata.get("timeout_sec", 180),
                        "task_id": metadata.get("task_id", raw.get("id", str(len(tasks)))),
                        "workdir": metadata.get("workdir", "/home/agent"),
                    },
                    "label": metadata.get("task_id", str(len(tasks))),
                })
        return tasks

    async def setup_task(self, sb: Sandbox, metadata: dict[str, Any]) -> None:
        workdir = metadata.get("workdir", "/home/agent")
        await ensure_agent_user(sb, workdir)

        setup_script = metadata.get("setup_script", "")
        if setup_script.strip():
            await sb.write_file(f"{workdir}/.cli_gym_setup.sh", setup_script, user="agent")
            await sb.exec(
                f"chmod +x {workdir}/.cli_gym_setup.sh && cd {workdir} && bash {workdir}/.cli_gym_setup.sh",
                user="agent", check=False, timeout=120,
            )

    async def evaluate_task(
        self, sb: Sandbox, metadata: dict[str, Any], *, timeout_sec: int = 300
    ) -> float:
        workdir = metadata.get("workdir", "/home/agent")

        # Priority 1: check_script (most flexible)
        check_script = metadata.get("check_script", "")
        if check_script.strip():
            await sb.write_file(f"{workdir}/.cli_gym_check.sh", check_script, user="agent")
            ec, _, _ = await sb.exec(
                f"chmod +x {workdir}/.cli_gym_check.sh && cd {workdir} && bash {workdir}/.cli_gym_check.sh",
                user="agent", check=False, timeout=timeout_sec,
            )
            return 1.0 if ec == 0 else 0.0

        # Priority 2: expected_state dict
        expected_state = metadata.get("expected_state", {})
        if expected_state:
            checks_passed = 0
            for key, expected_value in expected_state.items():
                ec, stdout, _ = await sb.exec(
                    f"cd {workdir} && {key}",
                    user="agent", check=False, timeout=30,
                )
                actual = stdout.strip() if stdout else ""
                if actual == str(expected_value):
                    checks_passed += 1
            if checks_passed == len(expected_state):
                return 1.0
            return float(checks_passed) / max(len(expected_state), 1)

        logger.warning("[cli_gym] No check_script or expected_state; reward=0")
        return 0.0
