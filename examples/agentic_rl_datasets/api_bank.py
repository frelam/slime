"""API-Bank dataset adapter.

API-Bank evaluates the model's ability to call APIs correctly — picking the
right endpoint, constructing the request, and handling the response.  Tasks
include a natural language goal and an API specification.

Native format (JSONL):
    Each line has:
    - ``prompt`` / ``instruction``: what the model should accomplish
    - ``metadata.api_spec``: OpenAPI / tool definitions the model can use
    - ``metadata.setup_script``: optional commands to start a mock API server
    - ``metadata.check_script``: shell command that exits 0 when the task is solved
    - ``metadata.expected_api_calls``: list of expected API call signatures
    - ``metadata.timeout_sec``: optional per-task timeout (default 120)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from examples.agentic_rl_datasets import DatasetAdapter, register_adapter
from slime.agent.sandbox import Sandbox, ensure_agent_user

logger = logging.getLogger(__name__)


@register_adapter
class APIBankAdapter(DatasetAdapter):
    """Adapter for API-Bank tasks."""

    name = "api_bank"

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
                        "api_spec": metadata.get("api_spec", {}),
                        "setup_script": metadata.get("setup_script", ""),
                        "check_script": metadata.get("check_script", ""),
                        "expected_api_calls": metadata.get("expected_api_calls", []),
                        "timeout_sec": metadata.get("timeout_sec", 120),
                        "task_id": metadata.get("task_id", raw.get("id", str(len(tasks)))),
                        "workdir": metadata.get("workdir", "/home/agent"),
                    },
                    "label": metadata.get("task_id", str(len(tasks))),
                })
        return tasks

    async def setup_task(self, sb: Sandbox, metadata: dict[str, Any]) -> None:
        workdir = metadata.get("workdir", "/home/agent")
        await ensure_agent_user(sb, workdir)

        # Write API spec for model reference
        api_spec = metadata.get("api_spec", {})
        if api_spec:
            await sb.write_file(
                f"{workdir}/api_spec.json",
                json.dumps(api_spec, indent=2),
                user="agent",
            )

        # Start mock API server if setup_script provided
        setup_script = metadata.get("setup_script", "")
        if setup_script.strip():
            await sb.write_file(f"{workdir}/.api_bank_setup.sh", setup_script, user="agent")
            await sb.exec(
                f"chmod +x {workdir}/.api_bank_setup.sh && cd {workdir} && bash {workdir}/.api_bank_setup.sh",
                user="agent", check=False, timeout=120,
            )

    async def evaluate_task(
        self, sb: Sandbox, metadata: dict[str, Any], *, timeout_sec: int = 300
    ) -> float:
        workdir = metadata.get("workdir", "/home/agent")

        # Priority 1: check_script (most flexible)
        check_script = metadata.get("check_script", "")
        if check_script.strip():
            await sb.write_file(f"{workdir}/.api_bank_check.sh", check_script, user="agent")
            ec, _, _ = await sb.exec(
                f"chmod +x {workdir}/.api_bank_check.sh && cd {workdir} && bash {workdir}/.api_bank_check.sh",
                user="agent", check=False, timeout=timeout_sec,
            )
            return 1.0 if ec == 0 else 0.0

        # Priority 2: expected_api_calls — check the agent's call log
        expected_calls = metadata.get("expected_api_calls", [])
        if expected_calls:
            # Check if agent call log exists and matches expectations
            ec, stdout, _ = await sb.exec(
                f"cat {workdir}/.api_call_log.json 2>/dev/null || echo '[]'",
                user="agent", check=False, timeout=10,
            )
            try:
                actual_calls = json.loads(stdout) if stdout.strip() != "[]" else []
            except json.JSONDecodeError:
                actual_calls = []

            if not actual_calls:
                logger.warning("[api_bank] No API calls recorded; reward=0")
                return 0.0

            # Check each expected call is present
            matches = 0
            for expected in expected_calls:
                for actual in actual_calls:
                    if self._call_matches(expected, actual):
                        matches += 1
                        break

            return float(matches) / max(len(expected_calls), 1)

        logger.warning("[api_bank] No check_script or expected_api_calls; reward=0")
        return 0.0

    @staticmethod
    def _call_matches(expected: dict, actual: dict) -> bool:
        """Check if an actual API call matches the expected signature."""
        for key, expected_val in expected.items():
            actual_val = actual.get(key)
            if isinstance(expected_val, str) and isinstance(actual_val, str):
                if expected_val.lower() != actual_val.lower():
                    return False
            elif expected_val != actual_val:
                return False
        return True
