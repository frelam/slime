"""Simple dataset adapters for agentic RL GRPO training.

Provides adapters for synthetic simple datasets that work with the local
sglang_loop mode (no Docker/E2B required):

- ``SimpleShellAdapter``: Shell command tasks with check_command verification
- ``SimpleMathAdapter``: Math problems with answer matching
- ``SimpleCodeAdapter``: Coding tasks with test-based verification

All use pure outcome reward (rule-based verifier).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from examples.agentic_rl_datasets import DatasetAdapter, register_adapter
from slime.agent.sandbox import Sandbox, ensure_agent_user

logger = logging.getLogger(__name__)


# =============================================================================
# Simple Shell Adapter
# =============================================================================


@register_adapter
class SimpleShellAdapter(DatasetAdapter):
    """Adapter for simple shell command tasks.

    Tasks involve basic shell operations: file listing, text search,
    file creation, archiving, etc. Evaluation uses check_command exit code.

    This is the simplest agent RL dataset — binary outcome reward.
    """

    name = "simple_shell"

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                metadata = raw.get("metadata") or {}
                tasks.append({
                    "prompt": raw.get("prompt", ""),
                    "label": raw.get("label", metadata.get("task_id", "")),
                    "metadata": {
                        "benchmark": "simple_shell",
                        "task_id": metadata.get("task_id", ""),
                        "setup_commands": metadata.get("setup_commands", []),
                        "check_command": metadata.get("check_command", ""),
                        "expected_exit_code": metadata.get("expected_exit_code", 0),
                        "timeout_sec": metadata.get("timeout_sec", 120),
                        "workdir": metadata.get("workdir", "/home/agent"),
                        "max_turns": metadata.get("max_turns", 10),
                        "tags": metadata.get("tags", []),
                    },
                })
        return tasks

    async def setup_task(self, sb: Any, metadata: dict[str, Any]) -> None:
        await ensure_agent_user(sb, "/home/agent")
        for cmd in metadata.get("setup_commands", []):
            if cmd.strip():
                await sb.exec(cmd, user="agent", check=False, timeout=60)

    async def evaluate_task(
        self, sb: Any, metadata: dict[str, Any], *, timeout_sec: int = 300
    ) -> float:
        check_cmd = metadata.get("check_command", "")
        expected_exit_code = metadata.get("expected_exit_code", 0)

        if not check_cmd:
            logger.warning("[simple_shell] No check_command; reward=0")
            return 0.0

        try:
            ec, stdout, stderr = await sb.exec(
                check_cmd, user="agent", check=False, timeout=timeout_sec,
            )
            if ec == expected_exit_code:
                return 1.0
            else:
                logger.debug(
                    "[simple_shell] Check failed: exit=%d expected=%d cmd=%s",
                    ec, expected_exit_code, check_cmd[:100],
                )
                return 0.0
        except Exception:
            logger.exception("[simple_shell] Check command failed")
            return 0.0


# =============================================================================
# Simple Math Adapter
# =============================================================================


@register_adapter
class SimpleMathAdapter(DatasetAdapter):
    """Adapter for simple math problems (GSM8K-style).

    Agent uses Python tool to compute answers. Reward is based on
    answer matching against ground truth.
    """

    name = "simple_math"

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                metadata = raw.get("metadata") or {}
                tasks.append({
                    "prompt": raw.get("prompt", ""),
                    "label": raw.get("label", metadata.get("task_id", "")),
                    "metadata": {
                        "benchmark": "simple_math",
                        "task_id": metadata.get("task_id", ""),
                        "problem": metadata.get("problem", ""),
                        "expected_answer": str(metadata.get("expected_answer", "")),
                        "verify_code": metadata.get("verify_code", ""),
                        "timeout_sec": metadata.get("timeout_sec", 120),
                        "workdir": metadata.get("workdir", "/home/agent"),
                        "max_turns": metadata.get("max_turns", 5),
                        "setup_commands": metadata.get("setup_commands", []),
                    },
                })
        return tasks

    async def setup_task(self, sb: Any, metadata: dict[str, Any]) -> None:
        await ensure_agent_user(sb, "/home/agent")
        for cmd in metadata.get("setup_commands", []):
            if cmd.strip():
                await sb.exec(cmd, user="agent", check=False, timeout=60)

    async def evaluate_task(
        self, sb: Any, metadata: dict[str, Any], *, timeout_sec: int = 300
    ) -> float:
        """Evaluate math task by comparing answer to ground truth.

        Strategy:
          1. Run verify_code if available (Python assertion)
          2. Fall back: search for answer in agent's output files
          3. Both fail → 0.0
        """
        verify_code = metadata.get("verify_code", "")
        expected_answer = str(metadata.get("expected_answer", "")).strip()

        # Method 1: Run verify code (written to temp file to avoid
        # command injection via single-quote escaping in -c '...')
        if verify_code:
            try:
                import secrets
                import shlex

                # Write verify_code to a temp .py file, then execute it.
                # Avoids shell injection through -c quoting.
                verify_code = verify_code.replace("\\n", "\n").replace("\\t", "    ")
                tmp_name = f"/home/agent/_verify_{secrets.token_hex(8)}.py"
                await sb.write_file(tmp_name, verify_code, user="agent")
                ec, stdout, stderr = await sb.exec(
                    f"python3 {shlex.quote(tmp_name)}",
                    user="agent", check=False, timeout=timeout_sec,
                )
                # Clean up
                await sb.exec(
                    f"rm -f {shlex.quote(tmp_name)}",
                    user="agent", check=False, timeout=10,
                )
                if ec == 0:
                    return 1.0
            except Exception:
                logger.debug("[simple_math] verify_code execution failed", exc_info=True)

        # Method 2: Search for answer in output files
        try:
            ec, stdout, stderr = await sb.exec(
                "find /home/agent -name '*.py' -exec grep -l 'print' {} \\; 2>/dev/null | head -5",
                user="agent", check=False, timeout=30,
            )
        except Exception:
            stdout = ""

        # Method 3: Check file for answer
        if expected_answer:
            try:
                # Look for answer in any .txt or result files
                ec, stdout, stderr = await sb.exec(
                    f"grep -r '{expected_answer}' /home/agent/ --include='*.txt' "
                    f"--include='*.py' -l 2>/dev/null | head -3",
                    user="agent", check=False, timeout=30,
                )
                if stdout.strip():
                    return 1.0
            except Exception:
                pass

        return 0.0


# =============================================================================
# Simple Code Adapter
# =============================================================================


@register_adapter
class SimpleCodeAdapter(DatasetAdapter):
    """Adapter for simple coding tasks with test-based verification.

    Agent writes Python code to solve a problem. Reward is based on
    test pass rate (1.0 if all tests pass, 0.0 otherwise).
    """

    name = "simple_code"

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                metadata = raw.get("metadata") or {}
                tasks.append({
                    "prompt": raw.get("prompt", ""),
                    "label": raw.get("label", metadata.get("task_id", "")),
                    "metadata": {
                        "benchmark": "simple_code",
                        "task_id": metadata.get("task_id", ""),
                        "description": metadata.get("description", ""),
                        "test_code": metadata.get("test_code", ""),
                        "starter_code": metadata.get("starter_code", ""),
                        "check_command": metadata.get("check_command", ""),
                        "timeout_sec": metadata.get("timeout_sec", 180),
                        "workdir": metadata.get("workdir", "/home/agent"),
                        "max_turns": metadata.get("max_turns", 15),
                        "setup_commands": metadata.get("setup_commands", []),
                    },
                })
        return tasks

    async def setup_task(self, sb: Any, metadata: dict[str, Any]) -> None:
        await ensure_agent_user(sb, "/home/agent")
        for cmd in metadata.get("setup_commands", []):
            if cmd.strip():
                await sb.exec(cmd, user="agent", check=False, timeout=60)

    async def evaluate_task(
        self, sb: Any, metadata: dict[str, Any], *, timeout_sec: int = 300
    ) -> float:
        """Evaluate code task by running the test suite.

        Uses ``check_command`` (default: ``python3 test_solution.py``).
        Reward = 1.0 if tests pass, 0.0 otherwise.
        """
        check_cmd = metadata.get("check_command", "")
        if not check_cmd:
            # Default: run tests from /home/agent
            check_cmd = "cd /home/agent && python3 test_solution.py"

        try:
            ec, stdout, stderr = await sb.exec(
                check_cmd, user="agent", check=False, timeout=timeout_sec,
            )
            if ec == 0:
                return 1.0

            # Partial credit: count passed assertions
            if stdout:
                passed = stdout.count("passed") + stdout.count("All tests")
                if passed > 0:
                    return 0.5  # Some tests passed but not all

            logger.debug(
                "[simple_code] Tests failed (exit=%d): stdout=%s stderr=%s",
                ec, (stdout or "")[:200], (stderr or "")[:200],
            )
            return 0.0
        except Exception:
            logger.exception("[simple_code] Test execution failed")
            return 0.0
