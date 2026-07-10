"""AgentBench dataset adapter.

AgentBench (https://github.com/THUDM/AgentBench) has 8 environments.
For sandbox-based training, OS (bash commands) and DB (SQL) are fully
supported. Other task types (kg, dcg, ltp, hh, ws, wb) use stub evaluation
that returns 0 — they are loaded but do not contribute reward signal.

Native format (JSONL):
    Each line has:
    - ``prompt`` / ``instruction``: task description
    - ``metadata.task_type``: "os" | "db" | "kg" | "dcg" | "ltp" | "hh" | "ws" | "wb"
    - ``metadata.evaluation``: dict with type-specific grading parameters
    - ``metadata.setup_script``: optional shell commands for environment prep
"""

from __future__ import annotations

import json
import logging
from typing import Any

from examples.agentic_rl_datasets import DatasetAdapter, register_adapter
from slime.agent.sandbox import Sandbox, ensure_agent_user

logger = logging.getLogger(__name__)

_SUPPORTED_TASK_TYPES = frozenset({
    "os", "db", "kg", "dcg", "ltp", "hh", "ws", "wb",
})
_FULLY_SUPPORTED = frozenset({"os", "db"})
_DEFAULT_WORKDIR = "/home/agent"


@register_adapter
class AgentBenchAdapter(DatasetAdapter):
    """Adapter for AgentBench multi-environment tasks."""

    name = "agent_bench"

    # ------------------------------------------------------------------
    # load_dataset
    # ------------------------------------------------------------------

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        """Load tasks from a JSONL file.

        Each line must have a ``metadata.task_type`` in the supported set.
        Unknown task types produce a warning and are skipped.
        """
        tasks: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("[agent_bench] Skipping malformed JSON line")
                    continue

                meta = raw.get("metadata") or {}
                task_type = meta.get("task_type", "")

                if task_type not in _SUPPORTED_TASK_TYPES:
                    logger.warning(
                        "[agent_bench] Unknown task_type %r; skipping", task_type
                    )
                    continue

                prompt = raw.get("prompt") or raw.get("instruction", "")
                task_id = meta.get("task_id") or raw.get("id", str(len(tasks)))

                evaluation = meta.get("evaluation") or {}
                if isinstance(evaluation, str):
                    try:
                        evaluation = json.loads(evaluation)
                    except json.JSONDecodeError:
                        evaluation = {}

                tasks.append({
                    "prompt": prompt,
                    "label": task_id,
                    "metadata": {
                        "task_type": task_type,
                        "task_id": task_id,
                        "timeout_sec": int(meta.get("timeout_sec", 180)),
                        "workdir": meta.get("workdir", _DEFAULT_WORKDIR),
                        "setup_script": meta.get("setup_script", ""),
                        "evaluation": evaluation,
                    },
                })
        return tasks

    # ------------------------------------------------------------------
    # setup_task
    # ------------------------------------------------------------------

    async def setup_task(self, sb: Sandbox, metadata: dict[str, Any]) -> None:
        task_type = metadata.get("task_type", "os")
        workdir = metadata.get("workdir", _DEFAULT_WORKDIR)
        setup_script = metadata.get("setup_script", "")
        evaluation = metadata.get("evaluation", {})

        await ensure_agent_user(sb, workdir)

        if task_type == "db":
            await self._setup_db(sb, workdir, evaluation)
        elif task_type in _FULLY_SUPPORTED:
            # OS tasks need no special setup beyond ensure_agent_user
            pass
        else:
            logger.info(
                "[agent_bench] Stub setup for task_type=%s", task_type
            )

        # Run optional setup script
        if setup_script.strip():
            await sb.write_file(
                f"{workdir}/.agentbench_setup.sh", setup_script, user="agent"
            )
            await sb.exec(
                f"chmod +x {workdir}/.agentbench_setup.sh && "
                f"cd {workdir} && bash {workdir}/.agentbench_setup.sh",
                user="agent",
                check=False,
                timeout=120,
            )

    async def _setup_db(
        self, sb: Sandbox, workdir: str, evaluation: dict[str, Any]
    ) -> None:
        """Initialize MySQL database for DB-type tasks."""
        db_name = evaluation.get("db_name", "testdb")
        db_password = evaluation.get("db_root_password", "root")
        setup_sql = evaluation.get("setup_sql", "")

        # Start MySQL
        await sb.exec(
            "service mysql status 2>/dev/null || "
            "(service mysql start 2>/dev/null || mysqld_safe --skip-grant-tables &) "
            "&& sleep 2",
            user="root", check=False, timeout=30,
        )

        # Create database (with or without password)
        await sb.exec(
            f'mysql -u root -p{db_password} -e '
            f'"CREATE DATABASE IF NOT EXISTS {db_name}" 2>/dev/null || '
            f'mysql -u root -e "CREATE DATABASE IF NOT EXISTS {db_name}"',
            user="root", check=False, timeout=30,
        )

        # Run setup SQL
        if setup_sql.strip():
            await sb.write_file(
                f"{workdir}/.agentbench_db_setup.sql", setup_sql, user="root"
            )
            await sb.exec(
                f"mysql -u root -p{db_password} {db_name} "
                f"< {workdir}/.agentbench_db_setup.sql 2>/dev/null || "
                f"mysql -u root {db_name} "
                f"< {workdir}/.agentbench_db_setup.sql",
                user="root", check=False, timeout=30,
            )

    # ------------------------------------------------------------------
    # evaluate_task
    # ------------------------------------------------------------------

    async def evaluate_task(
        self, sb: Sandbox, metadata: dict[str, Any], *, timeout_sec: int = 300
    ) -> float:
        task_type = metadata.get("task_type", "os")
        evaluation = metadata.get("evaluation", {})

        if task_type == "os":
            return await self._evaluate_os(metadata, evaluation, sb, timeout_sec)
        elif task_type == "db":
            return await self._evaluate_db(metadata, evaluation, sb, timeout_sec)
        else:
            logger.info(
                "[agent_bench] Stub eval for task_type=%s: %s",
                task_type,
                evaluation.get("message", "No evaluation implemented"),
            )
            return 0.0

    # -- OS evaluation --------------------------------------------------

    async def _evaluate_os(
        self,
        metadata: dict[str, Any],
        evaluation: dict[str, Any],
        sb: Sandbox,
        timeout: int,
    ) -> float:
        workdir = metadata.get("workdir", _DEFAULT_WORKDIR)
        eval_type = evaluation.get("type", "command_check")
        check_command = evaluation.get("check_command", "")
        expected_output = evaluation.get("expected_output", "")
        exact_output = evaluation.get("exact_output", "")
        expected_exit_code = int(evaluation.get("expected_exit_code", 0))

        if not check_command and not expected_output and not exact_output:
            logger.warning("[agent_bench.os] No evaluation criteria; reward=0")
            return 0.0

        if check_command:
            ec, stdout, stderr = await sb.exec(
                f"cd {workdir} && {check_command}",
                user="agent", check=False, timeout=timeout,
            )
            combined = (stdout or "") + (stderr or "")
        else:
            ec, stdout, _ = await sb.exec(
                f"cat {workdir}/.agent_log.txt 2>/dev/null || "
                f"tail -c 10240 {workdir}/.bash_history 2>/dev/null || echo ''",
                user="agent", check=False, timeout=10,
            )
            combined = stdout or ""

        score = 1.0

        if eval_type == "command_check":
            if ec != expected_exit_code:
                score -= 1.0
        elif eval_type == "output_match":
            if exact_output:
                if combined.strip() != exact_output.strip():
                    score -= 1.0
            elif expected_output:
                if expected_output not in combined:
                    score -= 1.0

        return max(0.0, score)

    # -- DB evaluation --------------------------------------------------

    async def _evaluate_db(
        self,
        metadata: dict[str, Any],
        evaluation: dict[str, Any],
        sb: Sandbox,
        timeout: int,
    ) -> float:
        workdir = metadata.get("workdir", _DEFAULT_WORKDIR)
        db_name = evaluation.get("db_name", "testdb")
        db_password = evaluation.get("db_root_password", "root")
        expected_query = evaluation.get("expected_query", "")
        expected_rows: list[list[Any]] = evaluation.get("expected_result", []) or []
        exact_match = bool(evaluation.get("exact_match", False))

        if not expected_query and not expected_rows:
            logger.warning(
                "[agent_bench.db] No evaluation query or expected result; reward=0"
            )
            return 0.0

        if expected_query:
            ec, stdout, _ = await sb.exec(
                f'cd {workdir} && '
                f'mysql -u root -p{db_password} {db_name} -e "{expected_query}" '
                f'2>/dev/null || mysql -u root {db_name} -e "{expected_query}"',
                user="root", check=False, timeout=timeout,
            )
            if ec != 0:
                logger.warning("[agent_bench.db] Query returned non-zero exit code")
                return 0.0

            lines = [
                ln.strip() for ln in (stdout or "").split("\n") if ln.strip()
            ]
            if not lines:
                return 1.0 if not expected_rows else 0.0

            # Header row is skipped; rest are data
            data_rows: list[list[str]] = [
                ln.split("\t") for ln in lines[1:]
            ]

            if not expected_rows:
                return 1.0 if not data_rows else 0.0

            if exact_match:
                ok = data_rows == expected_rows
            else:
                ok = sorted(str(r) for r in data_rows) == sorted(
                    str(r) for r in expected_rows
                )
            return 1.0 if ok else 0.0

        return 0.0
