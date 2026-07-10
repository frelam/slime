"""SWE-Gym-Lite dataset adapter.

Reuses the grading infrastructure from ``examples/coding_agent_rl/swe.py``
(scaleswe and swebench protocols) and wraps it in the unified
``DatasetAdapter`` interface.

Native format (JSONL):
    Each line is an SWE-bench-style instance dict with keys like
    ``instance_id``, ``repo``, ``base_commit``, ``problem_statement``,
    ``hints_text``, ``test_patch``, ``FAIL_TO_PASS``, ``PASS_TO_PASS``,
    and optionally ``image`` / ``pre_commands`` / ``workdir``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from examples.agentic_rl_datasets import DatasetAdapter, register_adapter
from slime.agent.sandbox import Sandbox, ensure_agent_user, exec_and_wait

logger = logging.getLogger(__name__)

# Re-export grading helpers from coding_agent_rl so callers can use them
# without a separate import.
from examples.coding_agent_rl.swe import (  # noqa: F401
    EvalResult,
    PROTOCOL_SCALESWE,
    PROTOCOL_SWEBENCH,
    apply_pre_commands,
    evaluability_check,
    get_metadata,
    git_diff,
    prepare_workspace,
    run_evaluation,
)


def _load_scaleswe(path: str) -> list[dict[str, Any]]:
    """Load scaleswe-format JSONL."""
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            prompt = (
                raw.get("problem_statement")
                or raw.get("prompt")
                or ""
            )
            metadata = {
                k: raw[k]
                for k in ("instance_id", "image", "workdir", "swepro", "eval_cmd",
                          "pre_commands", "remote_env_info")
                if k in raw
            }
            metadata.setdefault("instance_id", raw.get("instance_id", "unknown"))
            tasks.append({"prompt": prompt, "metadata": metadata, "label": metadata["instance_id"]})
    return tasks


def _load_swebench(path: str) -> list[dict[str, Any]]:
    """Load SWE-bench Verified JSONL (remote_env_info envelope)."""
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rem = raw.get("remote_env_info") or {}
            prompt = rem.get("problem_statement") or raw.get("problem_statement", "")
            metadata = {
                "instance_id": rem.get("instance_id", raw.get("instance_id", "unknown")),
                "image": rem.get("image"),
                "workdir": rem.get("workdir", "/testbed"),
                "remote_env_info": rem,
            }
            tasks.append({"prompt": prompt, "metadata": metadata, "label": metadata["instance_id"]})
    return tasks


@register_adapter
class SWEGymLiteAdapter(DatasetAdapter):
    """Adapter for SWE-Gym-Lite / SWE-bench style coding tasks."""

    name = "swe_gym_lite"

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        path = str(path)
        # Detect format by looking at the first line
        with open(path) as f:
            first = f.readline().strip()
        if not first:
            return []
        first = json.loads(first)
        if "remote_env_info" in first:
            return _load_swebench(path)
        return _load_scaleswe(path)

    async def setup_task(self, sb: Sandbox, metadata: dict[str, Any]) -> None:
        # Build a minimal Sample-like object so get_metadata / prepare_workspace
        # work.

        class _FakeSample:
            prompt = metadata.get("problem_statement", "")
            metadata = metadata
            label = metadata.get("instance_id")

        md = get_metadata(_FakeSample, protocol=PROTOCOL_SCALESWE)  # type: ignore[arg-type]
        # Fall back to swebench if scaleswe detection + heuristics say so
        if md.get("looks_swebench") or metadata.get("remote_env_info", {}).get("test_patch"):
            md = get_metadata(_FakeSample, protocol=PROTOCOL_SWEBENCH)  # type: ignore[arg-type]

        workdir = md.get("workdir") or "/testbed"
        await ensure_agent_user(sb, workdir)

        # Prepare workspace: clone repo, write problem statement, run pre-commands
        image = md.get("image") or metadata.get("image")
        if image and not hasattr(sb, "_swe_sandbox_image"):
            sb._swe_sandbox_image = image  # type: ignore[attr-defined]

        if md.get("protocol") == PROTOCOL_SWEBENCH:
            inst = md.get("grading", {}).get("sweb_instance") or {}
            # For SWE-bench: clone the repo at base_commit
            repo_url = f"https://github.com/{inst.get('repo', '')}.git"
            await sb.exec(
                f"git clone {repo_url} {workdir} 2>/dev/null || true",
                user="root", check=False, timeout=120
            )
            base_commit = inst.get("base_commit", "")
            if base_commit:
                await sb.exec(
                    f"cd {workdir} && git config --global --add safe.directory {workdir} "
                    f"&& git checkout {base_commit} -f",
                    user="root", check=False, timeout=60
                )
            await sb.write_file(
                f"{workdir}/PROBLEM_STATEMENT.md",
                inst.get("problem_statement", ""),
                user="root",
            )
        else:
            await prepare_workspace(sb, workdir, md)

    async def evaluate_task(
        self, sb: Sandbox, metadata: dict[str, Any], *, timeout_sec: int = 300
    ) -> float:
        # Build metadata dict from sandbox state
        md = self._build_md(metadata)
        diff_text = await git_diff(sb, md.get("workdir", "/testbed"))
        result = await run_evaluation(md, diff_text=diff_text, timeout_sec=timeout_sec)
        return result.reward

    def _build_md(self, metadata: dict) -> dict:
        """Rebuild the metadata dict for evaluation (same shape as get_metadata)."""

        class _FakeSample:
            prompt = metadata.get("problem_statement", "")
            metadata = metadata
            label = metadata.get("instance_id")

        md = get_metadata(_FakeSample, protocol=PROTOCOL_SCALESWE)  # type: ignore[arg-type]
        if md.get("looks_swebench") or metadata.get("remote_env_info", {}).get("test_patch"):
            md = get_metadata(_FakeSample, protocol=PROTOCOL_SWEBENCH)  # type: ignore[arg-type]
        return md
