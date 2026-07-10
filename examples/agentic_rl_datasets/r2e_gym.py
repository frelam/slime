"""R2E-Gym-Subset dataset adapter.

R2E-Gym (https://github.com/R2E-Gym/R2E-Gym) provides coding tasks in Docker
environments. The dataset ``R2E-Gym/R2E-Gym-Lite`` on HuggingFace has fields:
  - repo, base_commit, problem_statement
  - FAIL_TO_PASS, PASS_TO_PASS, test_patch

Supports both pre-converted JSONL and direct HuggingFace dataset loading
(path prefix ``hf:``).

Native format (JSONL / HuggingFace):
    Each task has a natural language ``problem_statement``, a repo to clone at
    ``base_commit``, a ``test_patch`` that adds the failing tests, and two lists
    of tests: ``FAIL_TO_PASS`` (should pass after fix) and ``PASS_TO_PASS``
    (must still pass).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from examples.agentic_rl_datasets import DatasetAdapter, register_adapter
from slime.agent.sandbox import Sandbox, ensure_agent_user

logger = logging.getLogger(__name__)

_DEFAULT_F2P_WEIGHT = 0.7
_DEFAULT_P2P_WEIGHT = 0.3
_DEFAULT_WORKDIR = "/testbed"
_CLONE_TIMEOUT = 120
_CHECKOUT_TIMEOUT = 60
_PATCH_APPLY_TIMEOUT = 60
_SETUP_TIMEOUT = 300


@register_adapter
class R2EGymSubsetAdapter(DatasetAdapter):
    """Adapter for R2E-Gym-Lite / R2E-Gym style coding tasks."""

    name = "r2e_gym"

    # ------------------------------------------------------------------
    # load_dataset
    # ------------------------------------------------------------------

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        """Load tasks from a JSONL file or HuggingFace dataset.

        Args:
            path: Either a local ``.jsonl`` file path or a string starting with
                ``hf:`` followed by an optional split name (default ``"train"``).

        Returns:
            List of task dicts with keys ``prompt``, ``label``, ``metadata``.
        """
        if path.startswith("hf:"):
            return self._load_from_huggingface(path[3:])
        return self._load_from_jsonl(path)

    def _load_from_jsonl(self, path: str) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("[r2e] Skipping malformed JSON line")
                    continue
                instance_id = raw.get("instance_id") or raw.get("id", "")
                if not instance_id:
                    logger.warning("[r2e] Skipping line without instance_id")
                    continue
                prompt = raw.get("prompt") or raw.get("problem_statement", "")
                meta = raw.get("metadata") or raw
                tasks.append(self._build_task(meta, prompt, instance_id))
        return tasks

    def _load_from_huggingface(self, split: str) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError as err:
            raise ImportError(
                "The 'datasets' library is required to load from HuggingFace. "
                "Install it with: pip install datasets"
            ) from err
        split = split or "train"
        ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split=split)
        tasks: list[dict[str, Any]] = []
        for row in ds:
            prompt = row.get("problem_statement", "")
            instance_id = row.get("instance_id", "unknown")
            tasks.append(self._build_task(dict(row), prompt, instance_id))
        return tasks

    @staticmethod
    def _build_task(meta: dict[str, Any], prompt: str, instance_id: str) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "label": instance_id,
            "metadata": {
                "instance_id": instance_id,
                "repo": meta.get("repo", ""),
                "base_commit": meta.get("base_commit", ""),
                "problem_statement": prompt,
                "FAIL_TO_PASS": meta.get("FAIL_TO_PASS", []) or [],
                "PASS_TO_PASS": meta.get("PASS_TO_PASS", []) or [],
                "test_patch": meta.get("test_patch", ""),
                "workdir": meta.get("workdir", _DEFAULT_WORKDIR),
                "image": meta.get("image"),
                "env_setup_script": meta.get("env_setup_script", ""),
                "f2p_weight": float(meta.get("f2p_weight", _DEFAULT_F2P_WEIGHT)),
                "p2p_weight": float(meta.get("p2p_weight", _DEFAULT_P2P_WEIGHT)),
                "timeout_sec": int(meta.get("timeout_sec", 300)),
            },
        }

    # ------------------------------------------------------------------
    # setup_task
    # ------------------------------------------------------------------

    async def setup_task(self, sb: Sandbox, metadata: dict[str, Any]) -> None:
        """Prepare the sandbox: clone repo, checkout commit, apply test patch."""
        workdir = metadata.get("workdir", _DEFAULT_WORKDIR)
        repo = metadata.get("repo", "")
        base_commit = metadata.get("base_commit", "")
        test_patch = metadata.get("test_patch", "")
        env_setup = metadata.get("env_setup_script", "")

        await ensure_agent_user(sb, workdir)

        # 1. Clone repository
        if repo:
            repo_url = f"https://github.com/{repo}.git"
            ec, _, _ = await sb.exec(
                f"git clone --depth 1 {repo_url} {workdir} 2>/dev/null || true",
                user="root",
                check=False,
                timeout=_CLONE_TIMEOUT,
            )
            if ec not in (0, None):
                logger.warning(
                    "[r2e] git clone failed (exit=%s) for %s", ec, repo
                )

        # 2. Checkout base commit
        if base_commit:
            await sb.exec(
                f"cd {workdir} && "
                f"git config --global --add safe.directory {workdir} && "
                f"git checkout {base_commit} -f",
                user="root",
                check=False,
                timeout=_CHECKOUT_TIMEOUT,
            )

        # 3. Write problem statement
        await sb.write_file(
            f"{workdir}/PROBLEM_STATEMENT.md",
            metadata.get("problem_statement", ""),
            user="agent",
        )

        # 4. Apply test patch
        if test_patch.strip():
            await sb.write_file(
                f"{workdir}/__r2e_test_patch.diff", test_patch, user="root"
            )
            ladder = " || ".join(
                f"({cmd})"
                for cmd in (
                    f"cd {workdir} && git apply --3way --whitespace=nowarn __r2e_test_patch.diff",
                    f"cd {workdir} && git apply --whitespace=nowarn __r2e_test_patch.diff",
                    f"cd {workdir} && patch -p1 --no-backup-if-mismatch < __r2e_test_patch.diff",
                )
            )
            ec, _, _ = await sb.exec(
                ladder, user="root", check=False, timeout=_PATCH_APPLY_TIMEOUT
            )
            if ec != 0:
                logger.warning(
                    "[r2e] test_patch apply failed for %s",
                    metadata.get("instance_id"),
                )
            # Clean up
            await sb.exec(
                f"rm -f {workdir}/__r2e_test_patch.diff",
                user="root",
                check=False,
                timeout=10,
            )

        # 5. Environment setup
        if env_setup.strip():
            await sb.write_file(
                f"{workdir}/__r2e_env_setup.sh", env_setup, user="root"
            )
            await sb.exec(
                f"chmod +x {workdir}/__r2e_env_setup.sh && "
                f"cd {workdir} && bash __r2e_env_setup.sh",
                user="root",
                check=False,
                timeout=_SETUP_TIMEOUT,
            )

    # ------------------------------------------------------------------
    # evaluate_task
    # ------------------------------------------------------------------

    async def evaluate_task(
        self, sb: Sandbox, metadata: dict[str, Any], *, timeout_sec: int = 300
    ) -> float:
        """Run pytest and return a fractional reward in [0, 1].

        Weight: 0.7 * FAIL_TO_PASS_ratio + 0.3 * PASS_TO_PASS_ratio.
        """
        workdir = metadata.get("workdir", _DEFAULT_WORKDIR)
        f2p_tests: list[str] = metadata.get("FAIL_TO_PASS", []) or []
        p2p_tests: list[str] = metadata.get("PASS_TO_PASS", []) or []
        f2p_weight = float(metadata.get("f2p_weight", _DEFAULT_F2P_WEIGHT))
        p2p_weight = float(metadata.get("p2p_weight", _DEFAULT_P2P_WEIGHT))

        if not f2p_tests and not p2p_tests:
            logger.warning("[r2e] No tests to run; reward=0")
            return 0.0

        f2p_passed, f2p_total = await self._run_tests(
            sb, workdir, f2p_tests, timeout_sec, "f2p"
        )
        p2p_passed, p2p_total = await self._run_tests(
            sb, workdir, p2p_tests, timeout_sec, "p2p"
        )

        f2p_score = f2p_passed / f2p_total if f2p_total > 0 else 1.0
        p2p_score = p2p_passed / p2p_total if p2p_total > 0 else 1.0

        reward = f2p_weight * f2p_score + p2p_weight * p2p_score
        logger.info(
            "[r2e] F2P=%d/%d P2P=%d/%d reward=%.4f",
            f2p_passed, f2p_total, p2p_passed, p2p_total, reward,
        )
        return float(min(max(reward, 0.0), 1.0))

    async def _run_tests(
        self,
        sb: Sandbox,
        workdir: str,
        tests: list[str],
        timeout_sec: int,
        tag: str,
    ) -> tuple[int, int]:
        """Run a set of pytest tests and return (passed, total)."""
        if not tests:
            return 0, 0
        test_ids = " ".join(tests)
        _, stdout, stderr = await sb.exec(
            f"cd {workdir} && "
            f"python -m pytest {test_ids} --no-header -q --tb=no -rN 2>&1 || true",
            user="agent",
            check=False,
            timeout=timeout_sec,
        )
        output = (stdout or "") + (stderr or "")

        if "no tests ran" in output:
            logger.warning(
                "[r2e.%s] No tests collected; tests=%s output=%r",
                tag, test_ids, output[-200:],
            )
            return 0, len(tests)

        # Count PASSED per test_id in pytest short output
        passed = 0
        for test_id in tests:
            if re.search(rf"{re.escape(test_id)}\s+PASSED", output):
                passed += 1
        return passed, len(tests)
