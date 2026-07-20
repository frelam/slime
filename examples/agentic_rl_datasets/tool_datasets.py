"""Tool RL dataset adapter — for function-calling / tool-use training.

This adapter handles datasets produced by ``download_tool_data.py``.
Since tool RL training uses mock execution (no real API backend needed),
``setup_task`` is a no-op and ``evaluate_task`` returns 0.0 — the actual
reward is computed by the 4-dim reward system in ``tool_rl_reward.py``.

Sources: APIGen, ToolACE, Hammer, BFCL.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from examples.agentic_rl_datasets import DatasetAdapter, register_adapter

logger = logging.getLogger(__name__)


@register_adapter
class ToolRLAdapter(DatasetAdapter):
    """Adapter for tool-use RL training data (APIGen, ToolACE, Hammer, BFCL).

    The data is already in slime JSONL format from ``download_tool_data.py``.
    Each line has ``prompt``, ``label``, and ``metadata`` with tool definitions
    and optional ground truth.

    This adapter is intentionally minimal — the generate function
    (``tool_rl_generate.py``) handles all the agent loop and reward logic.
    """

    name = "tool_rl"

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        """Load the slime-format JSONL dataset.

        Each line is already a complete task dict with ``prompt``, ``label``,
        ``metadata`` keys.

        Args:
            path: Path to a JSONL file produced by ``download_tool_data.py``.

        Returns:
            List of task dicts.
        """
        tasks: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON at line %d", line_num + 1)
                    continue

                prompt = raw.get("prompt", "")
                if not prompt:
                    continue

                tasks.append({
                    "prompt": prompt,
                    "label": raw.get("label", ""),
                    "metadata": raw.get("metadata", {}),
                })

        logger.info("Loaded %d tool RL tasks from %s", len(tasks), path)
        return tasks

    async def setup_task(self, sb: Any, metadata: dict[str, Any]) -> None:
        """No-op: tool RL uses mock execution, no real environment setup needed.

        The available tools are defined in the prompt itself — no sandbox
        preparation is required.
        """
        # Ensure agent workdir exists (compatibility with the framework)
        try:
            from slime.agent.sandbox import ensure_agent_user
            workdir = metadata.get("workdir", "/home/agent")
            await ensure_agent_user(sb, workdir)
        except Exception:
            # Mock execution doesn't need a real sandbox
            pass

    async def evaluate_task(
        self, sb: Any, metadata: dict[str, Any], *, timeout_sec: int = 300,
    ) -> float:
        """Return 0.0 — the actual reward is computed by the 4-dim RM+Verifier.

        For tool RL, the reward comes from the trajectory quality, not from
        a binary task-completion check. The RM scores planning and hallucination,
        while the verifier scores format and tool call correctness.

        Returns:
            0.0 (placeholder — real reward is in sample metadata).
        """
        return 0.0


# =============================================================================
# Source-specific adapters (registered for completeness)
# =============================================================================


@register_adapter
class APIGenAdapter(DatasetAdapter):
    """Adapter for APIGen-specific data (source='apigen').

    Same behavior as ToolRLAdapter — just with a different name for
    source-level tracking.
    """

    name = "apigen"

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not raw.get("prompt"):
                    continue
                tasks.append({
                    "prompt": raw["prompt"],
                    "label": raw.get("label", ""),
                    "metadata": raw.get("metadata", {}),
                })
        return tasks

    async def setup_task(self, sb: Any, metadata: dict[str, Any]) -> None:
        pass

    async def evaluate_task(
        self, sb: Any, metadata: dict[str, Any], *, timeout_sec: int = 300,
    ) -> float:
        return 0.0


@register_adapter
class ToolACEAdapter(DatasetAdapter):
    """Adapter for ToolACE-specific data."""

    name = "toolace"

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        return _load_jsonl(path)

    async def setup_task(self, sb: Any, metadata: dict[str, Any]) -> None:
        pass

    async def evaluate_task(
        self, sb: Any, metadata: dict[str, Any], *, timeout_sec: int = 300,
    ) -> float:
        return 0.0


@register_adapter
class HammerAdapter(DatasetAdapter):
    """Adapter for Hammer-specific data."""

    name = "hammer"

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        return _load_jsonl(path)

    async def setup_task(self, sb: Any, metadata: dict[str, Any]) -> None:
        pass

    async def evaluate_task(
        self, sb: Any, metadata: dict[str, Any], *, timeout_sec: int = 300,
    ) -> float:
        return 0.0


@register_adapter
class BFCLAdapter(DatasetAdapter):
    """Adapter for BFCL-specific data."""

    name = "bfcl"

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        return _load_jsonl(path)

    async def setup_task(self, sb: Any, metadata: dict[str, Any]) -> None:
        pass

    async def evaluate_task(
        self, sb: Any, metadata: dict[str, Any], *, timeout_sec: int = 300,
    ) -> float:
        return 0.0


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    """Generic JSONL loader for slime-format data."""
    tasks: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not raw.get("prompt"):
                continue
            tasks.append({
                "prompt": raw["prompt"],
                "label": raw.get("label", ""),
                "metadata": raw.get("metadata", {}),
            })
    return tasks
