"""Tool RL dataset adapter — for function-calling / tool-use training.

This adapter handles datasets produced by ``download_tool_data.py``.
Since tool RL training uses mock execution (no real API backend needed),
``setup_task`` is a no-op and ``evaluate_task`` returns 0.0 — the actual
reward is computed by the 4-dim reward system in ``tool_rl_reward.py``.

Supports both ``prompt`` and ``messages`` keys in JSONL input.

Sources: APIGen, ToolACE, Hammer, BFCL.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from examples.agentic_rl_datasets import DatasetAdapter, register_adapter

logger = logging.getLogger(__name__)


def _extract_fields(raw: dict) -> dict | None:
    """Extract prompt/label/metadata from a JSONL line, supporting both formats.

    Format A (standard slime, used by ``--input-key messages``):
        {"messages": [...], "tools": [...], "label": "...", "metadata": {...}}

    Format B (pre-formatted, used by dataset adapter):
        {"prompt": "...", "label": "...", "metadata": {...}}

    Returns:
        ``{"prompt": str, "label": str, "metadata": dict, "tools": list}``
        or ``None`` if invalid.
    """
    prompt = raw.get("prompt", "")
    if not prompt:
        # Format A: messages-based — convert to prompt text for adapter API
        messages = raw.get("messages", [])
        if messages:
            parts = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                parts.append(f"<|{role}|>\n{content}")
            prompt = "\n".join(parts)

    if not prompt:
        return None

    # Carry tools through metadata so the generate function can find them
    metadata = dict(raw.get("metadata", {}))
    tools = raw.get("tools", [])
    if tools and "tools" not in metadata:
        metadata["tools"] = tools

    return {
        "prompt": prompt,
        "label": raw.get("label", ""),
        "metadata": metadata,
    }


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    """Generic JSONL loader for tool RL data, supporting both formats."""
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
            fields = _extract_fields(raw)
            if fields:
                tasks.append(fields)
    return tasks


@register_adapter
class ToolRLAdapter(DatasetAdapter):
    """Adapter for tool-use RL training data (APIGen, ToolACE, Hammer, BFCL).

    The data is already in slime JSONL format from ``download_tool_data.py``.
    Supports both:
      - ``messages`` + ``tools`` + ``label`` + ``metadata`` (standard slime format)
      - ``prompt`` + ``label`` + ``metadata`` (pre-formatted adapter format)

    This adapter is intentionally minimal — the generate function
    (``tool_rl_generate.py``) handles all the agent loop and reward logic.
    """

    name = "tool_rl"

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        """Load the slime-format JSONL dataset.

        Args:
            path: Path to a JSONL file produced by ``download_tool_data.py``.

        Returns:
            List of task dicts with ``prompt``, ``label``, ``metadata``.
        """
        tasks = _load_jsonl(path)
        logger.info("Loaded %d tool RL tasks from %s", len(tasks), path)
        return tasks

    async def setup_task(self, sb: Any, metadata: dict[str, Any]) -> None:
        """No-op: tool RL uses mock execution, no real environment setup needed."""
        try:
            from slime.agent.sandbox import ensure_agent_user

            workdir = metadata.get("workdir", "/home/agent")
            await ensure_agent_user(sb, workdir)
        except Exception:
            pass

    async def evaluate_task(
        self,
        sb: Any,
        metadata: dict[str, Any],
        *,
        timeout_sec: int = 300,
    ) -> float:
        """Return 0.0 — the actual reward is computed by the 4-dim RM+Verifier."""
        return 0.0


# =============================================================================
# Source-specific adapters (registered for completeness)
# =============================================================================


@register_adapter
class APIGenAdapter(DatasetAdapter):
    """Adapter for APIGen-specific data (source='apigen')."""

    name = "apigen"

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        return _load_jsonl(path)

    async def setup_task(self, sb: Any, metadata: dict[str, Any]) -> None:
        pass

    async def evaluate_task(
        self,
        sb: Any,
        metadata: dict[str, Any],
        *,
        timeout_sec: int = 300,
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
        self,
        sb: Any,
        metadata: dict[str, Any],
        *,
        timeout_sec: int = 300,
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
        self,
        sb: Any,
        metadata: dict[str, Any],
        *,
        timeout_sec: int = 300,
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
        self,
        sb: Any,
        metadata: dict[str, Any],
        *,
        timeout_sec: int = 300,
    ) -> float:
        return 0.0
