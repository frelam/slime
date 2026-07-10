"""Unified dataset adapters for agentic RL training.

Each adapter converts a benchmark's native data format into slime's JSONL
convention and provides setup/evaluation functions callable from a sandbox.

Usage::

    from examples.agentic_rl_datasets import get_adapter

    adapter = get_adapter("swe_gym_lite")
    tasks = adapter.load_dataset("/path/to/data.jsonl")
    async with Sandbox(image) as sb:
        await adapter.setup_task(sb, tasks[0]["metadata"])
        ...  # run agent loop
        reward = await adapter.evaluate_task(sb, tasks[0]["metadata"])
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class DatasetAdapter(ABC):
    """Interface each benchmark adapter implements."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name, e.g. ``"swe_gym_lite"``."""

    @abstractmethod
    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        """Load the native dataset and return a list of task dicts.

        Each task dict must have at least:
            - ``prompt`` (str): the user-facing instruction.
            - ``metadata`` (dict): benchmark-specific payload (task_id, setup
              commands, eval commands, image, …).

        The dict may also carry ``label`` for reference.
        """

    @abstractmethod
    async def setup_task(self, sb: Any, metadata: dict[str, Any]) -> None:
        """Prepare the sandbox for a single task.

        Called once per task before the agent loop starts. May clone repos,
        install dependencies, write problem statements, etc.
        """

    @abstractmethod
    async def evaluate_task(self, sb: Any, metadata: dict[str, Any], *, timeout_sec: int = 300) -> float:
        """Grade the agent's work and return a reward in [0, 1].

        The agent's changes should already be present in the sandbox (applied
        via git, file-writes, or equivalent). This method runs the benchmark's
        official evaluation and returns a scalar reward.
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_adapters: dict[str, type[DatasetAdapter]] = {}


def register_adapter(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    _adapters[cls.name] = cls  # type: ignore[attr-defined]
    return cls


def get_adapter(name: str) -> DatasetAdapter:
    """Return a cached instance of the named adapter."""
    if name not in _adapters:
        available = ", ".join(sorted(_adapters))
        raise KeyError(f"Unknown dataset adapter {name!r}. Available: {available}")
    return _adapters[name]()


# Lazy imports so adapters can be imported independently
def import_all() -> None:
    """Force-import every adapter module so their ``@register_adapter``
    decorators fire. Safe to call more than once."""
    import examples.agentic_rl_datasets.swe_gym_lite  # noqa: F401
    import examples.agentic_rl_datasets.tau_bench  # noqa: F401
    import examples.agentic_rl_datasets.terminal_bench  # noqa: F401
    import examples.agentic_rl_datasets.cli_gym  # noqa: F401
    import examples.agentic_rl_datasets.api_bank  # noqa: F401
    import examples.agentic_rl_datasets.r2e_gym  # noqa: F401
    import examples.agentic_rl_datasets.agent_bench  # noqa: F401
