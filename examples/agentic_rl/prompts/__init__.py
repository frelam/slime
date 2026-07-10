"""Per-benchmark prompt templates for agentic RL.

Each module exposes a ``build_prompt(task: dict) -> str`` that turns a task
dict (from ``DatasetAdapter.load_dataset``) into a model-facing instruction.
"""

from typing import Any


def build_prompt(benchmark: str, task: dict[str, Any]) -> str:
    """Dispatch prompt construction to the benchmark-specific builder."""
    if benchmark == "swe_gym_lite":
        from .swe import build_prompt as fn
    elif benchmark == "tau_bench":
        from .tau import build_prompt as fn
    elif benchmark == "terminal_bench":
        from .terminal import build_prompt as fn
    elif benchmark == "cli_gym":
        from .cli import build_prompt as fn
    elif benchmark == "api_bank":
        from .api_bank import build_prompt as fn
    elif benchmark == "r2e_gym":
        from .r2e import build_prompt as fn
    elif benchmark == "agent_bench":
        from .agent_bench import build_prompt as fn
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return fn(task)
