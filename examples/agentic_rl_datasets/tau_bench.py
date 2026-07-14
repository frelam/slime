"""τ-bench dataset adapter.

Wraps the existing ``examples/tau-bench/generate_with_tau.py`` interaction
loop into the unified ``DatasetAdapter`` interface.

Native format (JSONL):
    Each line has ``index`` (task index into tau-bench's train split) and
    optionally ``metadata`` with ``env`` (retail/airline), ``user_strategy``,
    etc.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from examples.agentic_rl_datasets import DatasetAdapter, register_adapter
from slime.agent.sandbox import Sandbox

logger = logging.getLogger(__name__)


# Default τ-bench configuration (mirrors examples/tau-bench/generate_with_tau.py).
# Users override via metadata keys in the dataset JSONL.
DEFAULT_TAU_CONFIG = {
    "env": "retail",
    "agent_strategy": "tool-calling",
    "user_model": "gemini-2.5-flash-lite",
    "task_split": "train",
    "user_strategy": "llm",
    "user_model_provider": "gemini",
}


@register_adapter
class TauBenchAdapter(DatasetAdapter):
    """Adapter for τ-bench (retail / airline tool-use environments)."""

    name = "tau_bench"

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        tasks = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                # Task index can be in "prompt" (as string) or "index"
                index = raw.get("index")
                if index is None:
                    try:
                        index = int(raw.get("prompt", ""))
                    except (ValueError, TypeError):
                        index = 0
                raw_meta = raw.get("metadata") or {}
                metadata = {
                    "task_index": index,
                    "env": raw_meta.get("env", DEFAULT_TAU_CONFIG["env"]),
                    "agent_strategy": raw_meta.get("agent_strategy", DEFAULT_TAU_CONFIG["agent_strategy"]),
                    "user_strategy": raw_meta.get("user_strategy", DEFAULT_TAU_CONFIG["user_strategy"]),
                    "user_model": raw_meta.get("user_model", DEFAULT_TAU_CONFIG["user_model"]),
                    "user_model_provider": raw_meta.get("user_model_provider", DEFAULT_TAU_CONFIG["user_model_provider"]),
                    "task_split": raw_meta.get("task_split", DEFAULT_TAU_CONFIG["task_split"]),
                }
                prompt = (
                    raw.get("prompt")
                    or raw_meta.get("instruction")
                    or f"τ-bench task {index} in {metadata['env']}"
                )
                tasks.append({"prompt": prompt, "metadata": metadata, "label": str(index)})
        return tasks

    async def setup_task(self, sb: Sandbox, metadata: dict[str, Any]) -> None:
        # τ-bench environments are typically handled in-process (via the
        # tau_bench library) rather than in a sandbox.  If a sandbox-based
        # setup is needed (e.g. installing tau_bench), do it here.
        # Currently a no-op; the actual env is created in evaluate_task.
        pass

    async def evaluate_task(self, sb: Sandbox, metadata: dict[str, Any], *, timeout_sec: int = 300) -> float:
        """Run the τ-bench environment to completion and return the cumulative
        reward as a fraction of max_possible_reward.

        NOTE: This requires the ``tau_bench`` and ``trainable_agents`` packages
        installed in the Python environment that runs this adapter.
        """
        # Lazy imports so missing dependencies don't break adapter discovery.
        from tau_bench.envs import get_env
        from tau_bench.types import RunConfig
        from examples.tau_bench.trainable_agents import InteractionResult, agent_factory  # type: ignore[import-untyped]

        env_name = metadata.get("env", DEFAULT_TAU_CONFIG["env"])
        tau_config = RunConfig(
            env=env_name,
            agent_strategy=metadata.get("agent_strategy", DEFAULT_TAU_CONFIG["agent_strategy"]),
            user_model=metadata.get("user_model", DEFAULT_TAU_CONFIG["user_model"]),
            user_strategy=metadata.get("user_strategy", DEFAULT_TAU_CONFIG["user_strategy"]),
            user_model_provider=metadata.get("user_model_provider", DEFAULT_TAU_CONFIG["user_model_provider"]),
            task_split=metadata.get("task_split", DEFAULT_TAU_CONFIG["task_split"]),
            model_provider="auto_router",
            model="slime_model",
        )

        task_index = metadata.get("task_index", 0)
        env = get_env(
            env_name=tau_config.env,
            user_strategy=tau_config.user_strategy,
            user_model=tau_config.user_model,
            user_provider=tau_config.user_model_provider,
            task_split=tau_config.task_split,
            task_index=task_index,
        )

        agent = agent_factory(
            tools_info=env.tools_info,
            wiki=env.wiki,
            config=tau_config,
            rollout_args=metadata.get("_rollout_args"),
            sampling_params=metadata.get("_sampling_params"),
        )

        result: InteractionResult = await agent.asolve(env, agent.rollout_args, agent.sampling_params, task_index)

        # τ-bench returns total_reward as a float (e.g. 0.75). Normalize to [0,1]
        # by dividing by max_possible_reward (typically 1.0 for retail single-task).
        raw_reward = getattr(result, "reward", 0.0) or 0.0
        return float(min(max(raw_reward, 0.0), 1.0))

    async def llm_judge(
        self,
        trajectory: list[dict[str, Any]],
        metadata: dict[str, Any],
        args: Any,
    ) -> float | None:
        from examples.agentic_rl.llm_judge import (
            DEFAULT_JUDGE_SYSTEM_PROMPT,
            build_judge_messages,
            call_llm_judge,
        )

        env = metadata.get("env", "retail")
        system_prompt = DEFAULT_JUDGE_SYSTEM_PROMPT + (
            f"\n\nFor τ-bench ({env}) tasks, evaluate whether the agent:"
            "\n1. Correctly understood the customer service scenario"
            "\n2. Took appropriate actions using available tools"
            "\n3. Achieved the task goal (e.g., booked flight, processed refund)"
        )
        task_desc = metadata.get("instruction", "")
        messages = build_judge_messages(system_prompt, task_desc, trajectory)
        max_retries = getattr(args, "llm_judge_max_retries", 2)
        return await call_llm_judge(args, messages, max_retries=max_retries)
