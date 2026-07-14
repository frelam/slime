"""GRPO/PPO custom generate function for agentic RL.

Plug into slime via ``--custom-generate-function-path``::

    python train_async.py \\
        --custom-generate-function-path examples.agentic_rl_grpo.generate.agentic_grpo_generate \\
        ...

For each input sample, this function:

1. Detects the benchmark from sample metadata and resolves the dataset adapter.
2. Routes to the appropriate harness based on task type:
   - **SWE tasks** (swe_gym_lite, r2e_gym): Claude Code harness +
     AnthropicAdapter for logprob capture.
   - **General tasks** (terminal_bench, cli_gym, tau_bench, api_bank,
     agent_bench, …): Hermes harness + OpenAIAdapter for logprob capture.
3. Boots an E2B sandbox, installs the harness, and runs the agent.
4. Evaluates the task via adapter.evaluate_task().
5. Computes reward:
   - **General tasks**: multi-dimensional reward (RM + verifier, 7 dims).
   - **SWE tasks**: task evaluation reward only (test pass rate).
6. Returns ``list[Sample]`` with rollout_log_probs and scalar reward.

Harness routing
---------------

Task type is read from ``sample.metadata["benchmark"]`` (or auto-detected from
metadata fields).  The mapping is:

===============  =============  ======================
Task types       Harness        Adapter (logprobs)
===============  =============  ======================
swe_gym_lite     Claude Code    AnthropicAdapter
r2e_gym          Claude Code    AnthropicAdapter
everything else  Hermes         OpenAIAdapter
===============  =============  ======================

Two adapter HTTP servers run on different ports so both protocols can coexist
in the same rollout:
  - Port 18001 → Anthropic (Claude Code)
  - Port 18002 → OpenAI (Hermes)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import shlex
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from slime.agent.adapters import AnthropicAdapter, OpenAIAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.harness import (
    ClaudeCodeHarness,
    HermesHarness,
)
from slime.agent.sandbox import E2BSandbox
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# =============================================================================
# Task-type → harness routing
# =============================================================================

# SWE tasks → Claude Code (Anthropic protocol)
_SWE_TASK_TYPES = frozenset({"swe_gym_lite", "r2e_gym"})

# Harness + adapter for each category
_SWE_HARNESS_CLS = ClaudeCodeHarness
_SWE_ADAPTER_CLS = AnthropicAdapter
_GENERAL_HARNESS_CLS = HermesHarness
_GENERAL_ADAPTER_CLS = OpenAIAdapter

# =============================================================================
# Configuration from environment
# =============================================================================


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key, "")
    return int(val) if val else default


# Ports for the two adapter protocols
_ANTHROPIC_ADAPTER_PORT = int(os.environ.get("ADAPTER_PORT_ANTHROPIC", "18001"))
_OPENAI_ADAPTER_PORT = int(os.environ.get("ADAPTER_PORT_OPENAI", "18002"))

CONFIG = {
    "adapter_public_host": os.environ.get("ADAPTER_PUBLIC_HOST", ""),
    "adapter_bind_host": os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
    "anthropic_port": _ANTHROPIC_ADAPTER_PORT,
    "openai_port": _OPENAI_ADAPTER_PORT,
    "agent_time_budget_sec": _env_int("SWE_AGENT_TIME_BUDGET_SEC", 1800),
    "eval_timeout_sec": _env_int("SWE_EVAL_TIMEOUT_SEC", 600),
    "rollout_guard_sec": _env_int("SWE_ROLLOUT_GUARD_SEC", 0) or (
        _env_int("SWE_AGENT_TIME_BUDGET_SEC", 1800)
        + _env_int("SWE_EVAL_TIMEOUT_SEC", 600)
        + 180
    ),
    "boot_concurrency": _env_int("SWE_BOOT_CONCURRENCY", 16),
    "boot_retries": _env_int("SWE_BOOT_RETRIES", 2),
    "fork_merge_threshold": (
        int(os.environ["SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS"])
        if "SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS" in os.environ
        else None
    ),
}

_BOOT_SEM = asyncio.Semaphore(CONFIG["boot_concurrency"])


# =============================================================================
# Sandbox boot (shared by all harness types)
# =============================================================================


@asynccontextmanager
async def boot_agent_sandbox(
    image: str,
    instance_id: str,
    harness_cls: type,
) -> AsyncIterator[E2BSandbox]:
    """Boot an E2B sandbox with *harness_cls* CLI installed.

    Args:
        image: E2B Docker image name.
        instance_id: Task instance ID (for logging).
        harness_cls: The harness class to install (e.g. ``HermesHarness``).

    Yields:
        A ready-to-use ``E2BSandbox``.
    """
    import random

    sb: E2BSandbox | None = None
    last_err: Exception | None = None

    for attempt in range(CONFIG["boot_retries"]):
        cand = E2BSandbox(image)
        try:
            async with _BOOT_SEM:
                await cand.__aenter__()
                try:
                    await harness_cls().install_cli(cand)
                except BaseException:
                    await cand.__aexit__(None, None, None)
                    raise
            sb = cand
            break
        except Exception as e:
            last_err = e
            logger.warning(
                "[agentic_grpo] %s: boot attempt %d/%d failed: %s: %s",
                instance_id,
                attempt + 1,
                CONFIG["boot_retries"],
                type(e).__name__,
                str(e)[:200],
            )
            await asyncio.sleep(1 + attempt + random.random())

    if sb is None:
        assert last_err is not None
        raise last_err

    try:
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


# =============================================================================
# Adapter services — two singletons, one per protocol
# =============================================================================


def _make_adapter_service(
    adapter_cls: type,
    port: int,
    thread_name: str,
    args: Any,
) -> _AdapterService:
    """Create and start an adapter HTTP service."""
    return _AdapterService(args, adapter_cls, port, thread_name)


class _AdapterService:
    """One adapter + HTTP server, explicitly keyed by protocol.

    Two instances coexist:
      - Anthropic on port 18001 (for Claude Code)
      - OpenAI   on port 18002 (for Hermes)
    """

    _instances: dict[str, _AdapterService] = {}

    def __init__(
        self,
        args: Any,
        adapter_cls: type,
        port: int,
        thread_name: str,
    ) -> None:
        self.tokenizer = load_tokenizer(
            args.hf_checkpoint, trust_remote_code=True
        )
        self.max_context_len = int(
            getattr(args, "rollout_max_context_len", 0) or 0
        )
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = (
            getattr(args, "sglang_reasoning_parser", None) or None
        )
        sglang_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        )
        if not CONFIG["adapter_public_host"]:
            raise RuntimeError(
                "ADAPTER_PUBLIC_HOST is not set. "
                "Export it to the host IP that sandboxes can reach."
            )
        self.adapter = adapter_cls(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            fork_threshold_tokens=CONFIG["fork_merge_threshold"],
        )
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=CONFIG["adapter_bind_host"],
            port=port,
            thread_name=thread_name,
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        self.adapter_url = (
            f"http://{CONFIG['adapter_public_host']}:{self.app_handle.port}"
        )
        logger.info(
            "[agentic_grpo] adapter %s on %s max_ctx=%d",
            thread_name,
            self.adapter_url,
            self.max_context_len,
        )

    @classmethod
    def get_or_create(
        cls,
        args: Any,
        adapter_cls: type,
        port: int,
        thread_name: str,
    ) -> _AdapterService:
        """Return the singleton for *thread_name*, creating it if needed."""
        if thread_name not in cls._instances:
            cls._instances[thread_name] = _AdapterService(
                args, adapter_cls, port, thread_name
            )
        return cls._instances[thread_name]


# =============================================================================
# Main generate function — entry point for slime
# =============================================================================


async def agentic_grpo_generate(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any] | None = None,
    evaluation: bool = False,
) -> list[Sample]:
    """GRPO/PPO custom generate function.

    Called by slime's rollout loop for each sample. Produces one or more
    ``Sample`` objects with ``rollout_log_probs`` and scalar ``reward`` set.

    Args:
        args: Slime training arguments.
        sample: Input sample with ``prompt`` and ``metadata``.
        sampling_params: SGLang sampling overrides.
        evaluation: If True, generate eval-only.

    Returns:
        ``list[Sample]`` with reward and logprobs ready for GRPO/PPO training.
    """
    # 1. Resolve benchmark and dataset adapter
    metadata = sample.metadata or {}
    benchmark = metadata.get("benchmark", "")
    if not benchmark:
        benchmark = _auto_detect_benchmark(sample)
        metadata["benchmark"] = benchmark

    from examples.agentic_rl_datasets import get_adapter, import_all

    import_all()
    dataset_adapter = get_adapter(benchmark)

    # 2. Route by task type
    if benchmark in _SWE_TASK_TYPES:
        return await _generate_swe(
            args, sample, sampling_params, dataset_adapter,
            metadata, benchmark, evaluation,
        )
    else:
        return await _generate_general(
            args, sample, sampling_params, dataset_adapter,
            metadata, benchmark, evaluation,
        )


# =============================================================================
# General task generation (Hermes harness + OpenAIAdapter)
# =============================================================================


async def _generate_general(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any] | None,
    dataset_adapter: Any,
    metadata: dict[str, Any],
    task_type: str,
    evaluation: bool,
) -> list[Sample]:
    """Generate a trajectory for general tasks using the Hermes harness.

    Flow:
      1. Get/create the OpenAI adapter service.
      2. Boot E2B sandbox with Hermes CLI installed.
      3. Let the dataset adapter set up the task environment.
      4. Run Hermes (calls back to SGLang via adapter → logprobs captured).
      5. Evaluate via adapter.evaluate_task().
      6. Compute multi-dimensional reward (RM + verifier, 7 dims).
      7. finish_session() → list[Sample] with logprobs + reward.
    """
    harness_cls = _GENERAL_HARNESS_CLS
    adapter_cls = _GENERAL_ADAPTER_CLS

    state = _AdapterService.get_or_create(
        args, adapter_cls, CONFIG["openai_port"], "agentic-grpo-openai",
    )

    instance_id = metadata.get("instance_id") or metadata.get("task_id", "unknown")
    image = _resolve_sandbox_image(metadata, task_type)
    workdir = metadata.get("workdir", "/home/agent")

    session_id = sample.session_id or _make_session_id(sample, instance_id, "gen")
    sample.session_id = session_id
    state.adapter.open_session(
        session_id,
        sampling_defaults=dict(sampling_params or {}),
        max_context_tokens=state.max_context_len,
    )

    t0 = time.time()
    try:
        async with asyncio.timeout(CONFIG["rollout_guard_sec"]):
            async with boot_agent_sandbox(image, instance_id, harness_cls) as sb:
                # 1. Setup task environment (dataset-specific)
                logger.info(
                    "[agentic_grpo] Setting up general task %s/%s",
                    task_type, instance_id,
                )
                await dataset_adapter.setup_task(sb, metadata)

                # 2. Run Hermes agent
                logger.info(
                    "[agentic_grpo] Running Hermes for %s/%s",
                    task_type, instance_id,
                )
                agent_prompt = metadata.get(
                    "agent_prompt",
                    sample.prompt if isinstance(sample.prompt, str) else "",
                )
                agent_exit_code = await harness_cls().run(
                    sb,
                    workdir=workdir,
                    session_id=session_id,
                    adapter_url=state.adapter_url,
                    time_budget_sec=CONFIG["agent_time_budget_sec"],
                    prompt=agent_prompt,
                )

            # 3. Evaluate task (rule-based)
            logger.info(
                "[agentic_grpo] Evaluating %s/%s", task_type, instance_id,
            )
            try:
                task_reward = await dataset_adapter.evaluate_task(
                    sb, metadata, timeout_sec=CONFIG["eval_timeout_sec"]
                )
            except Exception:
                logger.exception(
                    "Task evaluation failed for %s/%s", task_type, instance_id,
                )
                task_reward = 0.0

            if evaluation:
                logger.info(
                    "[agentic_grpo] %s/%s eval: reward=%.3f elapsed=%.1fs",
                    task_type, instance_id, task_reward, time.time() - t0,
                )
                return _eval_result(sample, task_reward, instance_id)

            # 4. Get training samples (logprobs captured by adapter)
            samples = await state.adapter.finish_session(
                session_id,
                base_sample=sample,
                reward=0.0,  # will be overridden
                extra_metadata={
                    "benchmark": task_type,
                    "instance_id": instance_id,
                },
            )
            if not samples:
                return _abort_result(
                    sample, "adapter_session_empty", instance_id,
                )

            # 5. Compute multi-dimensional reward
            raw_trajectory = _extract_adapter_trajectory(state, session_id)
            task_description = (
                sample.prompt if isinstance(sample.prompt, str)
                else str(sample.prompt)
            )
            from examples.agentic_rl_grpo.reward import (
                compute_multi_dimensional_reward,
            )

            breakdown = await compute_multi_dimensional_reward(
                args, raw_trajectory, task_description, task_type,
                task_eval_reward=task_reward,
            )

            # 6. Stamp reward on all samples
            for s in samples:
                s.reward = breakdown.total
                if s.metadata is None:
                    s.metadata = {}
                s.metadata["reward_dimensions"] = breakdown.to_dict()
                s.metadata["agent_exit_code"] = agent_exit_code
                s.metadata["benchmark"] = task_type

            logger.info(
                "[agentic_grpo] %s/%s: reward=%.3f task_eval=%.3f "
                "agent_exit=%d elapsed=%.1fs segments=%d",
                task_type, instance_id, breakdown.total, task_reward,
                agent_exit_code, time.time() - t0, len(samples),
            )
            return samples

    except asyncio.TimeoutError:
        logger.warning(
            "[agentic_grpo] %s/%s: wall-clock timeout after %.1fs",
            task_type, instance_id, time.time() - t0,
        )
        return _abort_result(sample, "wall_clock_timeout", instance_id)
    except Exception:
        logger.warning(
            "[agentic_grpo] %s/%s: rollout failed:\n%s",
            task_type, instance_id, traceback.format_exc(),
        )
        return _abort_result(
            sample,
            f"exception:{traceback.format_exc()[:200]}",
            instance_id,
        )
    finally:
        await state.adapter.drop_session(session_id, wait_timeout=30)


# =============================================================================
# SWE task generation (Claude Code harness + AnthropicAdapter)
# =============================================================================

_SWE_AGENT_PROMPT = os.environ.get(
    "SWE_CC_PROMPT",
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the "
    "relevant tests to verify your fix passes. Do NOT modify "
    "PROBLEM_STATEMENT.md and do NOT commit. When finished, print a one-line "
    "summary and exit.",
)


async def _generate_swe(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any] | None,
    dataset_adapter: Any,
    metadata: dict[str, Any],
    task_type: str,
    evaluation: bool,
) -> list[Sample]:
    """Generate a trajectory for SWE tasks using Claude Code harness.

    SWE tasks use **only** the task evaluation reward (test pass rate).
    Multi-dimensional reward is NOT applied — that is reserved for general
    tasks.  The user will register SWE-specific reward rules later.
    """
    harness_cls = _SWE_HARNESS_CLS
    adapter_cls = _SWE_ADAPTER_CLS

    state = _AdapterService.get_or_create(
        args, adapter_cls, CONFIG["anthropic_port"],
        "agentic-grpo-anthropic",
    )

    instance_id = metadata.get("instance_id", "unknown")
    image = metadata.get("image", "")
    workdir = metadata.get("workdir", "/testbed")

    if not image:
        logger.warning("[agentic_grpo] %s: no image, aborting", instance_id)
        return _abort_result(sample, "missing_image", instance_id)

    reason = _check_evaluability(metadata)
    if reason:
        return _abort_result(sample, f"unevaluatable:{reason}", instance_id)

    session_id = sample.session_id or _make_session_id(sample, instance_id, "swe")
    sample.session_id = session_id
    state.adapter.open_session(
        session_id,
        sampling_defaults=dict(sampling_params or {}),
        max_context_tokens=state.max_context_len,
    )

    t0 = time.time()
    try:
        async with asyncio.timeout(CONFIG["rollout_guard_sec"]):
            async with boot_agent_sandbox(image, instance_id, harness_cls) as sb:
                await _prepare_swe_workspace(sb, metadata, task_type)

                agent_exit_code = await harness_cls().run(
                    sb,
                    workdir=workdir,
                    session_id=session_id,
                    adapter_url=state.adapter_url,
                    time_budget_sec=CONFIG["agent_time_budget_sec"],
                    prompt=_SWE_AGENT_PROMPT,
                )

                diff_text = await _git_diff(sb, workdir)

            # Evaluate (test pass rate only — no multi-dim reward for SWE yet)
            task_reward, applied_cleanly = await _run_swe_evaluation(
                metadata, diff_text, task_type,
            )

            if evaluation:
                logger.info(
                    "[agentic_grpo] SWE %s eval: reward=%.3f elapsed=%.1fs",
                    instance_id, task_reward, time.time() - t0,
                )
                return _eval_result(sample, task_reward, instance_id)

            samples = await state.adapter.finish_session(
                session_id,
                base_sample=sample,
                reward=float(task_reward),
                extra_metadata={
                    "grading_solved": float(task_reward) == 1.0,
                    "instance_id": instance_id,
                    "benchmark": task_type,
                },
            )

            if not samples:
                return _abort_result(
                    sample, "adapter_session_empty", instance_id,
                )

            for s in samples:
                s.reward = float(task_reward)
                if s.metadata is None:
                    s.metadata = {}
                s.metadata["agent_exit_code"] = agent_exit_code

            logger.info(
                "[agentic_grpo] SWE %s: reward=%.3f applied=%s "
                "agent_exit=%d elapsed=%.1fs segments=%d",
                instance_id, task_reward, applied_cleanly,
                agent_exit_code, time.time() - t0, len(samples),
            )
            return samples

    except asyncio.TimeoutError:
        logger.warning(
            "[agentic_grpo] SWE %s: wall-clock timeout after %.1fs",
            instance_id, time.time() - t0,
        )
        return _abort_result(sample, "wall_clock_timeout", instance_id)
    except Exception:
        logger.warning(
            "[agentic_grpo] SWE %s: rollout failed:\n%s",
            instance_id, traceback.format_exc(),
        )
        return _abort_result(
            sample, f"exception:{traceback.format_exc()[:200]}", instance_id,
        )
    finally:
        await state.adapter.drop_session(session_id, wait_timeout=30)


# =============================================================================
# Input validation (defense-in-depth for shell commands)
# =============================================================================

_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")
_COMMIT_PATTERN = re.compile(r"^[a-f0-9]{7,40}$")
_ALLOWED_WORKDIR_PREFIXES = ("/testbed", "/workspace", "/home/agent", "/tmp/")


def _is_under_prefix(path: str, prefix: str) -> bool:
    """Check that *path* is exactly *prefix* or a descendant of it."""
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def _validate_repo(repo: str) -> str:
    repo = repo.strip()
    if not _REPO_PATTERN.match(repo):
        raise ValueError(
            f"Invalid repo name {repo!r}; "
            f"expected owner/repo (alphanumeric, dots, dashes, underscores)"
        )
    return repo


def _validate_base_commit(commit: str) -> str:
    commit = commit.strip()
    if commit and not _COMMIT_PATTERN.match(commit):
        raise ValueError(
            f"Invalid base_commit {commit!r}; expected 7-40 hex characters"
        )
    return commit


def _validate_workdir(workdir: str) -> str:
    """Validate workdir is within an allowed prefix after path normalization.

    Resolves ``..`` components via ``os.path.normpath`` and rejects paths
    that escape the allowed prefix set or contain shell metacharacters.
    """
    workdir = workdir.strip()
    # Resolve '..' traversal attempts
    resolved = os.path.normpath(workdir)
    if not any(_is_under_prefix(resolved, p) for p in _ALLOWED_WORKDIR_PREFIXES):
        raise ValueError(
            f"Workdir {workdir!r} (resolved: {resolved!r}) is not within "
            f"allowed prefixes: {_ALLOWED_WORKDIR_PREFIXES}"
        )
    if re.search(r"[;&|`$(){}\[\]]", resolved):
        raise ValueError(
            f"Workdir {resolved!r} contains shell metacharacters"
        )
    return resolved


# =============================================================================
# SWE workspace preparation
# =============================================================================


async def _prepare_swe_workspace(
    sb: Any, metadata: dict[str, Any], task_type: str
) -> None:
    """Clone repo, checkout base commit, apply test patch, write problem statement."""
    from examples.coding_agent_rl import swe
    from slime.agent.sandbox import ensure_agent_user

    class _FakeSample:
        prompt = metadata.get("problem_statement", "")
        metadata = metadata
        label = metadata.get("instance_id")

    protocol = swe.PROTOCOL_SCALESWE
    md = swe.get_metadata(_FakeSample, protocol)  # type: ignore[arg-type]
    if md.get("looks_swebench") or metadata.get("remote_env_info", {}).get(
        "test_patch"
    ):
        protocol = swe.PROTOCOL_SWEBENCH
        md = swe.get_metadata(_FakeSample, protocol)  # type: ignore[arg-type]

    workdir = _validate_workdir(md.get("workdir", "/testbed"))
    await ensure_agent_user(sb, workdir)

    if md.get("protocol") == swe.PROTOCOL_SWEBENCH:
        inst = md.get("grading", {}).get("sweb_instance") or {}
        repo_name = _validate_repo(inst.get("repo", ""))
        repo_url = f"https://github.com/{repo_name}.git"
        await sb.exec(
            f"git clone {shlex.quote(repo_url)} {shlex.quote(workdir)} "
            f"2>/dev/null || true",
            user="root", check=False, timeout=120,
        )
        base_commit = _validate_base_commit(inst.get("base_commit", ""))
        if base_commit:
            await sb.exec(
                f"cd {shlex.quote(workdir)} && "
                f"git config --global --add safe.directory {shlex.quote(workdir)} && "
                f"git checkout {shlex.quote(base_commit)} -f",
                user="root", check=False, timeout=60,
            )
        await sb.write_file(
            f"{workdir}/PROBLEM_STATEMENT.md",
            inst.get("problem_statement", ""),
            user="root",
        )
    else:
        await swe.prepare_workspace(sb, workdir, md)


def _check_evaluability(metadata: dict[str, Any]) -> str | None:
    """Check SWE task evaluability. Returns reason string or None."""
    from examples.coding_agent_rl import swe

    class _FakeSample:
        prompt = metadata.get("problem_statement", "")
        metadata = metadata
        label = metadata.get("instance_id")

    md = swe.get_metadata(_FakeSample, swe.PROTOCOL_SCALESWE)  # type: ignore[arg-type]
    if md.get("looks_swebench") or metadata.get("remote_env_info", {}).get(
        "test_patch"
    ):
        md = swe.get_metadata(_FakeSample, swe.PROTOCOL_SWEBENCH)  # type: ignore[arg-type]
    return swe.evaluability_check(md)


async def _git_diff(sb: Any, workdir: str) -> str:
    ec, stdout, stderr = await sb.exec(
        f"cd {shlex.quote(workdir)} && git diff --no-color 2>&1 || true",
        user="root", check=False, timeout=60,
    )
    return (stdout or "") + (stderr or "")


async def _run_swe_evaluation(
    metadata: dict[str, Any], diff_text: str, task_type: str
) -> tuple[float, bool]:
    """Return (reward, applied_cleanly) from SWE grading."""
    from examples.coding_agent_rl import swe

    class _FakeSample:
        prompt = metadata.get("problem_statement", "")
        metadata = metadata
        label = metadata.get("instance_id")

    protocol = swe.PROTOCOL_SCALESWE
    md = swe.get_metadata(_FakeSample, protocol)  # type: ignore[arg-type]
    if md.get("looks_swebench") or metadata.get("remote_env_info", {}).get(
        "test_patch"
    ):
        protocol = swe.PROTOCOL_SWEBENCH
        md = swe.get_metadata(_FakeSample, protocol)  # type: ignore[arg-type]

    result = await swe.run_evaluation(
        md, diff_text=diff_text, timeout_sec=CONFIG["eval_timeout_sec"],
    )
    return float(result.reward), bool(result.applied_cleanly)


# =============================================================================
# Trajectory extraction from adapter sessions (for reward model)
# =============================================================================


def _extract_adapter_trajectory(
    state: _AdapterService, session_id: str
) -> list[dict[str, Any]]:
    """Extract conversation messages from the adapter's internal session tree."""
    try:
        manager = state.adapter.manager
        tree = manager._trees.get(session_id) if hasattr(manager, "_trees") else None
        if tree is None:
            return []

        messages: list[dict[str, Any]] = []
        for node in _traverse_tree(tree):
            if hasattr(node, "prompt_messages") and node.prompt_messages:
                for msg in node.prompt_messages:
                    messages.append({
                        "role": msg.get("role", ""),
                        "content": msg.get("content", ""),
                        "turn": getattr(node, "turn_idx", len(messages)),
                    })
            if hasattr(node, "response_message") and node.response_message:
                messages.append({
                    "role": node.response_message.get("role", "assistant"),
                    "content": node.response_message.get("content", ""),
                    "turn": getattr(node, "turn_idx", len(messages)),
                    "finish_reason": getattr(node, "finish_reason", ""),
                })
        return messages
    except Exception:
        logger.debug("Could not extract adapter trajectory", exc_info=True)
        return []


def _traverse_tree(root: Any) -> list[Any]:
    nodes: list[Any] = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        nodes.append(node)
        if hasattr(node, "children"):
            queue.extend(node.children)
    return nodes


# =============================================================================
# Sandbox image resolution
# =============================================================================


def _resolve_sandbox_image(metadata: dict[str, Any], task_type: str) -> str:
    """Resolve the E2B sandbox image for a task.

    Priority:
      1. ``metadata["image"]`` — explicit image
      2. ``SLIME_E2B_SANDBOX_IMAGE`` env var — global default
      3. Task-type-specific default

    Args:
        metadata: Task metadata (may contain ``image``).
        task_type: Benchmark name.

    Returns:
        E2B Docker image name.

    Raises:
        RuntimeError: If no image can be resolved.
    """
    # Explicit image in metadata
    image = metadata.get("image", "")
    if image:
        return image

    # Global fallback via env
    image = os.environ.get("SLIME_E2B_SANDBOX_IMAGE", "")
    if image:
        return image

    raise RuntimeError(
        f"No sandbox image configured for task {task_type!r}. "
        f"Set 'image' in dataset metadata or export SLIME_E2B_SANDBOX_IMAGE."
    )


# =============================================================================
# Helpers
# =============================================================================


def _make_session_id(sample: Sample, instance_id: str, prefix: str) -> str:
    """Generate a unique session ID for the adapter."""
    if sample.session_id:
        return sample.session_id
    if sample.index is not None and sample.group_index is not None:
        return f"{prefix}-{instance_id}-{sample.index}-{sample.group_index}"
    return f"{prefix}-{instance_id}-{secrets.token_hex(8)}"


def _auto_detect_benchmark(sample: Sample) -> str:
    """Guess the benchmark from metadata fields."""
    md = sample.metadata or {}
    if md.get("instance_id") or md.get("repo"):
        if md.get("FAIL_TO_PASS") is not None or (
            isinstance(md.get("test_patch"), str)
            and md.get("test_patch", "").startswith("diff")
        ):
            return "r2e_gym"
        return "swe_gym_lite"
    if md.get("task_type") in (
        "os", "db", "kg", "dcg", "ltp", "hh", "ws", "wb",
    ):
        return "agent_bench"
    if md.get("env") in ("retail", "airline"):
        return "tau_bench"
    if md.get("check_command") is not None:
        return "terminal_bench"
    if md.get("check_script") is not None:
        return "cli_gym"
    if md.get("api_spec"):
        return "api_bank"
    logger.warning("Could not auto-detect benchmark from metadata=%s", md)
    return "terminal_bench"


def _abort_result(
    sample: Sample, reason: str, instance_id: str,
) -> list[Sample]:
    """Return an aborted sample when rollout fails."""
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED
    sample.metadata = {
        **(sample.metadata or {}),
        "abort_reason": reason,
        "instance_id": instance_id,
    }
    logger.warning("[agentic_grpo] %s aborted: %s", instance_id, reason)
    return [sample]


def _eval_result(
    sample: Sample, reward: float, instance_id: str,
) -> list[Sample]:
    """Return an eval-only result (no training samples)."""
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = float(reward)
    sample.remove_sample = True
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        **(sample.metadata or {}),
        "instance_id": instance_id,
        "grading_solved": float(reward) == 1.0,
    }
    return [sample]
