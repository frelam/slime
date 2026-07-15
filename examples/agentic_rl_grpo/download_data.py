#!/usr/bin/env python3
"""Download and prepare datasets for agentic RL GRPO training.

Downloads public benchmark datasets, extracts **prompts only** (not full
trajectories — those are generated during rollout), preserves environment
metadata for SWE-agent tasks, and outputs unified JSONL files compatible with
``examples/agentic_rl_grpo/run.sh``.

Usage::

    # Download everything (needs HF token for gated datasets):
    python examples/agentic_rl_grpo/download_data.py --output-dir /root/datasets/mixed_agentic_rl

    # Download specific benchmarks:
    python examples/agentic_rl_grpo/download_data.py -o ./data --benchmarks swe_gym_lite,r2e_gym

    # Dry-run (show what would be downloaded, no actual download):
    python examples/agentic_rl_grpo/download_data.py -o ./data --dry-run

    # Limit samples per benchmark:
    python examples/agentic_rl_grpo/download_data.py -o ./data --max-samples 100

    # Skip synthetic data generation:
    python examples/agentic_rl_grpo/download_data.py -o ./data --no-synthetic

    # Use HF mirror (for users behind firewalls / in China):
    python examples/agentic_rl_grpo/download_data.py -o ./data --hf-mirror hf-mirror.com

    # Use locally downloaded files (skip HF entirely):
    python examples/agentic_rl_grpo/download_data.py -o ./data \\
        --swe-input /path/to/swe_bench_verified.jsonl \\
        --r2e-input /path/to/r2e_gym.jsonl

Requirements::

    pip install datasets huggingface_hub pyyaml

Output files::

    {output_dir}/
    ├── swe_gym_lite.jsonl       # SWE tasks (SWE-bench Verified + SWE-Gym Lite)
    ├── r2e_gym.jsonl            # R2E-Gym coding tasks
    ├── terminal_bench.jsonl     # Terminal-Bench shell tasks
    ├── tau_bench.jsonl          # τ-bench task indices
    ├── cli_gym.jsonl            # CLI-Gym synthetic tasks (if --no-synthetic not set)
    ├── api_bank.jsonl           # API-Bank synthetic tasks (if --no-synthetic not set)
    ├── agent_bench.jsonl        # AgentBench tasks (requires --agent-bench-input)
    └── mixed_agentic_rl.jsonl   # All benchmarks merged

Data sources
------------

===========================  ===================================================  ==========
Benchmark                    Source                                              Auto-download?
===========================  ===================================================  ==========
swe_gym_lite (SWE-bench)     ``princeton-nlp/SWE-bench_Verified`` (HF)           Yes
swe_gym_lite (SWE-Gym)       ``SWE-Gym/SWE-Gym-Lite`` (HF)                       Yes
r2e_gym                      ``R2E-Gym/R2E-Gym-Lite`` (HF)                       Yes
terminal_bench               ``ia03/terminal-bench`` (HF) + synthetic fallback    Yes
tau_bench                    τ-bench ``train`` split task indices                 Yes (indices)
cli_gym                      Synthetic templates (no public dataset)              Synthetic
api_bank                     Synthetic templates (no public dataset)              Synthetic
agent_bench                  User-provided JSONL (``--agent-bench-input``)        Manual
===========================  ===================================================  ==========

SWE task metadata preservation
------------------------------

For SWE tasks, the following environment information is preserved from the
original datasets so the rollout harness can set up the sandbox correctly:

- ``instance_id``: Unique task identifier
- ``repo``: GitHub repository (owner/name)
- ``base_commit``: Base commit to check out
- ``problem_statement``: Task description (also used as ``prompt``)
- ``FAIL_TO_PASS`` / ``PASS_TO_PASS``: Test lists for grading
- ``test_patch``: Patch adding failing tests
- ``image``: Docker image name (derived from instance_id when missing)
- ``workdir``: Working directory inside sandbox
- ``version``: Repository version (SWE-bench)
- ``hints_text``: Optional hints

The full SWE-bench grading payload is carried under ``metadata.remote_env_info``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

_HF_SWEBENCH_VERIFIED = "princeton-nlp/SWE-bench_Verified"
_HF_SWE_GYM_LITE = "SWE-Gym/SWE-Gym-Lite"
_HF_R2E_GYM_LITE = "R2E-Gym/R2E-Gym-Lite"
_HF_TERMINAL_BENCH = "ia03/terminal-bench"

_SWEBENCH_SPLIT = "test"  # SWE-bench Verified has a single 'test' split
_SWE_GYM_LITE_SPLIT = "train"
_R2E_GYM_LITE_SPLIT = "train"
_TERMINAL_BENCH_SPLIT = "test"

# Default number of synthetic tasks per benchmark
_DEFAULT_NUM_SYNTHETIC = 64


# =============================================================================
# HuggingFace helpers
# =============================================================================

# Known HF mirrors (set via --hf-mirror or HF_ENDPOINT env var)
_HF_MIRRORS = {
    "hf-mirror.com": "https://hf-mirror.com",
}


def _setup_hf_endpoint(mirror: str | None = None) -> None:
    """Configure the HuggingFace endpoint BEFORE any HF imports.

    Must be called before ``_ensure_datasets()`` or ``_maybe_login_hf()``.
    HF libraries read ``HF_ENDPOINT`` at import time, so we set it in
    ``os.environ`` early.

    Args:
        mirror: One of the known mirror keys (e.g. ``"hf-mirror.com"``)
                or a full URL. If None, checks ``HF_ENDPOINT`` env var.
    """
    endpoint = None

    if mirror:
        # Check if it's a known shortcut
        endpoint = _HF_MIRRORS.get(mirror, mirror)
    else:
        endpoint = os.environ.get("HF_ENDPOINT", "")

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        # Also set for the datasets library (some versions read this instead)
        os.environ.setdefault("HF_DATASETS_ENDPOINT", endpoint)
        logger.info("HF endpoint: %s", endpoint)
    else:
        logger.info(
            "HF endpoint: https://huggingface.co (default). "
            "If you're behind a firewall, set --hf-mirror hf-mirror.com "
            "or export HF_ENDPOINT=https://hf-mirror.com"
        )


def _ensure_datasets():
    """Lazy-import ``datasets`` with a helpful error message."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]

        return load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' library is required. Install it with:\n"
            "  pip install datasets"
        )
        sys.exit(1)


def _maybe_login_hf():
    """Log in to HuggingFace if a token is available (for gated datasets)."""
    try:
        from huggingface_hub import HfFolder
    except ImportError:
        return  # huggingface_hub not installed; skip

    token = HfFolder.get_token()
    if token:
        logger.info("Using HF token from cache")
    else:
        logger.info(
            "No HF token found. If a dataset requires authentication, set "
            "HF_TOKEN or run `huggingface-cli login`."
        )


# =============================================================================
# Benchmark downloaders
# =============================================================================


def download_swe_gym_lite(
    output_dir: Path,
    max_samples: int | None = None,
    dry_run: bool = False,
    local_input: str | None = None,
) -> Path | None:
    """Download SWE-bench Verified + SWE-Gym Lite, output unified JSONL.

    Args:
        local_input: If provided, load from this local JSONL file instead of
            downloading from HuggingFace. The file should be in SWE-bench
            or scaleswe format (auto-detected).

    Returns the output file path or None if nothing was downloaded.
    """
    tasks: list[dict[str, Any]] = []

    # ---- Local input (skip HF download entirely) ----
    if local_input:
        logger.info("Loading SWE data from local file: %s", local_input)
        tasks = _load_local_swe_data(local_input)
        if not tasks:
            logger.warning("No tasks loaded from %s", local_input)
            return None
        tasks = _deduplicate_by(tasks, key=lambda t: t["metadata"]["instance_id"])
        if max_samples:
            tasks = tasks[:max_samples]
        output_path = output_dir / "swe_gym_lite.jsonl"
        if not dry_run:
            _write_jsonl(output_path, tasks)
        logger.info("  Wrote %d SWE tasks → %s", len(tasks), output_path)
        return output_path

    # ---- HF download path ----
    load_dataset = _ensure_datasets()

    # ---- SWE-bench Verified ----
    logger.info("Downloading SWE-bench Verified from HF...")
    try:
        if dry_run:
            logger.info(
                "  [dry-run] Would download %s (split=%s)",
                _HF_SWEBENCH_VERIFIED, _SWEBENCH_SPLIT,
            )
        else:
            ds = load_dataset(_HF_SWEBENCH_VERIFIED, split=_SWEBENCH_SPLIT)
            for row in ds:
                instance_id = row.get("instance_id", "unknown")
                repo = row.get("repo", "")
                problem_statement = row.get("problem_statement", "")

                metadata = {
                    "benchmark": "swe_gym_lite",
                    "instance_id": instance_id,
                    "repo": repo,
                    "base_commit": row.get("base_commit", ""),
                    "problem_statement": problem_statement,
                    "FAIL_TO_PASS": _parse_json_list(row.get("FAIL_TO_PASS", "[]")),
                    "PASS_TO_PASS": _parse_json_list(row.get("PASS_TO_PASS", "[]")),
                    "test_patch": row.get("test_patch", ""),
                    "version": row.get("version", ""),
                    "hints_text": row.get("hints_text", ""),
                    "workdir": "/testbed",
                    # Build image from repo name (SWE-bench convention)
                    "image": _swebench_image(repo),
                    # Carry full remote_env_info for the swebench protocol
                    "remote_env_info": {
                        "instance_id": instance_id,
                        "repo": repo,
                        "base_commit": row.get("base_commit", ""),
                        "problem_statement": problem_statement,
                        "test_patch": row.get("test_patch", ""),
                        "FAIL_TO_PASS": _parse_json_list(
                            row.get("FAIL_TO_PASS", "[]")
                        ),
                        "PASS_TO_PASS": _parse_json_list(
                            row.get("PASS_TO_PASS", "[]")
                        ),
                        "version": row.get("version", ""),
                        "image": _swebench_image(repo),
                        "workdir": "/testbed",
                    },
                }

                tasks.append({
                    "prompt": problem_statement,
                    "label": instance_id,
                    "metadata": metadata,
                })
            logger.info(
                "  Downloaded %d SWE-bench Verified tasks", len(tasks),
            )
    except Exception as e:
        logger.warning(
            "  Failed to download SWE-bench Verified: %s", e,
        )

    # ---- SWE-Gym Lite ----
    logger.info("Downloading SWE-Gym Lite from HF...")
    try:
        if dry_run:
            logger.info(
                "  [dry-run] Would download %s (split=%s)",
                _HF_SWE_GYM_LITE, _SWE_GYM_LITE_SPLIT,
            )
        else:
            ds = load_dataset(_HF_SWE_GYM_LITE, split=_SWE_GYM_LITE_SPLIT)
            n_before = len(tasks)
            for row in ds:
                instance_id = row.get("instance_id", "unknown")
                problem_statement = row.get("problem_statement", "")

                metadata = {
                    "benchmark": "swe_gym_lite",
                    "instance_id": instance_id,
                    "repo": row.get("repo", ""),
                    "base_commit": row.get("base_commit", ""),
                    "problem_statement": problem_statement,
                    "FAIL_TO_PASS": _parse_json_list(
                        row.get("FAIL_TO_PASS", "[]")
                    ),
                    "PASS_TO_PASS": _parse_json_list(
                        row.get("PASS_TO_PASS", "[]")
                    ),
                    "test_patch": row.get("test_patch", ""),
                    "image": row.get("image", ""),
                    "workdir": row.get("workdir", "/testbed"),
                    "pre_commands": row.get("pre_commands", []),
                    "swepro": row.get("swepro"),
                    "eval_cmd": row.get("eval_cmd"),
                    "remote_env_info": row.get("remote_env_info", {}),
                }

                tasks.append({
                    "prompt": problem_statement,
                    "label": instance_id,
                    "metadata": metadata,
                })
            logger.info(
                "  Downloaded %d SWE-Gym Lite tasks",
                len(tasks) - n_before,
            )
    except Exception as e:
        logger.warning("  Failed to download SWE-Gym Lite: %s", e)

    if not tasks:
        logger.warning("  No SWE tasks downloaded")
        return None

    # Deduplicate by instance_id
    tasks = _deduplicate_by(tasks, key=lambda t: t["metadata"]["instance_id"])

    if max_samples:
        tasks = tasks[:max_samples]

    output_path = output_dir / "swe_gym_lite.jsonl"
    if not dry_run:
        _write_jsonl(output_path, tasks)
    logger.info("  Wrote %d SWE tasks → %s", len(tasks), output_path)
    return output_path


def download_r2e_gym(
    output_dir: Path,
    max_samples: int | None = None,
    dry_run: bool = False,
    local_input: str | None = None,
) -> Path | None:
    """Download R2E-Gym Lite from HuggingFace.

    Args:
        local_input: If provided, load from this local JSONL file (or hf:split)
            instead of downloading from the default HF dataset.
    """
    tasks: list[dict[str, Any]] = []

    # ---- Local input ----
    if local_input:
        logger.info("Loading R2E data from: %s", local_input)
        from examples.agentic_rl_datasets.r2e_gym import R2EGymSubsetAdapter
        adapter = R2EGymSubsetAdapter()
        tasks = adapter.load_dataset(local_input)
        if not tasks:
            logger.warning("No tasks loaded from %s", local_input)
            return None
        for t in tasks:
            t["metadata"]["benchmark"] = "r2e_gym"
        tasks = _deduplicate_by(tasks, key=lambda t: t["metadata"]["instance_id"])
        if max_samples:
            tasks = tasks[:max_samples]
        output_path = output_dir / "r2e_gym.jsonl"
        if not dry_run:
            _write_jsonl(output_path, tasks)
        logger.info("  Wrote %d R2E-Gym tasks → %s", len(tasks), output_path)
        return output_path

    # ---- HF download path ----
    load_dataset = _ensure_datasets()
    tasks: list[dict[str, Any]] = []

    logger.info("Downloading R2E-Gym Lite from HF...")
    try:
        if dry_run:
            logger.info(
                "  [dry-run] Would download %s (split=%s)",
                _HF_R2E_GYM_LITE, _R2E_GYM_LITE_SPLIT,
            )
        else:
            ds = load_dataset(_HF_R2E_GYM_LITE, split=_R2E_GYM_LITE_SPLIT)
            for row in ds:
                instance_id = row.get("instance_id", "unknown")
                problem_statement = row.get("problem_statement", "")

                metadata = {
                    "benchmark": "r2e_gym",
                    "instance_id": instance_id,
                    "repo": row.get("repo", ""),
                    "base_commit": row.get("base_commit", ""),
                    "problem_statement": problem_statement,
                    "FAIL_TO_PASS": _parse_json_list(
                        row.get("FAIL_TO_PASS", "[]")
                    ),
                    "PASS_TO_PASS": _parse_json_list(
                        row.get("PASS_TO_PASS", "[]")
                    ),
                    "test_patch": row.get("test_patch", ""),
                    "workdir": row.get("workdir", "/testbed"),
                    "image": row.get("image", ""),
                    "env_setup_script": row.get("env_setup_script", ""),
                    "f2p_weight": 0.7,
                    "p2p_weight": 0.3,
                    "timeout_sec": 300,
                }

                tasks.append({
                    "prompt": problem_statement,
                    "label": instance_id,
                    "metadata": metadata,
                })
            logger.info("  Downloaded %d R2E-Gym tasks", len(tasks))
    except Exception as e:
        logger.warning("  Failed to download R2E-Gym Lite: %s", e)
        return None

    if not tasks:
        return None

    tasks = _deduplicate_by(tasks, key=lambda t: t["metadata"]["instance_id"])

    if max_samples:
        tasks = tasks[:max_samples]

    output_path = output_dir / "r2e_gym.jsonl"
    if not dry_run:
        _write_jsonl(output_path, tasks)
    logger.info("  Wrote %d R2E-Gym tasks → %s", len(tasks), output_path)
    return output_path


def download_terminal_bench(
    output_dir: Path,
    max_samples: int | None = None,
    dry_run: bool = False,
    no_synthetic: bool = False,
) -> Path | None:
    """Download Terminal-Bench from HF, fall back to synthetic if unavailable.

    The HF dataset contains full environment tarballs. We extract the task
    descriptions (prompts) and evaluation metadata from the task YAML configs,
    keeping only what the rollout harness needs.
    """
    load_dataset = _ensure_datasets()
    tasks: list[dict[str, Any]] = []

    logger.info("Downloading Terminal-Bench from HF...")
    hf_ok = False
    try:
        if not dry_run:
            ds = load_dataset(_HF_TERMINAL_BENCH, split=_TERMINAL_BENCH_SPLIT)
            for row in ds:
                task_id = row.get("task_id", "unknown")
                base_desc = row.get("base_description", "")
                task_yaml_str = row.get("task_yaml", "") or ""
                tags = row.get("tags", []) or []
                category = row.get("category", "")

                # Try to parse task_yaml for evaluation info
                task_yaml = _parse_task_yaml(task_yaml_str)
                check_cmd = task_yaml.get("check_command", "")
                expected_exit_code = task_yaml.get("expected_exit_code", 0)

                metadata = {
                    "benchmark": "terminal_bench",
                    "task_id": task_id,
                    "setup_commands": task_yaml.get("setup_commands", []),
                    "check_command": check_cmd,
                    "expected_output": task_yaml.get("expected_output", ""),
                    "expected_exit_code": expected_exit_code,
                    "timeout_sec": int(
                        row.get("max_agent_timeout_sec", 120)
                    ),
                    "tags": tags,
                    "category": category,
                    "difficulty": row.get("difficulty", ""),
                }

                tasks.append({
                    "prompt": base_desc,
                    "label": task_id,
                    "metadata": metadata,
                })
            logger.info("  Downloaded %d Terminal-Bench tasks", len(tasks))
            hf_ok = True
    except Exception as e:
        logger.warning("  Failed to download Terminal-Bench: %s", e)

    # Fall back to synthetic if HF download failed and synthetic is allowed
    if not hf_ok and not no_synthetic:
        logger.info("  Generating synthetic Terminal-Bench tasks...")
        tasks = _generate_terminal_bench_tasks(
            max_samples or _DEFAULT_NUM_SYNTHETIC,
        )

    if not tasks:
        return None

    if max_samples:
        tasks = tasks[:max_samples]

    output_path = output_dir / "terminal_bench.jsonl"
    if not dry_run:
        _write_jsonl(output_path, tasks)
    logger.info("  Wrote %d Terminal-Bench tasks → %s", len(tasks), output_path)
    return output_path


def prepare_tau_bench(
    output_dir: Path,
    num_tasks: int = 100,
    dry_run: bool = False,
) -> Path:
    """Generate τ-bench task indices (τ-bench loads its own data at runtime).

    τ-bench uses its own package ``tau_bench`` to load tasks from its internal
    dataset. Each JSONL line is just an index into the τ-bench train split.
    """
    logger.info("Generating τ-bench task indices (%d tasks)...", num_tasks)

    tasks: list[dict[str, Any]] = []
    for i in range(num_tasks):
        tasks.append({
            "prompt": f"τ-bench task {i}",
            "label": str(i),
            "metadata": {
                "benchmark": "tau_bench",
                "task_index": i,
                "env": "retail",
                "task_split": "train",
                "user_strategy": "llm",
                "user_model": "gemini-2.5-flash-lite",
                "user_model_provider": "gemini",
                "agent_strategy": "tool-calling",
            },
        })

    output_path = output_dir / "tau_bench.jsonl"
    if not dry_run:
        _write_jsonl(output_path, tasks)
    logger.info("  Wrote %d τ-bench tasks → %s", len(tasks), output_path)
    return output_path


def prepare_cli_gym(
    output_dir: Path,
    num_tasks: int = _DEFAULT_NUM_SYNTHETIC,
    dry_run: bool = False,
) -> Path:
    """Generate synthetic CLI-Gym tasks.

    CLI-Gym has no public standalone dataset. We generate templated tasks
    covering common CLI operations: git, file management, text processing.
    """
    logger.info("Generating synthetic CLI-Gym tasks (%d tasks)...", num_tasks)

    tasks = _generate_cli_gym_tasks(num_tasks)

    output_path = output_dir / "cli_gym.jsonl"
    if not dry_run:
        _write_jsonl(output_path, tasks)
    logger.info("  Wrote %d CLI-Gym tasks → %s", len(tasks), output_path)
    return output_path


def prepare_api_bank(
    output_dir: Path,
    num_tasks: int = _DEFAULT_NUM_SYNTHETIC,
    dry_run: bool = False,
) -> Path:
    """Generate synthetic API-Bank tasks.

    API-Bank has no public standalone dataset. We generate templated tasks
    covering REST API operations.
    """
    logger.info("Generating synthetic API-Bank tasks (%d tasks)...", num_tasks)

    tasks = _generate_api_bank_tasks(num_tasks)

    output_path = output_dir / "api_bank.jsonl"
    if not dry_run:
        _write_jsonl(output_path, tasks)
    logger.info("  Wrote %d API-Bank tasks → %s", len(tasks), output_path)
    return output_path


def prepare_agent_bench(
    output_dir: Path,
    input_path: str | None,
    max_samples: int | None = None,
    dry_run: bool = False,
) -> Path | None:
    """Convert AgentBench JSONL to unified format.

    Requires a user-provided input file in AgentBench's native format.
    """
    if not input_path:
        logger.info(
            "Skipping agent_bench: no input file (use --agent-bench-input)"
        )
        return None

    logger.info("Loading AgentBench from %s...", input_path)

    from examples.agentic_rl_datasets.agent_bench import AgentBenchAdapter

    adapter = AgentBenchAdapter()
    tasks = adapter.load_dataset(input_path)
    if max_samples:
        tasks = tasks[:max_samples]

    for task in tasks:
        task["metadata"]["benchmark"] = "agent_bench"

    output_path = output_dir / "agent_bench.jsonl"
    if not dry_run:
        _write_jsonl(output_path, tasks)
    logger.info("  Wrote %d AgentBench tasks → %s", len(tasks), output_path)
    return output_path


# =============================================================================
# Merge
# =============================================================================


def merge_all(
    output_dir: Path,
    max_per_benchmark: int | None = None,
    dry_run: bool = False,
) -> Path:
    """Merge all per-benchmark JSONL files into one mixed dataset."""
    import glob

    all_lines: list[str] = []
    for jsonl_path in sorted(glob.glob(str(output_dir / "*.jsonl"))):
        # Skip the merged file itself
        if jsonl_path.endswith("mixed_agentic_rl.jsonl"):
            continue
        with open(jsonl_path) as f:
            for line in f:
                all_lines.append(line)

    # Apply per-benchmark cap if requested
    if max_per_benchmark:
        bench_count: dict[str, int] = {}
        filtered: list[str] = []
        for line in all_lines:
            task = json.loads(line)
            bench = task.get("metadata", {}).get("benchmark", "unknown")
            if bench_count.get(bench, 0) < max_per_benchmark:
                filtered.append(line)
                bench_count[bench] = bench_count.get(bench, 0) + 1
        all_lines = filtered

    output_path = output_dir / "mixed_agentic_rl.jsonl"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for line in all_lines:
                f.write(line)

    # Show per-benchmark counts
    counts: dict[str, int] = {}
    for line in all_lines:
        task = json.loads(line)
        bench = task.get("metadata", {}).get("benchmark", "unknown")
        counts[bench] = counts.get(bench, 0) + 1

    logger.info("Merged %d tasks → %s", len(all_lines), output_path)
    for bench, count in sorted(counts.items()):
        logger.info("  %s: %d", bench, count)

    return output_path


# =============================================================================
# Synthetic task generators
# =============================================================================


def _generate_terminal_bench_tasks(num_tasks: int) -> list[dict[str, Any]]:
    """Generate synthetic Terminal-Bench tasks."""
    templates = [
        {
            "prompt": "List all files in /etc that have a .conf extension.",
            "setup_commands": [],
            "check_command": "ls /etc/*.conf > /dev/null 2>&1",
            "expected_exit_code": 0,
        },
        {
            "prompt": "Count the number of running processes owned by the current user.",
            "setup_commands": [],
            "check_command": "ps -u $(whoami) | wc -l",
            "expected_exit_code": 0,
        },
        {
            "prompt": "Find all directories named 'logs' under /var and list their sizes.",
            "setup_commands": [
                "mkdir -p /var/log/app1 /var/log/app2 /tmp/logs"
            ],
            "check_command": "find /var -name logs -type d | wc -l",
            "expected_exit_code": 0,
        },
        {
            "prompt": "Create a file named 'hello.txt' containing 'Hello, World!' in /tmp.",
            "setup_commands": [],
            "check_command": "grep -q 'Hello, World!' /tmp/hello.txt",
            "expected_exit_code": 0,
        },
        {
            "prompt": "Extract the tar archive /tmp/test.tar.gz to /tmp/extracted/.",
            "setup_commands": [
                "mkdir -p /tmp/extracted",
                "echo 'test content' > /tmp/testfile.txt",
                "cd /tmp && tar czf test.tar.gz testfile.txt",
            ],
            "check_command": "test -f /tmp/extracted/testfile.txt",
            "expected_exit_code": 0,
        },
        {
            "prompt": "Find the top 5 largest files under /usr.",
            "setup_commands": [],
            "check_command": "find /usr -type f -exec ls -s {} + 2>/dev/null | sort -n -r | head -n 5",
            "expected_exit_code": 0,
        },
        {
            "prompt": "Rename all .txt files in /tmp to .bak in the same directory.",
            "setup_commands": [
                "touch /tmp/a.txt /tmp/b.txt /tmp/c.other"
            ],
            "check_command": "ls /tmp/*.bak 2>/dev/null | wc -l",
            "expected_exit_code": 0,
        },
        {
            "prompt": "Display the last 20 lines of the system log /var/log/syslog.",
            "setup_commands": [],
            "check_command": "tail -n 20 /var/log/syslog 2>/dev/null | wc -l || echo 20",
            "expected_exit_code": 0,
        },
    ]

    tasks: list[dict[str, Any]] = []
    for i in range(num_tasks):
        tmpl = templates[i % len(templates)]
        task_id = f"terminal_{i:04d}"
        tasks.append({
            "prompt": tmpl["prompt"],
            "label": task_id,
            "metadata": {
                "benchmark": "terminal_bench",
                "task_id": task_id,
                "setup_commands": tmpl["setup_commands"],
                "check_command": tmpl["check_command"],
                "expected_output": "",
                "expected_exit_code": tmpl["expected_exit_code"],
                "timeout_sec": 120,
            },
        })
    return tasks


def _generate_cli_gym_tasks(num_tasks: int) -> list[dict[str, Any]]:
    """Generate synthetic CLI-Gym tasks covering git, files, and text processing."""
    templates = [
        {
            "prompt": "Initialize a git repository in /home/agent/repo, create a README.md, commit it, and create a branch called 'feature/init'.",
            "setup_script": (
                "mkdir -p /home/agent/repo && cd /home/agent/repo && "
                "git init && "
                "git config user.email 'agent@test.com' && "
                "git config user.name 'Agent'"
            ),
            "check_script": (
                "cd /home/agent/repo && "
                "git branch --list 'feature/init' | grep -q 'feature/init' && "
                "git log --oneline | grep -q 'README'"
            ),
        },
        {
            "prompt": "Clone a git repository from /home/agent/source to /home/agent/dest, then check out the 'develop' branch.",
            "setup_script": (
                "mkdir -p /home/agent/source && cd /home/agent/source && "
                "git init && "
                "git config user.email 'admin@test.com' && "
                "git config user.name 'Admin' && "
                "echo '# Source' > README.md && git add . && git commit -m 'init' && "
                "git checkout -b develop && echo 'branch content' > dev.md && "
                "git add . && git commit -m 'develop' && "
                "git checkout master 2>/dev/null || git checkout main 2>/dev/null || true"
            ),
            "check_script": (
                "cd /home/agent/dest && "
                "git branch --show-current | grep -q 'develop'"
            ),
        },
        {
            "prompt": "In /home/agent/workdir, find all Python files containing the word 'TODO' and write their paths to /home/agent/workdir/todos.txt.",
            "setup_script": (
                "mkdir -p /home/agent/workdir && "
                "echo '# TODO: refactor' > /home/agent/workdir/a.py && "
                "echo 'print(1)' > /home/agent/workdir/b.py && "
                "echo 'x = 1  # TODO' > /home/agent/workdir/c.py"
            ),
            "check_script": (
                "test -f /home/agent/workdir/todos.txt && "
                "grep -q 'a.py' /home/agent/workdir/todos.txt && "
                "grep -q 'c.py' /home/agent/workdir/todos.txt"
            ),
        },
        {
            "prompt": "Replace all occurrences of 'old_domain.com' with 'new_domain.com' in all .yaml files under /home/agent/config/.",
            "setup_script": (
                "mkdir -p /home/agent/config && "
                "echo 'url: http://old_domain.com/api' > /home/agent/config/app.yaml && "
                "echo 'cdn: https://old_domain.com/static' > /home/agent/config/cdn.yaml"
            ),
            "check_script": (
                "cd /home/agent/config && "
                "! grep -r 'old_domain.com' *.yaml"
            ),
        },
        {
            "prompt": "Count the total lines of Python code (excluding blank lines and comments) in /home/agent/project/.",
            "setup_script": (
                "mkdir -p /home/agent/project && "
                "echo -e 'import os\\n\\n# comment\\nx = 1\\n' > /home/agent/project/main.py && "
                "echo -e 'def foo():\\n    # bar\\n    return 42\\n' > /home/agent/project/utils.py"
            ),
            "check_script": (
                "cd /home/agent/project && "
                "grep -rv '^\\s*$' *.py | grep -v '^\\s*#' | wc -l > /tmp/code_lines.txt"
            ),
        },
        {
            "prompt": "Sort the CSV file /home/agent/data.csv by the second column numerically and save to /home/agent/data_sorted.csv.",
            "setup_script": (
                "mkdir -p /home/agent && "
                "echo -e 'name,score\\nalice,95\\nbob,87\\ncharlie,92' > /home/agent/data.csv"
            ),
            "check_script": (
                "cd /home/agent && "
                "test -f data_sorted.csv && "
                "head -n 2 data_sorted.csv | tail -n 1 | grep -q 'bob'"
            ),
        },
    ]

    tasks: list[dict[str, Any]] = []
    for i in range(num_tasks):
        tmpl = templates[i % len(templates)]
        task_id = f"cli_{i:04d}"
        tasks.append({
            "prompt": tmpl["prompt"],
            "label": task_id,
            "metadata": {
                "benchmark": "cli_gym",
                "task_id": task_id,
                "setup_script": tmpl["setup_script"],
                "check_script": tmpl["check_script"],
                "expected_state": {},
                "timeout_sec": 180,
                "workdir": "/home/agent",
            },
        })
    return tasks


def _generate_api_bank_tasks(num_tasks: int) -> list[dict[str, Any]]:
    """Generate synthetic API-Bank tasks covering REST API operations."""
    templates: list[dict[str, Any]] = [
        {
            "prompt": "Call the weather API to get the current temperature in San Francisco.",
            "api_spec": {
                "openapi": "3.0.0",
                "info": {"title": "Weather API", "version": "1.0.0"},
                "paths": {
                    "/weather": {
                        "get": {
                            "parameters": [
                                {"name": "city", "in": "query", "schema": {"type": "string"}},
                            ],
                            "responses": {"200": {"description": "OK"}},
                        }
                    }
                },
                "servers": [{"url": "https://api.example.com"}],
            },
            "expected_api_calls": [{"method": "GET", "path": "/weather"}],
        },
        {
            "prompt": "Create a new user account with username 'alice' and email 'alice@example.com'.",
            "api_spec": {
                "openapi": "3.0.0",
                "info": {"title": "User API", "version": "1.0.0"},
                "paths": {
                    "/users": {
                        "post": {
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "username": {"type": "string"},
                                                "email": {"type": "string"},
                                            },
                                            "required": ["username", "email"],
                                        }
                                    }
                                }
                            },
                            "responses": {"201": {"description": "Created"}},
                        }
                    }
                },
                "servers": [{"url": "https://api.example.com"}],
            },
            "expected_api_calls": [{"method": "POST", "path": "/users"}],
        },
        {
            "prompt": "List all orders placed in the last 7 days for customer ID 'cust_42'.",
            "api_spec": {
                "openapi": "3.0.0",
                "info": {"title": "Order API", "version": "1.0.0"},
                "paths": {
                    "/orders": {
                        "get": {
                            "parameters": [
                                {"name": "customer_id", "in": "query", "schema": {"type": "string"}},
                                {"name": "since", "in": "query", "schema": {"type": "string", "format": "date"}},
                            ],
                            "responses": {"200": {"description": "OK"}},
                        }
                    }
                },
                "servers": [{"url": "https://api.example.com"}],
            },
            "expected_api_calls": [{"method": "GET", "path": "/orders"}],
        },
        {
            "prompt": "Update the price of product 'SKU-789' to $29.99 using the product API.",
            "api_spec": {
                "openapi": "3.0.0",
                "info": {"title": "Product API", "version": "1.0.0"},
                "paths": {
                    "/products/{sku}": {
                        "put": {
                            "parameters": [
                                {"name": "sku", "in": "path", "required": True, "schema": {"type": "string"}},
                            ],
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {"price": {"type": "number"}},
                                            "required": ["price"],
                                        }
                                    }
                                }
                            },
                            "responses": {"200": {"description": "OK"}},
                        }
                    }
                },
                "servers": [{"url": "https://api.example.com"}],
            },
            "expected_api_calls": [{"method": "PUT", "path": "/products/SKU-789"}],
        },
        {
            "prompt": "Delete the expired session token 'sess_old_12345' from the auth server.",
            "api_spec": {
                "openapi": "3.0.0",
                "info": {"title": "Auth API", "version": "1.0.0"},
                "paths": {
                    "/sessions/{token}": {
                        "delete": {
                            "parameters": [
                                {"name": "token", "in": "path", "required": True, "schema": {"type": "string"}},
                            ],
                            "responses": {"204": {"description": "No Content"}},
                        }
                    }
                },
                "servers": [{"url": "https://api.example.com"}],
            },
            "expected_api_calls": [{"method": "DELETE", "path": "/sessions/sess_old_12345"}],
        },
        {
            "prompt": "Search for events tagged 'conference' happening in 'New York' during March 2026.",
            "api_spec": {
                "openapi": "3.0.0",
                "info": {"title": "Events API", "version": "1.0.0"},
                "paths": {
                    "/events/search": {
                        "get": {
                            "parameters": [
                                {"name": "tag", "in": "query", "schema": {"type": "string"}},
                                {"name": "location", "in": "query", "schema": {"type": "string"}},
                                {"name": "month", "in": "query", "schema": {"type": "string"}},
                            ],
                            "responses": {"200": {"description": "OK"}},
                        }
                    }
                },
                "servers": [{"url": "https://api.example.com"}],
            },
            "expected_api_calls": [{"method": "GET", "path": "/events/search"}],
        },
    ]

    tasks: list[dict[str, Any]] = []
    for i in range(num_tasks):
        tmpl = templates[i % len(templates)]
        task_id = f"api_{i:04d}"
        tasks.append({
            "prompt": tmpl["prompt"],
            "label": task_id,
            "metadata": {
                "benchmark": "api_bank",
                "task_id": task_id,
                "api_spec": tmpl["api_spec"],
                "setup_script": "",
                "check_script": "",
                "expected_api_calls": tmpl["expected_api_calls"],
                "timeout_sec": 120,
                "workdir": "/home/agent",
            },
        })
    return tasks


# =============================================================================
# Helpers
# =============================================================================


def _load_local_swe_data(path: str) -> list[dict[str, Any]]:
    """Load SWE tasks from a local JSONL file (SWE-bench or scaleswe format).

    Uses the same auto-detection logic as ``SWEGymLiteAdapter.load_dataset()``.
    """
    from examples.agentic_rl_datasets.swe_gym_lite import SWEGymLiteAdapter

    adapter = SWEGymLiteAdapter()
    tasks = adapter.load_dataset(path)
    for t in tasks:
        t["metadata"]["benchmark"] = "swe_gym_lite"
    return tasks


def _swebench_image(repo: str) -> str:
    """Derive SWE-bench Docker image name from repo.

    Standard SWE-bench convention: ``swebench/sweb.eval.x86_64.<repo>``
    where ``/`` in repo is replaced with ``__``.
    """
    if not repo:
        return ""
    safe = repo.replace("/", "__")
    return f"swebench/sweb.eval.x86_64.{safe}:latest"


def _parse_json_list(value: Any) -> list[Any]:
    """Parse a JSON-encoded list string or return the value as-is if already a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _parse_task_yaml(yaml_str: str) -> dict[str, Any]:
    """Parse Terminal-Bench task YAML into a dict.

    Falls back gracefully if YAML is malformed or pyyaml is missing.
    """
    if not yaml_str:
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("pyyaml not installed, skipping task YAML parsing")
        return {}
    try:
        data = yaml.safe_load(yaml_str)
        if isinstance(data, dict):
            return data
    except Exception:
        logger.debug("Failed to parse task YAML", exc_info=True)
    return {}


def _deduplicate_by(
    tasks: list[dict[str, Any]], key: Any
) -> list[dict[str, Any]]:
    """Deduplicate tasks by a key function, keeping first occurrence."""
    seen: set[Any] = set()
    result: list[dict[str, Any]] = []
    for task in tasks:
        k = key(task)
        if k not in seen:
            seen.add(k)
            result.append(task)
    if len(tasks) > len(result):
        logger.info("  Deduplicated: %d → %d tasks", len(tasks), len(result))
    return result


def _write_jsonl(path: Path, tasks: list[dict[str, Any]]) -> None:
    """Write tasks to a JSONL file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


# =============================================================================
# Data validation & preprocessing
# =============================================================================

# Required fields per benchmark (beyond the universal prompt + label + metadata)
_REQUIRED_METADATA: dict[str, list[str]] = {
    "swe_gym_lite": ["instance_id"],
    "r2e_gym": ["instance_id"],
    "terminal_bench": ["task_id"],
    "tau_bench": ["task_index", "env"],
    "cli_gym": ["task_id"],
    "api_bank": ["task_id"],
    "agent_bench": ["task_type", "task_id"],
}

# Fields that suggest SWE evaluability (should NOT be empty for SWE tasks)
_SWE_CRITICAL_FIELDS = [
    "repo", "base_commit", "FAIL_TO_PASS", "PASS_TO_PASS", "test_patch",
]


def validate_tasks(tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
    """Validate and clean tasks, returning (valid_tasks, report).

    Checks:
    - prompt is non-empty string
    - metadata.benchmark is present and recognized
    - Per-benchmark required metadata fields are present
    - SWE tasks have critical grading fields (warn, don't drop)
    - Deduplicates by (prompt, benchmark) to catch near-duplicates

    Returns:
        (valid_tasks, report_dict)
    """
    report = {
        "total": len(tasks),
        "valid": 0,
        "dropped_empty_prompt": 0,
        "dropped_no_benchmark": 0,
        "dropped_missing_fields": 0,
        "warned_swe_missing_critical": 0,
        "warned_duplicates": 0,
        "by_benchmark": {},  # benchmark → count
        "issues": [],  # list of (severity, benchmark, instance_id, message)
    }

    valid: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()  # (prompt[:200], benchmark)

    for task in tasks:
        prompt = task.get("prompt", "")
        metadata = task.get("metadata") or {}
        benchmark = metadata.get("benchmark", "")
        label = task.get("label", "?")

        # 1. Non-empty prompt
        if not prompt or not str(prompt).strip():
            report["dropped_empty_prompt"] += 1
            report["issues"].append((
                "ERROR", benchmark or "unknown", str(label),
                "Empty prompt — dropped",
            ))
            continue

        # 2. Must have a known benchmark
        if not benchmark:
            report["dropped_no_benchmark"] += 1
            report["issues"].append((
                "ERROR", "unknown", str(label),
                "Missing metadata.benchmark — dropped",
            ))
            continue

        # 3. Check required metadata fields
        required = _REQUIRED_METADATA.get(benchmark, [])
        missing = [f for f in required if f not in metadata or metadata[f] is None]
        if missing:
            report["dropped_missing_fields"] += 1
            report["issues"].append((
                "ERROR", benchmark, str(label),
                f"Missing required metadata fields: {missing} — dropped",
            ))
            continue

        # 4. SWE-specific: warn on missing critical grading fields
        if benchmark in ("swe_gym_lite", "r2e_gym"):
            critical_missing = [
                f for f in _SWE_CRITICAL_FIELDS
                if not metadata.get(f) and not metadata.get("remote_env_info", {}).get(f)
            ]
            if critical_missing:
                report["warned_swe_missing_critical"] += 1
                report["issues"].append((
                    "WARN", benchmark, str(label),
                    f"SWE task missing critical grading fields: {critical_missing} "
                    f"— reward may be unreliable",
                ))

        # 5. Dedup by prompt prefix + benchmark
        key = (str(prompt)[:200], benchmark)
        if key in seen:
            report["warned_duplicates"] += 1
            report["issues"].append((
                "WARN", benchmark, str(label),
                "Near-duplicate prompt — skipped",
            ))
            continue
        seen.add(key)

        valid.append(task)
        report["valid"] += 1
        report["by_benchmark"][benchmark] = (
            report["by_benchmark"].get(benchmark, 0) + 1
        )

    return valid, report


def print_validation_report(report: dict) -> None:
    """Print a human-readable validation report."""
    logger.info("=" * 60)
    logger.info("Data Validation Report")
    logger.info("=" * 60)
    logger.info("  Total input:      %5d", report["total"])
    logger.info("  Valid:            %5d", report["valid"])
    logger.info("  Dropped:")
    if report["dropped_empty_prompt"]:
        logger.info("    empty prompt:   %5d", report["dropped_empty_prompt"])
    if report["dropped_no_benchmark"]:
        logger.info("    no benchmark:   %5d", report["dropped_no_benchmark"])
    if report["dropped_missing_fields"]:
        logger.info("    missing fields: %5d", report["dropped_missing_fields"])
    if report["warned_swe_missing_critical"]:
        logger.info("  ⚠ SWE missing critical fields: %d", report["warned_swe_missing_critical"])
    if report["warned_duplicates"]:
        logger.info("  ⚠ Near-duplicates skipped: %d", report["warned_duplicates"])

    logger.info("  By benchmark:")
    for bench, count in sorted(report["by_benchmark"].items()):
        logger.info("    %-20s %5d", bench, count)

    # Print first few issues
    if report["issues"]:
        logger.info("  Issues (first 10):")
        for severity, bench, label, msg in report["issues"][:10]:
            logger.info("    [%s] %s/%s: %s", severity, bench, label, msg)
        if len(report["issues"]) > 10:
            logger.info("    ... and %d more", len(report["issues"]) - 10)

    logger.info("=" * 60)


def preprocess_with_tokenizer(
    tasks: list[dict[str, Any]],
    tokenizer_path: str,
    max_prompt_len: int | None = None,
) -> list[dict[str, Any]]:
    """Apply chat template and optionally filter by token length.

    Args:
        tasks: List of task dicts with ``prompt`` as raw string.
        tokenizer_path: Path to HF checkpoint (same as --hf-checkpoint).
        max_prompt_len: If set, drop tasks whose tokenized prompt exceeds this.

    Returns:
        Filtered tasks with chat-template-applied prompts.
    """
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer from %s ...", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    lengths: list[int] = []
    valid: list[dict[str, Any]] = []
    dropped_long = 0

    for task in tasks:
        prompt = task.get("prompt", "")

        # Convert to messages and apply chat template
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            logger.warning(
                "Chat template failed for %s, using raw prompt",
                task.get("label", "?"),
            )
            rendered = prompt

        # Tokenize for length check
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        n_tokens = len(token_ids)
        lengths.append(n_tokens)

        if max_prompt_len and n_tokens > max_prompt_len:
            dropped_long += 1
            continue

        task["prompt"] = rendered
        # Store token count in metadata for reference
        task.setdefault("metadata", {})["_prompt_tokens"] = n_tokens
        valid.append(task)

    # Statistics
    if lengths:
        lengths.sort()
        logger.info("Prompt token statistics (after chat template):")
        logger.info("  count:  %d", len(lengths))
        logger.info("  min:    %d", lengths[0])
        logger.info("  p25:    %d", lengths[len(lengths) // 4])
        logger.info("  median: %d", lengths[len(lengths) // 2])
        logger.info("  p75:    %d", lengths[3 * len(lengths) // 4])
        logger.info("  p95:    %d", lengths[95 * len(lengths) // 100])
        logger.info("  max:    %d", lengths[-1])

    if dropped_long:
        logger.info(
            "  Dropped >%d tokens: %d tasks", max_prompt_len, dropped_long,
        )

    return valid


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare agentic RL GRPO datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything
  python examples/agentic_rl_grpo/download_data.py -o /root/datasets/mixed_agentic_rl

  # Download just SWE tasks
  python examples/agentic_rl_grpo/download_data.py -o ./data --benchmarks swe_gym_lite

  # Dry-run to see what would be downloaded
  python examples/agentic_rl_grpo/download_data.py -o ./data --dry-run

  # No synthetic data, limited samples
  python examples/agentic_rl_grpo/download_data.py -o ./data --no-synthetic --max-samples 200

  # Use HF mirror (for users behind firewalls):
  python examples/agentic_rl_grpo/download_data.py -o ./data --hf-mirror hf-mirror.com
  # Or: export HF_ENDPOINT=https://hf-mirror.com

  # Include AgentBench with custom input
  python examples/agentic_rl_grpo/download_data.py -o ./data --agent-bench-input /path/to/agent_bench.jsonl

Use with run.sh:
  bash examples/agentic_rl_grpo/run.sh \\
      --prompt-data /root/datasets/mixed_agentic_rl/mixed_agentic_rl.jsonl \\
      --input-key prompt --label-key label --metadata-key metadata \\
      ...
        """,
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="/root/datasets/mixed_agentic_rl",
        help="Output directory for prepared JSONL files",
    )
    parser.add_argument(
        "--benchmarks",
        default="all",
        help="Comma-separated benchmark list, or 'all' (default: all). "
             "Available: swe_gym_lite, r2e_gym, terminal_bench, tau_bench, "
             "cli_gym, api_bank, agent_bench",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total samples per benchmark",
    )
    parser.add_argument(
        "--num-synthetic",
        type=int,
        default=_DEFAULT_NUM_SYNTHETIC,
        help=f"Number of synthetic tasks for benchmarks without public data "
             f"(default: {_DEFAULT_NUM_SYNTHETIC})",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Skip synthetic data generation (CLI-Gym, API-Bank)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merging all benchmarks into one file",
    )
    parser.add_argument(
        "--agent-bench-input",
        default=None,
        help="Path to AgentBench JSONL (required if agent_bench is in --benchmarks)",
    )
    parser.add_argument(
        "--swe-input",
        default=None,
        help="Local JSONL file for SWE tasks (SWE-bench or scaleswe format). "
             "When set, skips HF download for swe_gym_lite and uses this file.",
    )
    parser.add_argument(
        "--r2e-input",
        default=None,
        help="Local JSONL file (or hf:split) for R2E-Gym tasks. "
             "When set, skips HF download for r2e_gym and uses this path.",
    )
    parser.add_argument(
        "--hf-mirror",
        default=None,
        help="HF mirror to use for downloads. Known shortcuts: hf-mirror.com. "
             "Or pass a full URL. Also reads HF_ENDPOINT env var.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="HF model path for tokenizer (same as --hf-checkpoint in run.sh). "
             "When set: applies chat template, reports token-length stats. "
             "Requires: pip install transformers",
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=None,
        help="Drop prompts exceeding this token count after chat template. "
             "Requires --tokenizer-path.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip data validation (not recommended)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )
    args = parser.parse_args()

    # ---- Configure HF endpoint BEFORE any HF imports ----
    _setup_hf_endpoint(args.hf_mirror)

    # Resolve benchmark list
    all_benchmarks = [
        "swe_gym_lite", "r2e_gym", "terminal_bench", "tau_bench",
        "cli_gym", "api_bank", "agent_bench",
    ]
    if args.benchmarks == "all":
        benchmarks = all_benchmarks
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(",")]
        unknown = set(benchmarks) - set(all_benchmarks)
        if unknown:
            logger.error(
                "Unknown benchmarks: %s. Available: %s",
                ", ".join(sorted(unknown)), ", ".join(all_benchmarks),
            )
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log in to HF if possible
    _maybe_login_hf()

    paths: list[Path] = []

    # ---- Download / prepare each benchmark ----

    if "swe_gym_lite" in benchmarks:
        p = download_swe_gym_lite(
            output_dir, args.max_samples, args.dry_run,
            local_input=args.swe_input,
        )
        if p:
            paths.append(p)

    if "r2e_gym" in benchmarks:
        p = download_r2e_gym(
            output_dir, args.max_samples, args.dry_run,
            local_input=args.r2e_input,
        )
        if p:
            paths.append(p)

    if "terminal_bench" in benchmarks:
        p = download_terminal_bench(
            output_dir, args.max_samples, args.dry_run,
            no_synthetic=args.no_synthetic,
        )
        if p:
            paths.append(p)

    if "tau_bench" in benchmarks:
        p = prepare_tau_bench(
            output_dir,
            num_tasks=args.max_samples or args.num_synthetic,
            dry_run=args.dry_run,
        )
        paths.append(p)

    if "cli_gym" in benchmarks and not args.no_synthetic:
        p = prepare_cli_gym(
            output_dir,
            num_tasks=args.max_samples or args.num_synthetic,
            dry_run=args.dry_run,
        )
        paths.append(p)

    if "api_bank" in benchmarks and not args.no_synthetic:
        p = prepare_api_bank(
            output_dir,
            num_tasks=args.max_samples or args.num_synthetic,
            dry_run=args.dry_run,
        )
        paths.append(p)

    if "agent_bench" in benchmarks:
        p = prepare_agent_bench(
            output_dir, args.agent_bench_input,
            args.max_samples, args.dry_run,
        )
        if p:
            paths.append(p)

    # ---- Validate all generated files ----
    if not args.dry_run and not args.no_validate:
        for p in paths:
            if not p.exists():
                continue
            tasks = []
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tasks.append(json.loads(line))
            if not tasks:
                logger.warning("No tasks in %s, skipping validation", p.name)
                continue
            logger.info("Validating %s (%d tasks)...", p.name, len(tasks))
            valid_tasks, report = validate_tasks(tasks)
            _write_jsonl(p, valid_tasks)  # overwrite with cleaned data
            print_validation_report(report)

    # ---- Apply chat template + length filter (if tokenizer provided) ----
    if args.tokenizer_path and not args.dry_run:
        for p in paths:
            if not p.exists():
                continue
            tasks = []
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tasks.append(json.loads(line))
            if not tasks:
                continue
            logger.info("Preprocessing %s with tokenizer...", p.name)
            tasks = preprocess_with_tokenizer(
                tasks, args.tokenizer_path, args.max_prompt_len,
            )
            _write_jsonl(p, tasks)

    # ---- Merge ----
    if not args.no_merge and len(paths) >= 1 and not args.dry_run:
        merge_all(output_dir, args.max_samples)

    # ---- Summary ----
    logger.info("=" * 60)
    if args.dry_run:
        logger.info("Dry-run complete. No files were written.")
    else:
        logger.info("Done! Output directory: %s", output_dir)
        logger.info("")
        logger.info("Use with run.sh:")
        logger.info(
            "  --prompt-data %s/mixed_agentic_rl.jsonl \\",
            output_dir,
        )
        logger.info("  --input-key prompt --label-key label --metadata-key metadata \\")
        logger.info("  --apply-chat-template")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
