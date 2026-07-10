#!/usr/bin/env python3
"""Data preparation script for agentic RL training.

Converts native benchmark formats into slime's unified JSONL format.

Usage::

    # Prepare all benchmarks
    python examples/agentic_rl/prepare_data.py --output-dir /root/datasets/mixed_agentic_rl

    # Prepare a single benchmark
    python examples/agentic_rl/prepare_data.py --benchmark swe_gym_lite --input /path/to/swe.jsonl --output /root/datasets/
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def prepare_swe_gym_lite(input_path: str, output_dir: str, max_samples: int | None = None) -> str:
    """Convert SWE-Gym-Lite format to unified format."""
    from examples.agentic_rl_datasets.swe_gym_lite import SWEGymLiteAdapter

    adapter = SWEGymLiteAdapter()
    tasks = adapter.load_dataset(input_path)
    if max_samples:
        tasks = tasks[:max_samples]

    output_path = os.path.join(output_dir, "swe_gym_lite.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for task in tasks:
            task["metadata"]["benchmark"] = "swe_gym_lite"
            f.write(json.dumps(task) + "\n")

    print(f"Wrote {len(tasks)} tasks to {output_path}")
    return output_path


def prepare_tau_bench(output_dir: str, num_tasks: int = 100) -> str:
    """Generate τ-bench task indices."""
    output_path = os.path.join(output_dir, "tau_bench.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        for i in range(num_tasks):
            task = {
                "prompt": str(i),
                "label": str(i),
                "metadata": {
                    "benchmark": "tau_bench",
                    "task_index": i,
                    "env": "retail",
                    "task_split": "train",
                    "user_strategy": "llm",
                    "user_model": "gemini-2.5-flash-lite",
                    "user_model_provider": "gemini",
                },
            }
            f.write(json.dumps(task) + "\n")

    print(f"Wrote {num_tasks} τ-bench tasks to {output_path}")
    return output_path


def prepare_terminal_bench(
    output_dir: str,
    num_tasks: int = 50,
) -> str:
    """Generate Terminal-Bench tasks."""
    output_path = os.path.join(output_dir, "terminal_bench.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    sample_tasks = [
        {
            "prompt": "Find all Python files in /home that were modified in the last 24 hours.",
            "metadata": {
                "benchmark": "terminal_bench",
                "setup_commands": [],
                "check_command": "ls -la /home/",
                "expected_output": "",
                "expected_exit_code": 0,
                "task_id": "terminal_001",
                "timeout_sec": 120,
            },
        },
        {
            "prompt": "Count the number of running processes on the system.",
            "metadata": {
                "benchmark": "terminal_bench",
                "setup_commands": [],
                "check_command": "ps aux | wc -l",
                "expected_exit_code": 0,
                "task_id": "terminal_002",
                "timeout_sec": 60,
            },
        },
    ]

    with open(output_path, "w") as f:
        for i in range(num_tasks):
            template = sample_tasks[i % len(sample_tasks)]
            task = {
                "prompt": template["prompt"],
                "label": template["metadata"]["task_id"],
                "metadata": {
                    **template["metadata"],
                    "task_id": f"terminal_{i:04d}",
                },
            }
            f.write(json.dumps(task) + "\n")

    print(f"Wrote {num_tasks} Terminal-Bench tasks to {output_path}")
    return output_path


def prepare_cli_gym(output_dir: str, num_tasks: int = 50) -> str:
    """Generate CLI-Gym tasks."""
    output_path = os.path.join(output_dir, "cli_gym.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    sample_tasks = [
        {
            "prompt": "Use git to create a new branch called 'feature/logging', add a commit, and push.",
            "metadata": {
                "benchmark": "cli_gym",
                "setup_script": "cd /home/agent && git init && git config user.email test@test.com && git config user.name test && echo '# README' > README.md && git add . && git commit -m 'init'",
                "check_script": "cd /home/agent && git branch | grep -q 'feature/logging'",
                "task_id": "cli_001",
                "timeout_sec": 180,
                "workdir": "/home/agent",
            },
        },
    ]

    with open(output_path, "w") as f:
        for i in range(num_tasks):
            template = sample_tasks[i % len(sample_tasks)]
            task = {
                "prompt": template["prompt"],
                "label": f"cli_{i:04d}",
                "metadata": {
                    **template["metadata"],
                    "task_id": f"cli_{i:04d}",
                },
            }
            f.write(json.dumps(task) + "\n")

    print(f"Wrote {num_tasks} CLI-Gym tasks to {output_path}")
    return output_path


def prepare_api_bank(output_dir: str, num_tasks: int = 50) -> str:
    """Generate API-Bank tasks."""
    output_path = os.path.join(output_dir, "api_bank.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    sample_tasks = [
        {
            "prompt": "Call the weather API to get the current temperature in San Francisco.",
            "metadata": {
                "benchmark": "api_bank",
                "api_spec": {
                    "openapi": "3.0.0",
                    "info": {"title": "Weather API", "version": "1.0.0"},
                    "paths": {
                        "/weather": {
                            "get": {
                                "parameters": [
                                    {"name": "city", "in": "query", "type": "string"},
                                ],
                                "responses": {"200": {"description": "OK"}},
                            }
                        }
                    },
                    "servers": [{"url": "https://api.example.com"}],
                },
                "setup_script": "",
                "check_script": "",
                "expected_api_calls": [
                    {"method": "GET", "path": "/weather"},
                ],
                "task_id": "api_001",
                "timeout_sec": 120,
                "workdir": "/home/agent",
            },
        },
    ]

    with open(output_path, "w") as f:
        for i in range(num_tasks):
            template = sample_tasks[i % len(sample_tasks)]
            task = {
                "prompt": template["prompt"],
                "label": f"api_{i:04d}",
                "metadata": {
                    **template["metadata"],
                    "task_id": f"api_{i:04d}",
                },
            }
            f.write(json.dumps(task) + "\n")

    print(f"Wrote {num_tasks} API-Bank tasks to {output_path}")
    return output_path


def prepare_agent_bench(input_path: str, output_dir: str, max_samples: int | None = None) -> str:
    """Convert AgentBench JSONL to unified format."""
    from examples.agentic_rl_datasets.agent_bench import AgentBenchAdapter

    adapter = AgentBenchAdapter()
    tasks = adapter.load_dataset(input_path)
    if max_samples:
        tasks = tasks[:max_samples]

    output_path = os.path.join(output_dir, "agent_bench.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for task in tasks:
            task["metadata"]["benchmark"] = "agent_bench"
            f.write(json.dumps(task) + "\n")

    print(f"Wrote {len(tasks)} tasks to {output_path}")
    return output_path


def prepare_r2e_gym(input_path: str, output_dir: str, max_samples: int | None = None) -> str:
    """Convert R2E-Gym JSONL (or HF dataset) to unified format."""
    from examples.agentic_rl_datasets.r2e_gym import R2EGymSubsetAdapter

    adapter = R2EGymSubsetAdapter()
    tasks = adapter.load_dataset(input_path)
    if max_samples:
        tasks = tasks[:max_samples]

    output_path = os.path.join(output_dir, "r2e_gym.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for task in tasks:
            task["metadata"]["benchmark"] = "r2e_gym"
            f.write(json.dumps(task) + "\n")

    print(f"Wrote {len(tasks)} tasks to {output_path}")
    return output_path


def merge_all(output_dir: str, max_per_benchmark: int | None = None) -> str:
    """Merge all benchmark datasets into one."""
    import glob

    all_lines = []
    for jsonl_path in sorted(glob.glob(os.path.join(output_dir, "*.jsonl"))):
        with open(jsonl_path) as f:
            for line in f:
                all_lines.append(line)

    if max_per_benchmark:
        # Keep only max_per_benchmark per benchmark
        bench_count: dict[str, int] = {}
        filtered = []
        for line in all_lines:
            task = json.loads(line)
            bench = task.get("metadata", {}).get("benchmark", "unknown")
            if bench_count.get(bench, 0) < max_per_benchmark:
                filtered.append(line)
                bench_count[bench] = bench_count.get(bench, 0) + 1
            else:
                print(f"  Skipping {bench} (reached limit {max_per_benchmark})")
        all_lines = filtered

    output_path = os.path.join(output_dir, "mixed_agentic_rl.jsonl")
    with open(output_path, "w") as f:
        for line in all_lines:
            f.write(line)

    print(f"Merged {len(all_lines)} tasks into {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare agentic RL datasets")
    parser.add_argument("--output-dir", "-o", default="/root/datasets/mixed_agentic_rl")
    parser.add_argument(
        "--benchmark",
        choices=[
            "swe_gym_lite", "tau_bench", "terminal_bench", "cli_gym",
            "api_bank", "r2e_gym", "agent_bench", "all",
        ],
        default="all",
    )
    parser.add_argument("--input", help="Input path (for swe_gym_lite)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per benchmark")
    parser.add_argument("--num-tasks", type=int, default=50, help="Generated task count for synthetic benchmarks")
    parser.add_argument("--merge", action="store_true", default=True, help="Merge all benchmarks into one file")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    paths = []

    if args.benchmark in ("swe_gym_lite", "all"):
        if args.input:
            paths.append(prepare_swe_gym_lite(args.input, output_dir, args.max_samples))
        else:
            print("Skipping swe_gym_lite: --input not provided (use --input /path/to/swe.jsonl)", file=sys.stderr)

    if args.benchmark in ("tau_bench", "all"):
        paths.append(prepare_tau_bench(output_dir, args.num_tasks))

    if args.benchmark in ("terminal_bench", "all"):
        paths.append(prepare_terminal_bench(output_dir, args.num_tasks))

    if args.benchmark in ("cli_gym", "all"):
        paths.append(prepare_cli_gym(output_dir, args.num_tasks))

    if args.benchmark in ("api_bank", "all"):
        paths.append(prepare_api_bank(output_dir, args.num_tasks))

    if args.benchmark in ("r2e_gym", "all"):
        if args.input:
            paths.append(prepare_r2e_gym(args.input, output_dir, args.max_samples))
        else:
            print(
                "Skipping r2e_gym: --input not provided "
                "(use --input /path/to/r2e.jsonl or --input hf:train)",
                file=sys.stderr,
            )

    if args.benchmark in ("agent_bench", "all"):
        if args.input:
            paths.append(prepare_agent_bench(args.input, output_dir, args.max_samples))
        else:
            print(
                "Skipping agent_bench: --input not provided "
                "(use --input /path/to/agent_bench.jsonl)",
                file=sys.stderr,
            )

    if args.merge and len(paths) > 1:
        merge_all(output_dir, args.max_samples)


if __name__ == "__main__":
    main()
