#!/usr/bin/env python3
"""Download and prepare **simple** datasets for agentic RL GRPO training.

与 ``download_data.py`` 不同，这个脚本只处理简单数据集 —— 不需要 Docker/E2B，
使用本地 subprocess sandbox（sglang_loop 模式）。

Datasets prepared:
  - simple_shell:     Easy shell command tasks (terminal-bench style)
  - simple_math:      Math problems solved with Python tool (GSM8K-style)
  - simple_code:      Small coding problems with test-based verification
  - alfworld:         Simulated household navigation tasks
  - terminal_bench:   Existing terminal-bench (via HF or synthetic)

Usage::

    # Generate simple datasets:
    python examples/agentic_rl_grpo/download_simple_data.py -o ./data/simple

    # Specific benchmarks:
    python examples/agentic_rl_grpo/download_simple_data.py -o ./data/simple \\
        --benchmarks simple_shell,simple_math

    # Limit samples:
    python examples/agentic_rl_grpo/download_simple_data.py -o ./data/simple \\
        --max-samples 200

    # Include terminal-bench from HF:
    python examples/agentic_rl_grpo/download_simple_data.py -o ./data/simple \\
        --benchmarks terminal_bench,simple_shell

Output files::

    {output_dir}/
    ├── simple_shell.jsonl      # Shell command tasks
    ├── simple_math.jsonl       # Math problems
    ├── simple_code.jsonl       # Coding tasks
    ├── alfworld.jsonl          # Text-based household tasks
    ├── terminal_bench.jsonl    # Terminal-Bench (optional, from HF)
    └── mixed_simple_rl.jsonl   # All merged
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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

_DEFAULT_NUM_TASKS = 128


# =============================================================================
# Task generators
# =============================================================================

def generate_simple_shell_tasks(num_tasks: int = _DEFAULT_NUM_TASKS) -> list[dict[str, Any]]:
    """Generate simple shell command tasks.

    这些任务比 terminal-bench 更简单，适合作为 agent RL 的入门训练数据。
    每个任务有明确的 setup、check_command 和 outcome reward。
    """
    templates: list[dict[str, Any]] = [
        # ---- File listing & navigation ----
        {
            "prompt": (
                "List all files in the current directory that have a .txt extension. "
                "Use bash commands to explore and complete the task."
            ),
            "setup_commands": [
                "mkdir -p /home/agent/work",
                "cd /home/agent/work && touch a.txt b.txt c.log d.txt e.md",
            ],
            "check_command": (
                "cd /home/agent/work && "
                "test $(ls *.txt 2>/dev/null | wc -l) -eq 3"
            ),
            "tags": ["file_listing", "beginner"],
        },
        {
            "prompt": (
                "Count the total number of files (not directories) in /home/agent/work "
                "and write the count to /home/agent/work/file_count.txt."
            ),
            "setup_commands": [
                "mkdir -p /home/agent/work",
                "cd /home/agent/work && touch f1 f2 f3 f4 f5 && mkdir subdir",
            ],
            "check_command": (
                "cd /home/agent/work && "
                "test -f file_count.txt && "
                "grep -q '5' file_count.txt"
            ),
            "tags": ["counting", "beginner"],
        },
        {
            "prompt": (
                "Find all directories named 'logs' anywhere under /home/agent "
                "and list their full paths in /home/agent/log_dirs.txt."
            ),
            "setup_commands": [
                "mkdir -p /home/agent/app1/logs /home/agent/app2/logs "
                "/home/agent/app3/data /home/agent/tmp/logs",
            ],
            "check_command": (
                "test -f /home/agent/log_dirs.txt && "
                "test $(grep -c 'logs' /home/agent/log_dirs.txt) -ge 3"
            ),
            "tags": ["find", "beginner"],
        },

        # ---- File content ----
        {
            "prompt": (
                "Create a file /home/agent/greeting.txt with the content "
                "'Hello from agent!'. Then verify the file exists and has content."
            ),
            "setup_commands": ["mkdir -p /home/agent"],
            "check_command": (
                "grep -q 'Hello from agent!' /home/agent/greeting.txt"
            ),
            "tags": ["file_create", "beginner"],
        },
        {
            "prompt": (
                "Search all .txt files under /home/agent/docs for lines containing "
                "the word 'IMPORTANT' (case-sensitive). Write matching lines to "
                "/home/agent/important_lines.txt."
            ),
            "setup_commands": [
                "mkdir -p /home/agent/docs",
                "echo 'This is IMPORTANT info' > /home/agent/docs/a.txt",
                "echo 'nothing special' > /home/agent/docs/b.txt",
                "echo 'IMPORTANT: read this first' > /home/agent/docs/c.txt",
                "echo 'regular text' > /home/agent/docs/d.txt",
            ],
            "check_command": (
                "test -f /home/agent/important_lines.txt && "
                "test $(grep -c 'IMPORTANT' /home/agent/important_lines.txt) -eq 2"
            ),
            "tags": ["grep", "intermediate"],
        },
        {
            "prompt": (
                "Count how many lines in /home/agent/data/log.txt contain the "
                "word 'ERROR' and write the count to /home/agent/error_count.txt."
            ),
            "setup_commands": [
                "mkdir -p /home/agent/data",
                "echo -e 'INFO: start\\nERROR: failed\\nINFO: retry\\n"
                "ERROR: timeout\\nINFO: done\\nWARN: slow' > /home/agent/data/log.txt",
            ],
            "check_command": (
                "test -f /home/agent/error_count.txt && "
                "grep -q '2' /home/agent/error_count.txt"
            ),
            "tags": ["grep", "counting", "intermediate"],
        },

        # ---- Text processing ----
        {
            "prompt": (
                "The file /home/agent/names.txt contains one name per line. "
                "Sort the names alphabetically and write the sorted list to "
                "/home/agent/names_sorted.txt."
            ),
            "setup_commands": [
                "mkdir -p /home/agent",
                "echo -e 'Charlie\\nAlice\\nEve\\nBob\\nDave' > /home/agent/names.txt",
            ],
            "check_command": (
                "test -f /home/agent/names_sorted.txt && "
                "test \"$(head -1 /home/agent/names_sorted.txt)\" = 'Alice' && "
                "test \"$(tail -1 /home/agent/names_sorted.txt)\" = 'Eve'"
            ),
            "tags": ["sort", "intermediate"],
        },
        {
            "prompt": (
                "Replace all occurrences of 'old-server' with 'new-server' in "
                "/home/agent/config/app.conf and save the result. Verify no "
                "'old-server' remains."
            ),
            "setup_commands": [
                "mkdir -p /home/agent/config",
                "echo -e 'host=old-server.example.com\\n"
                "backup=old-server-backup.example.com\\n"
                "port=8080' > /home/agent/config/app.conf",
            ],
            "check_command": (
                "cd /home/agent/config && "
                "! grep -q 'old-server' app.conf && "
                "grep -q 'new-server' app.conf"
            ),
            "tags": ["sed", "replace", "intermediate"],
        },

        # ---- File operations ----
        {
            "prompt": (
                "Create a tar.gz archive of all .py files in /home/agent/src "
                "and save it to /home/agent/backup.tar.gz."
            ),
            "setup_commands": [
                "mkdir -p /home/agent/src",
                "echo 'print(1)' > /home/agent/src/main.py",
                "echo 'x = 42' > /home/agent/src/utils.py",
                "echo 'config' > /home/agent/src/readme.txt",
            ],
            "check_command": (
                "test -f /home/agent/backup.tar.gz && "
                "tar tzf /home/agent/backup.tar.gz | grep -q '.py'"
            ),
            "tags": ["tar", "archive", "intermediate"],
        },
        {
            "prompt": (
                "Rename all .jpeg files in /home/agent/photos to .jpg and "
                "list the renamed files in /home/agent/renamed.txt."
            ),
            "setup_commands": [
                "mkdir -p /home/agent/photos",
                "touch /home/agent/photos/img1.jpeg "
                "/home/agent/photos/img2.jpeg "
                "/home/agent/photos/img3.png "
                "/home/agent/photos/img4.jpeg",
            ],
            "check_command": (
                "cd /home/agent/photos && "
                "test $(ls *.jpg 2>/dev/null | wc -l) -eq 3 && "
                "test $(ls *.jpeg 2>/dev/null | wc -l) -eq 0"
            ),
            "tags": ["rename", "intermediate"],
        },

        # ---- System info ----
        {
            "prompt": (
                "Find the 3 largest files (by size) under /home/agent/data "
                "and write their paths and sizes to /home/agent/largest_files.txt."
            ),
            "setup_commands": [
                "mkdir -p /home/agent/data",
                "dd if=/dev/zero of=/home/agent/data/small.bin bs=100 count=1 2>/dev/null",
                "dd if=/dev/zero of=/home/agent/data/medium.bin bs=1000 count=1 2>/dev/null",
                "dd if=/dev/zero of=/home/agent/data/large.bin bs=5000 count=1 2>/dev/null",
                "dd if=/dev/zero of=/home/agent/data/huge.bin bs=10000 count=1 2>/dev/null",
                "dd if=/dev/zero of=/home/agent/data/tiny.bin bs=10 count=1 2>/dev/null",
            ],
            "check_command": (
                "test -f /home/agent/largest_files.txt && "
                "test $(wc -l < /home/agent/largest_files.txt) -eq 3"
            ),
            "tags": ["find", "sort", "advanced"],
        },
        {
            "prompt": (
                "Check disk usage of /home/agent and write a summary to "
                "/home/agent/disk_usage.txt containing the total size in "
                "human-readable format."
            ),
            "setup_commands": [
                "mkdir -p /home/agent/subdir",
                "dd if=/dev/zero of=/home/agent/test.bin bs=1024 count=100 2>/dev/null",
            ],
            "check_command": (
                "test -f /home/agent/disk_usage.txt && "
                "test -s /home/agent/disk_usage.txt"
            ),
            "tags": ["du", "system", "advanced"],
        },
    ]

    tasks: list[dict[str, Any]] = []
    for i in range(num_tasks):
        tmpl = templates[i % len(templates)]
        task_id = f"shell_{i:04d}"

        tasks.append({
            "prompt": tmpl["prompt"],
            "label": task_id,
            "metadata": {
                "benchmark": "simple_shell",
                "task_id": task_id,
                "setup_commands": tmpl["setup_commands"],
                "check_command": tmpl["check_command"],
                "expected_exit_code": 0,
                "timeout_sec": 120,
                "workdir": "/home/agent",
                "max_turns": 10,
                "tags": tmpl.get("tags", []),
            },
        })
    return tasks


def generate_simple_math_tasks(num_tasks: int = _DEFAULT_NUM_TASKS) -> list[dict[str, Any]]:
    """Generate simple math problems (GSM8K-style).

    Agent uses Python tool to solve. Reward: answer matches ground truth.
    """
    problems: list[dict[str, Any]] = [
        {
            "prompt": (
                "Solve this math problem using Python code. "
                "Output the final numerical answer on the last line.\n\n"
                "Problem: {problem}"
            ),
            "problem": (
                "A store sells apples for $2 each and oranges for $3 each. "
                "If John buys 5 apples and 4 oranges, how much does he spend in total?"
            ),
            "answer": "22",
            "verify_code": "assert 5 * 2 + 4 * 3 == 22",
        },
        {
            "prompt": (
                "Solve this math problem using Python code. "
                "Output the final numerical answer on the last line.\n\n"
                "Problem: {problem}"
            ),
            "problem": (
                "A train travels at 60 km/h for 2.5 hours, then at 80 km/h "
                "for 1.5 hours. What is the total distance traveled?"
            ),
            "answer": "270",
            "verify_code": "assert abs(60 * 2.5 + 80 * 1.5 - 270) < 0.01",
        },
        {
            "prompt": (
                "Solve this math problem using Python code. "
                "Output the final numerical answer on the last line.\n\n"
                "Problem: {problem}"
            ),
            "problem": (
                "What is the area of a circle with radius 7 cm? "
                "Use π = 3.14159. Round to 2 decimal places."
            ),
            "answer": "153.94",
            "verify_code": "import math; assert abs(math.pi * 7**2 - 153.94) < 0.1",
        },
        {
            "prompt": (
                "Solve this math problem using Python code. "
                "Output the final numerical answer on the last line.\n\n"
                "Problem: {problem}"
            ),
            "problem": (
                "If 3x + 7 = 22, what is the value of x?"
            ),
            "answer": "5",
            "verify_code": "assert abs(5 - 5) < 0.01",
        },
        {
            "prompt": (
                "Solve this math problem using Python code. "
                "Output the final numerical answer on the last line.\n\n"
                "Problem: {problem}"
            ),
            "problem": (
                "A rectangle has a length of 12 cm and a width of 8 cm. "
                "What is its perimeter?"
            ),
            "answer": "40",
            "verify_code": "assert 2 * (12 + 8) == 40",
        },
        {
            "prompt": (
                "Solve this math problem using Python code. "
                "Output the final numerical answer on the last line.\n\n"
                "Problem: {problem}"
            ),
            "problem": (
                "What is 15% of 240?"
            ),
            "answer": "36",
            "verify_code": "assert abs(240 * 0.15 - 36) < 0.01",
        },
        {
            "prompt": (
                "Solve this math problem using Python code. "
                "Output the final numerical answer on the last line.\n\n"
                "Problem: {problem}"
            ),
            "problem": (
                "The sum of three consecutive integers is 72. "
                "What is the largest of these integers?"
            ),
            "answer": "25",
            "verify_code": "assert 23 + 24 + 25 == 72 and max(23, 24, 25) == 25",
        },
        {
            "prompt": (
                "Solve this math problem using Python code. "
                "Output the final numerical answer on the last line.\n\n"
                "Problem: {problem}"
            ),
            "problem": (
                "A car depreciates by 10% of its value each year. "
                "If it costs $20000 new, what is its value after 2 years?"
            ),
            "answer": "16200",
            "verify_code": "assert abs(20000 * 0.9 * 0.9 - 16200) < 1",
        },
        # Generate variations
    ]

    tasks: list[dict[str, Any]] = []
    for i in range(num_tasks):
        tmpl = problems[i % len(problems)]
        task_id = f"math_{i:04d}"
        # Add slight variation
        if i >= len(problems):
            variation = f" (variation {i // len(problems)})"
            problem_text = tmpl["problem"] + variation
        else:
            problem_text = tmpl["problem"]

        tasks.append({
            "prompt": tmpl["prompt"].format(problem=problem_text),
            "label": task_id,
            "metadata": {
                "benchmark": "simple_math",
                "task_id": task_id,
                "problem": problem_text,
                "expected_answer": str(tmpl["answer"]),
                "verify_code": tmpl["verify_code"],
                "timeout_sec": 120,
                "workdir": "/home/agent",
                "max_turns": 5,
                "setup_commands": [],
            },
        })
    return tasks


def generate_simple_code_tasks(num_tasks: int = _DEFAULT_NUM_TASKS) -> list[dict[str, Any]]:
    """Generate simple coding tasks with test-based verification.

    Agent writes Python code to solve a problem, runs tests to verify.
    """
    problems: list[dict[str, Any]] = [
        {
            "prompt": (
                "Write a Python function that {description}. "
                "Save it to /home/agent/solution.py and run the tests "
                "at /home/agent/test_solution.py to verify."
            ),
            "description": "reverses a string",
            "test_code": (
                "import sys; sys.path.insert(0, '/home/agent')\n"
                "from solution import reverse_string\n"
                "assert reverse_string('hello') == 'olleh'\n"
                "assert reverse_string('') == ''\n"
                "assert reverse_string('a') == 'a'\n"
                "print('All tests passed!')"
            ),
            "starter_code": (
                "def reverse_string(s):\n"
                "    # TODO: implement\n"
                "    pass\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function that {description}. "
                "Save it to /home/agent/solution.py and run the tests "
                "at /home/agent/test_solution.py to verify."
            ),
            "description": "checks if a string is a palindrome",
            "test_code": (
                "import sys; sys.path.insert(0, '/home/agent')\n"
                "from solution import is_palindrome\n"
                "assert is_palindrome('racecar') == True\n"
                "assert is_palindrome('hello') == False\n"
                "assert is_palindrome('A man a plan a canal Panama'.lower().replace(' ', '')) == True\n"
                "print('All tests passed!')"
            ),
            "starter_code": (
                "def is_palindrome(s):\n"
                "    # TODO: implement\n"
                "    pass\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function that {description}. "
                "Save it to /home/agent/solution.py and run the tests "
                "at /home/agent/test_solution.py to verify."
            ),
            "description": "returns the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1)",
            "test_code": (
                "import sys; sys.path.insert(0, '/home/agent')\n"
                "from solution import fibonacci\n"
                "assert fibonacci(0) == 0\n"
                "assert fibonacci(1) == 1\n"
                "assert fibonacci(10) == 55\n"
                "assert fibonacci(20) == 6765\n"
                "print('All tests passed!')"
            ),
            "starter_code": (
                "def fibonacci(n):\n"
                "    # TODO: implement\n"
                "    pass\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function that {description}. "
                "Save it to /home/agent/solution.py and run the tests "
                "at /home/agent/test_solution.py to verify."
            ),
            "description": "counts the number of vowels in a string",
            "test_code": (
                "import sys; sys.path.insert(0, '/home/agent')\n"
                "from solution import count_vowels\n"
                "assert count_vowels('hello') == 2\n"
                "assert count_vowels('AEIOU') == 5\n"
                "assert count_vowels('rhythm') == 0\n"
                "assert count_vowels('') == 0\n"
                "print('All tests passed!')"
            ),
            "starter_code": (
                "def count_vowels(s):\n"
                "    # TODO: implement\n"
                "    pass\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function that {description}. "
                "Save it to /home/agent/solution.py and run the tests "
                "at /home/agent/test_solution.py to verify."
            ),
            "description": "finds the maximum value in a list (without using built-in max())",
            "test_code": (
                "import sys; sys.path.insert(0, '/home/agent')\n"
                "from solution import find_max\n"
                "assert find_max([1, 5, 3, 9, 2]) == 9\n"
                "assert find_max([-1, -5, -3]) == -1\n"
                "assert find_max([42]) == 42\n"
                "assert find_max([]) is None\n"
                "print('All tests passed!')"
            ),
            "starter_code": (
                "def find_max(lst):\n"
                "    # TODO: implement\n"
                "    pass\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function that {description}. "
                "Save it to /home/agent/solution.py and run the tests "
                "at /home/agent/test_solution.py to verify."
            ),
            "description": "checks if a number is prime",
            "test_code": (
                "import sys; sys.path.insert(0, '/home/agent')\n"
                "from solution import is_prime\n"
                "assert is_prime(2) == True\n"
                "assert is_prime(17) == True\n"
                "assert is_prime(4) == False\n"
                "assert is_prime(1) == False\n"
                "assert is_prime(97) == True\n"
                "print('All tests passed!')"
            ),
            "starter_code": (
                "def is_prime(n):\n"
                "    # TODO: implement\n"
                "    pass\n"
            ),
        },
    ]

    tasks: list[dict[str, Any]] = []
    for i in range(num_tasks):
        tmpl = problems[i % len(problems)]
        task_id = f"code_{i:04d}"

        tasks.append({
            "prompt": tmpl["prompt"].format(description=tmpl["description"]),
            "label": task_id,
            "metadata": {
                "benchmark": "simple_code",
                "task_id": task_id,
                "description": tmpl["description"],
                "test_code": tmpl["test_code"],
                "starter_code": tmpl["starter_code"],
                "timeout_sec": 180,
                "workdir": "/home/agent",
                "max_turns": 15,
                "setup_commands": [
                    f"cat > /home/agent/solution.py << 'PYEOF'\n{tmpl['starter_code']}\nPYEOF",
                    f"cat > /home/agent/test_solution.py << 'PYEOF'\n{tmpl['test_code']}\nPYEOF",
                ],
                "check_command": "cd /home/agent && python3 test_solution.py",
            },
        })
    return tasks


def _ensure_hf_datasets():
    """Lazy-import ``datasets`` with a helpful error message."""
    try:
        from datasets import load_dataset  # noqa: F401

        return load_dataset
    except ImportError:
        logger.warning(
            "The 'datasets' library is not installed. "
            "Install it with: pip install datasets"
        )
        raise


def _generate_synthetic_alfworld_tasks(num_tasks: int = 64) -> list[dict[str, Any]]:
    """Generate synthetic ALFWorld-style tasks for offline training.

    When the actual ALFWorld environment is not installed, we generate
    templated tasks that exercise the same reasoning patterns using
    file-system-based simulated environments.
    """
    templates: list[dict[str, Any]] = [
        {
            "task_type": "pick_and_place_simple",
            "prompt": (
                "You are in a household environment. Your task is to {goal}.\n\n"
                "The house has these rooms: kitchen, living room, bedroom, bathroom.\n"
                "Objects are in various locations. Use shell commands to navigate\n"
                "and interact with the environment.\n\n"
                "Available commands:\n"
                "  cd <room> — go to a room\n"
                "  ls — list objects in current room\n"
                "  cat <object> — examine an object\n"
                "  mv <object> <destination> — move an object\n"
                "  echo 'done' > /tmp/status — signal task completion\n\n"
                "Start in the kitchen. Complete the task efficiently."
            ),
            "goals": [
                "take the mug from the kitchen and put it on the desk in the living room",
                "take the apple from the living room and put it in the fridge in the kitchen",
                "find the book in the bedroom and put it on the shelf in the living room",
                "take the soap from the bathroom and put it on the kitchen counter",
            ],
            "setup_commands": [
                "mkdir -p /home/agent/kitchen /home/agent/living_room "
                "/home/agent/bedroom /home/agent/bathroom",
                "echo 'a ceramic mug' > /home/agent/kitchen/mug.txt",
                "echo 'a red apple' > /home/agent/living_room/apple.txt",
                "echo 'a thick book' > /home/agent/bedroom/book.txt",
                "echo 'a bar of soap' > /home/agent/bathroom/soap.txt",
                "echo 'a wooden desk' > /home/agent/living_room/desk.txt",
                "echo 'a white fridge' > /home/agent/kitchen/fridge.txt",
                "echo 'a bookshelf' > /home/agent/living_room/shelf.txt",
                "echo 'a kitchen counter' > /home/agent/kitchen/counter.txt",
            ],
            "check_commands": [
                "find /home/agent -name '*mug*' -path '*/living_room/*' | head -1 | grep -q .",
                "find /home/agent -name '*apple*' -path '*/kitchen/*' | head -1 | grep -q .",
                "find /home/agent -name '*book*' -path '*/living_room/*' | head -1 | grep -q .",
                "find /home/agent -name '*soap*' -path '*/kitchen/*' | head -1 | grep -q .",
            ],
        },
        {
            "task_type": "navigation",
            "prompt": (
                "You are in a large building. Your task: {goal}.\n\n"
                "The building has: lobby, office_A, office_B, server_room, break_room.\n"
                "Use shell commands to navigate (cd) and explore (ls, cat, find).\n"
                "Signal completion with: echo 'done' > /tmp/status"
            ),
            "goals": [
                "find the server_room and check if server_status.txt says 'online'",
                "go to office_B and check if the report.pdf exists",
                "find which room has the coffee_machine and report its status",
                "go to every room and count how many people.txt files exist",
            ],
            "setup_commands": [
                "mkdir -p /home/agent/lobby /home/agent/office_A "
                "/home/agent/office_B /home/agent/server_room "
                "/home/agent/break_room",
                "echo 'online' > /home/agent/server_room/server_status.txt",
                "echo 'Q3 report' > /home/agent/office_B/report.pdf",
                "echo 'brewing' > /home/agent/break_room/coffee_machine.txt",
                "echo 'Alice' > /home/agent/office_A/people.txt",
                "echo 'Bob' > /home/agent/office_B/people.txt",
            ],
            "check_commands": [
                "grep -q 'online' /home/agent/server_room/server_status.txt",
                "test -f /home/agent/office_B/report.pdf",
                "grep -q 'brewing' /home/agent/break_room/coffee_machine.txt",
                "find /home/agent -name 'people.txt' | wc -l | grep -q '2'",
            ],
        },
    ]

    tasks: list[dict[str, Any]] = []
    for i in range(num_tasks):
        tmpl = templates[i % len(templates)]
        goal_idx = i % len(tmpl["goals"])
        goal = tmpl["goals"][goal_idx]
        task_id = f"alfworld_{tmpl['task_type']}_{i:04d}"

        tasks.append({
            "prompt": tmpl["prompt"].format(goal=goal),
            "label": task_id,
            "metadata": {
                "benchmark": "alfworld",
                "task_type": tmpl["task_type"],
                "task_id": task_id,
                "goal": goal,
                "setup_commands": tmpl["setup_commands"],
                "check_command": tmpl["check_commands"][
                    goal_idx % len(tmpl["check_commands"])
                ],
                "max_turns": 15,
                "timeout_sec": 120,
                "workdir": "/home/agent",
            },
        })
    return tasks


# =============================================================================
# Merge
# =============================================================================


def merge_all(output_dir: Path, max_per_benchmark: int | None = None) -> Path:
    """Merge all per-benchmark JSONL files into one mixed dataset."""
    import glob

    all_lines: list[str] = []
    for jsonl_path in sorted(glob.glob(str(output_dir / "*.jsonl"))):
        if jsonl_path.endswith("mixed_simple_rl.jsonl"):
            continue
        with open(jsonl_path) as f:
            for line in f:
                all_lines.append(line)

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

    output_path = output_dir / "mixed_simple_rl.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for line in all_lines:
            f.write(line)

    # Per-benchmark counts
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
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate simple datasets for agentic RL GRPO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all simple datasets
  python examples/agentic_rl_grpo/download_simple_data.py -o ./data/simple

  # Generate specific benchmarks
  python examples/agentic_rl_grpo/download_simple_data.py -o ./data/simple \\
      --benchmarks simple_shell,simple_math

  # Quick smoke test (small number of tasks)
  python examples/agentic_rl_grpo/download_simple_data.py -o ./data/simple \\
      --max-samples 32

Use with run_simple.sh:
  bash examples/agentic_rl_grpo/run_simple.sh \\
      --prompt-data ./data/simple/mixed_simple_rl.jsonl \\
      ...
        """,
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./data/simple",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--benchmarks",
        default="all",
        help="Comma-separated: simple_shell,simple_math,simple_code,alfworld,terminal_bench "
             "or 'all'",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=_DEFAULT_NUM_TASKS,
        help=f"Max samples per benchmark (default: {_DEFAULT_NUM_TASKS})",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merging into mixed_simple_rl.jsonl",
    )
    parser.add_argument(
        "--hf-mirror",
        default=None,
        help="HF mirror for terminal-bench download (e.g., hf-mirror.com)",
    )
    args = parser.parse_args()

    all_benchmarks = [
        "simple_shell", "simple_math", "simple_code",
        "alfworld", "terminal_bench",
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

    paths: list[Path] = []

    # ---- Generate simple shell tasks ----
    if "simple_shell" in benchmarks:
        logger.info("Generating simple shell tasks...")
        tasks = generate_simple_shell_tasks(args.max_samples)
        output_path = output_dir / "simple_shell.jsonl"
        _write_jsonl(output_path, tasks)
        logger.info("  Wrote %d simple shell tasks → %s", len(tasks), output_path)
        paths.append(output_path)

    # ---- Generate simple math tasks ----
    if "simple_math" in benchmarks:
        logger.info("Generating simple math tasks...")
        tasks = generate_simple_math_tasks(args.max_samples)
        output_path = output_dir / "simple_math.jsonl"
        _write_jsonl(output_path, tasks)
        logger.info("  Wrote %d simple math tasks → %s", len(tasks), output_path)
        paths.append(output_path)

    # ---- Generate simple code tasks ----
    if "simple_code" in benchmarks:
        logger.info("Generating simple code tasks...")
        tasks = generate_simple_code_tasks(args.max_samples)
        output_path = output_dir / "simple_code.jsonl"
        _write_jsonl(output_path, tasks)
        logger.info("  Wrote %d simple code tasks → %s", len(tasks), output_path)
        paths.append(output_path)

    # ---- Generate ALFWorld tasks ----
    if "alfworld" in benchmarks:
        logger.info("Generating ALFWorld tasks...")
        tasks = _generate_synthetic_alfworld_tasks(args.max_samples)
        output_path = output_dir / "alfworld.jsonl"
        _write_jsonl(output_path, tasks)
        logger.info("  Wrote %d ALFWorld tasks → %s", len(tasks), output_path)
        paths.append(output_path)

    # ---- Terminal-Bench (from HF, with synthetic fallback) ----
    if "terminal_bench" in benchmarks:
        logger.info("Downloading Terminal-Bench from HF...")
        try:
            # Try HF download with transparent fallback
            _ensure_hf_datasets()
            from datasets import load_dataset

            # Set HF mirror if configured
            if args.hf_mirror:
                import os as _os
                _os.environ["HF_ENDPOINT"] = (
                    f"https://{args.hf_mirror}"
                    if not args.hf_mirror.startswith("http")
                    else args.hf_mirror
                )

            ds = load_dataset("ia03/terminal-bench", split="test")
            tasks: list[dict[str, Any]] = []
            for row in ds:
                task_id = row.get("task_id", "unknown")
                base_desc = row.get("base_description", "")
                task_yaml_str = row.get("task_yaml", "") or ""

                # Parse task_yaml for evaluation info
                check_cmd = ""
                expected_exit_code = 0
                setup_cmds: list[str] = []
                if task_yaml_str:
                    try:
                        import yaml
                        data = yaml.safe_load(task_yaml_str)
                        if isinstance(data, dict):
                            check_cmd = data.get("check_command", "")
                            expected_exit_code = data.get("expected_exit_code", 0)
                            setup_cmds = data.get("setup_commands", [])
                    except Exception:
                        pass

                tasks.append({
                    "prompt": base_desc,
                    "label": task_id,
                    "metadata": {
                        "benchmark": "terminal_bench",
                        "task_id": task_id,
                        "setup_commands": setup_cmds,
                        "check_command": check_cmd,
                        "expected_output": "",
                        "expected_exit_code": int(expected_exit_code),
                        "timeout_sec": 120,
                        "tags": row.get("tags", []) or [],
                        "category": row.get("category", ""),
                    },
                })
            if args.max_samples and len(tasks) > args.max_samples:
                tasks = tasks[: args.max_samples]
            output_path = output_dir / "terminal_bench.jsonl"
            _write_jsonl(output_path, tasks)
            logger.info("  Downloaded %d Terminal-Bench tasks → %s", len(tasks), output_path)
            paths.append(output_path)
        except Exception as e:
            logger.warning("Terminal-Bench HF download failed: %s", e)
            logger.info("Generating synthetic terminal tasks as fallback...")
            tasks = generate_simple_shell_tasks(args.max_samples)
            for t in tasks:
                t["metadata"]["benchmark"] = "terminal_bench"
            output_path = output_dir / "terminal_bench.jsonl"
            _write_jsonl(output_path, tasks)
            logger.info("  Wrote %d synthetic terminal tasks → %s", len(tasks), output_path)
            paths.append(output_path)

    # ---- Merge ----
    if not args.no_merge and len(paths) >= 1:
        merge_all(output_dir, args.max_samples)

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("Done! Output directory: %s", output_dir)
    logger.info("")
    logger.info("Use with run_simple.sh:")
    logger.info(
        "  --prompt-data %s/mixed_simple_rl.jsonl \\",
        output_dir.resolve(),
    )
    logger.info("  --input-key prompt --label-key label --metadata-key metadata \\")
    logger.info("  --apply-chat-template")
    logger.info("=" * 60)


def _write_jsonl(path: Path, tasks: list[dict[str, Any]]) -> None:
    """Write tasks to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
