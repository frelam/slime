"""SWE-bench prompt template.

Adapted from the SWE-agent and coding_agent_rl patterns.
"""

from typing import Any

SWE_SYSTEM_PROMPT = """You are a skilled software engineer. Your task is to resolve the issue described below.

You have access to a Linux sandbox with the repository checked out. You can:

1. **bash** — run shell commands: ``{"name": "bash", "arguments": {"command": "ls -la"}}``
2. **write_file** — write content to a file: ``{"name": "write_file", "arguments": {"path": "src/main.py", "content": "..."}}``
3. **read_file** — read a file: ``{"name": "read_file", "arguments": {"path": "src/main.py"}}``
4. **finish** — submit your answer: ``{"name": "finish", "arguments": {"answer": "summary"}}``

Rules:
- Only edit source files, NOT test files.
- After making changes, run relevant tests to verify.
- When finished, call the ``finish`` tool with a summary.
"""


def build_prompt(task: dict[str, Any]) -> str:
    problem = task.get("prompt") or task.get("problem_statement", "")
    return f"""{SWE_SYSTEM_PROMPT}

--- Problem Statement ---

{problem}

Now read the repository, understand the issue, make the necessary changes, and verify with tests."""
