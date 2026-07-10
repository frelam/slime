"""AgentBench prompt template.

Covers OS (bash commands) and DB (SQL queries) task types.
The model is instructed to use tool calls in a JSON format compatible with
the agent loop's parser.
"""

from typing import Any

AGENTBENCH_SYSTEM_PROMPT = """You are an autonomous AI agent. Your task is to complete the objective described below.

You have access to a Linux environment. You can use these tools:

1. **bash** — run shell commands: ``{"name": "bash", "arguments": {"command": "ls -la"}}``
2. **python** — execute Python code: ``{"name": "python", "arguments": {"code": "print('hello')"}}``
3. **write_file** — write content to a file: ``{"name": "write_file", "arguments": {"path": "output.txt", "content": "..."}}``
4. **read_file** — read a file: ``{"name": "read_file", "arguments": {"path": "output.txt"}}``
5. **finish** — submit your final answer: ``{"name": "finish", "arguments": {"answer": "result"}}``

Rules:
- For OS tasks: use bash to run commands, inspect outputs, and complete the objective.
- For DB tasks: use bash to connect to MySQL and write SQL queries.
- When done, call ``finish`` with your result.
"""


def build_prompt(task: dict[str, Any]) -> str:
    task_type = (
        task.get("metadata", {}).get("task_type", "os")
        if isinstance(task.get("metadata"), dict)
        else "os"
    )
    instruction = task.get("prompt") or task.get("instruction", "")

    type_hint = ""
    if task_type == "db":
        type_hint = (
            "\n\nThis is a DATABASE task. Use MySQL (mysql -u root -p) to "
            "access the database. Write SQL queries to solve the problem."
        )
    elif task_type == "os":
        type_hint = (
            "\n\nThis is an OS task. Use bash commands to navigate the "
            "filesystem, manipulate files, and complete the objective."
        )

    return f"""{AGENTBENCH_SYSTEM_PROMPT}

--- Task ---

{instruction}{type_hint}

Complete the task and call ``finish`` when done."""
