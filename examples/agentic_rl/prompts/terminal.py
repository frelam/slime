"""Terminal-Bench prompt template.

Terminal tasks require the model to execute shell commands in a Linux
environment to achieve a goal.
"""

from typing import Any

TERMINAL_PROMPT = """You are a Linux terminal expert. Your task is described below.

You have a bash shell at your disposal. You can:

1. **bash** — run any shell command: ``{{"name": "bash", "arguments": {{"command": "ls -la"}}}}``
2. **finish** — submit your answer: ``{{"name": "finish", "arguments": {{"answer": "summary of what you did"}}}}``

Work through the task step by step. When you are confident the task is complete, call ``finish``.
"""


def build_prompt(task: dict[str, Any]) -> str:
    instruction = task.get("prompt") or task.get("instruction", "")
    return f"""{TERMINAL_PROMPT}

--- Task ---

{instruction}
"""
