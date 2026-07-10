"""CLI-Gym prompt template.

CLI-Gym tasks involve interacting with interactive command-line tools to
achieve specific outcomes.
"""

from typing import Any

CLI_PROMPT = """You are a CLI tool expert. Your task is to use the available command-line tools
to accomplish the goal described below.

You can:

1. **bash** — run shell commands: ``{{"name": "bash", "arguments": {{"command": "my_command --flag value"}}}}``
2. **finish** — submit when complete: ``{{"name": "finish", "arguments": {{"answer": "summary"}}}}``

Explore the environment, use the tools, and accomplish the task.
"""


def build_prompt(task: dict[str, Any]) -> str:
    instruction = task.get("prompt") or task.get("instruction", "")
    return f"""{CLI_PROMPT}

--- Task ---

{instruction}
"""
