"""τ-bench prompt template.

τ-bench tasks involve interacting with a service environment (retail/airline)
via tool calls to accomplish goals.
"""

from typing import Any


BUILD_TOOL_DESCRIPTION = """You are a customer service agent. Your goal is to complete the task described below.

You have access to the following tools:
{tool_descriptions}

To use a tool, respond with:
```json
{{"name": "tool_name", "arguments": {{"param1": "value1", ...}}}}
```

After completing the task, use the ``finish`` tool to submit.
"""


def build_prompt(task: dict[str, Any]) -> str:
    instruction = task.get("prompt") or task.get("instruction", "")
    return f"{BUILD_TOOL_DESCRIPTION}\n\nTask: {instruction}"
