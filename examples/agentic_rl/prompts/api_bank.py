"""API-Bank prompt template.

API-Bank tasks require the model to call REST APIs correctly to accomplish
goals.
"""

from typing import Any

API_PROMPT = """You are an API integration expert. Your task is to use the available APIs
to accomplish the goal described below.

The API specification is available at /home/agent/api_spec.json

You can:

1. **bash** — make HTTP requests with curl: ``{{"name": "bash", "arguments": {{"command": "curl -X POST ..."}}}}``
2. **write_file** — write data to a file
3. **read_file** — read a file
4. **finish** — submit when complete: ``{{"name": "finish", "arguments": {{"answer": "summary"}}}}``

Read the API spec, understand the endpoints, and make the correct API calls.
"""


def build_prompt(task: dict[str, Any]) -> str:
    instruction = task.get("prompt") or task.get("instruction", "")
    return f"""{API_PROMPT}

--- Task ---

{instruction}
"""
