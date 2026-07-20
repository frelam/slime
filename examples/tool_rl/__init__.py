"""Tool RL training for Qwen3-4B — function calling / tool-use GRPO.

Directory structure
-------------------
::

    examples/tool_rl/
    ├── __init__.py
    ├── generate.py              # Custom generate function
    ├── run.sh                   # Training launch script
    ├── data/
    │   ├── __init__.py
    │   ├── download_data.py     # Dataset download & conversion
    │   └── dataset_adapter.py   # Dataset adapter (slime DatasetAdapter)
    └── reward/
        ├── __init__.py
        ├── verifier.py          # Rule-based verifier (Dim 2: format, Dim 3: tool call)
        ├── reward.py            # 4-dim reward composer (RM + Verifier)
        └── prompts/
            └── tool_rl.md       # RM system prompt
"""

from __future__ import annotations
