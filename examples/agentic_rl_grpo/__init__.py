"""GRPO/PPO training for agentic RL.

This package provides:
- :mod:`generate` — custom generate function for ``--custom-generate-function-path``
- :mod:`reward` — multi-dimensional reward composer (RM + verifier combined)
- :mod:`reward_model` — RM API client (Qwen3.5-9B, DeepSeek API)
- :mod:`verifier` — rule-based verifier dimensions (4.2, 4.3, 4.4, 4.7)
- :mod:`traj_analysis` — trajectory parsing utilities
"""
