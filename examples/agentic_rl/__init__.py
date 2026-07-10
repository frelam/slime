"""Agentic RL training pipeline for slime.

This package provides the full online DPO pipeline for agentic RL:

- :mod:`generate` — custom generate function for slime's rollout loop
- :mod:`agent_loop` — multi-turn SGLang interaction loop with tool execution
- :mod:`sandbox` — sandbox lifecycle management
- :mod:`reward` — task-reward function dispatching to dataset adapters
- :mod:`dpo_loss` — online DPO loss function for ``--custom-loss-function-path``
- :mod:`prompts` — per-benchmark prompt templates
"""
