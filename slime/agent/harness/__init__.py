"""Swappable coding-agent harnesses (Claude Code, Codex, Hermes, ...)."""

from __future__ import annotations

from .claude_code import ClaudeCodeHarness
from .codex import CodexHarness
from .common import BaseHarness, HarnessContext
from .hermes import HermesHarness

__all__ = [
    "BaseHarness",
    "HarnessContext",
    "ClaudeCodeHarness",
    "CodexHarness",
    "HermesHarness",
]
