"""Tests for Dim 2 format compliance scoring in the tool_rl verifier.

Covers ``check_format_compliance``:
  - <think> must appear exactly once with non-empty content for full credit
  - Empty <tool_call> blocks (no ``<function=...>`` header) are not counted
  - Multiple valid tool calls are allowed (no repetition penalty)
  - JSON-fallback tool calls still count toward the format score
"""

from __future__ import annotations

import pytest


TOOLS = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    },
]


def _traj(text: str) -> list[dict]:
    return [{"turn": 0, "text": text, "finish_reason": "stop", "type": "turn"}]


def _format_score(text: str, tools: list[dict] | None = TOOLS) -> float:
    from examples.tool_rl.reward.verifier import check_format_compliance

    return check_format_compliance(_traj(text), available_tools=tools)


def _xml_call(name: str = "get_weather", value: str = '"Beijing"') -> str:
    return (
        "<tool_call>\n"
        f"<function={name}>\n"
        f'<parameter=city>{value}</parameter>\n'
        "</function>\n"
        "</tool_call>"
    )


# ============================================================================
# <think> validity — exactly one, non-empty content
# ============================================================================


class TestThinkValidity:
    """Full credit requires exactly one non-empty <think> block."""

    def test_single_non_empty_think_full_credit(self):
        text = "<think>I need the weather.</think>\n" + _xml_call()
        assert _format_score(text) == pytest.approx(1.0)

    def test_empty_think_forfeits_format(self):
        """An empty <think> is not valid reasoning → no format points."""
        text = "<think></think>\n" + _xml_call()
        assert _format_score(text) == pytest.approx(0.0)

    def test_whitespace_only_think_forfeits_format(self):
        text = "<think>   </think>\n" + _xml_call()
        assert _format_score(text) == pytest.approx(0.0)

    def test_repeated_think_partial_credit(self):
        """Two non-empty thinks lose the 0.6, keep the 0.4 preceded credit."""
        text = (
            "<think>a</think><think>b</think>\n" + _xml_call()
        )
        assert _format_score(text) == pytest.approx(0.4)

    def test_no_think_zero(self):
        text = _xml_call()
        assert _format_score(text) == pytest.approx(0.0)

    def test_think_after_call_partial_credit(self):
        """Call before the think: 0.6 lost, half of the calls preceded."""
        text = _xml_call(value='"A"') + "\n<think>x</think>\n" + _xml_call(value='"B"')
        assert _format_score(text) == pytest.approx(0.2)

    def test_interleaved_think_call_pairs_partial_credit(self):
        text = (
            "<think>a</think>\n" + _xml_call(value='"A"') + "\n"
            "<think>b</think>\n" + _xml_call(value='"B"')
        )
        assert _format_score(text) == pytest.approx(0.4)


# ============================================================================
# Empty <tool_call> blocks are not counted as calls
# ============================================================================


class TestEmptyToolCallBlocks:
    """Empty blocks (no <function> header) must not count toward format."""

    def test_empty_block_alone_not_a_call(self):
        text = "<think>x</think>\n<tool_call></tool_call>"
        # Tools available + no real calls → 0.0 (previously counted as 1.0)
        assert _format_score(text) == pytest.approx(0.0)

    def test_empty_block_before_think_does_not_break_rule1(self):
        """Empty blocks are ignored entirely, so they can't fail the
        'all calls after think' check even when positioned oddly."""
        text = (
            "<tool_call></tool_call>\n"
            "<think>x</think>\n" + _xml_call()
        )
        assert _format_score(text) == pytest.approx(1.0)

    def test_empty_block_alongside_valid_call_same_score(self):
        text = "<think>x</think>\n" + _xml_call() + "\n<tool_call></tool_call>"
        assert _format_score(text) == pytest.approx(1.0)

    def test_empty_block_no_tools_falls_to_no_tools_rule(self):
        """With no tools defined, an empty block is not a call, so the
        no-calls + no-tools branch applies (1.0) — same as no output."""
        text = "<think>x</think>\n<tool_call></tool_call>"
        assert _format_score(text, tools=None) == pytest.approx(1.0)


# ============================================================================
# Multiple valid tool calls — allowed, no repetition penalty
# ============================================================================


class TestMultipleToolCalls:
    """Multiple tool calls in one response are fine (no penalty)."""

    def test_two_valid_calls_full_credit(self):
        text = "<think>x</think>\n" + _xml_call(value='"A"') + "\n" + _xml_call(value='"B"')
        assert _format_score(text) == pytest.approx(1.0)

    def test_two_valid_calls_no_tools_still_full(self):
        text = "<think>x</think>\n" + _xml_call(value='"A"') + "\n" + _xml_call(value='"B"')
        assert _format_score(text, tools=None) == pytest.approx(1.0)


# ============================================================================
# No tools defined / no calls
# ============================================================================


class TestNoTools:
    def test_no_calls_no_tools_full(self):
        text = "<think>nothing to do</think>"
        assert _format_score(text, tools=None) == pytest.approx(1.0)

    def test_no_calls_with_tools_zero(self):
        text = "<think>nothing to do</think>"
        assert _format_score(text, tools=TOOLS) == pytest.approx(0.0)


# ============================================================================
# JSON-fallback tool calls still count
# ============================================================================


class TestJsonFallback:
    def test_json_call_after_think_full(self):
        text = '<think>x</think>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        assert _format_score(text) == pytest.approx(1.0)


# ============================================================================
# _xml_tool_call_spans
# ============================================================================


class TestXmlToolCallSpans:
    def test_empty_block_excluded(self):
        from examples.tool_rl.reward.verifier import _xml_tool_call_spans

        assert _xml_tool_call_spans("<tool_call></tool_call>") == []
        assert _xml_tool_call_spans("<tool_call>\n</tool_call>") == []

    def test_block_with_function_included(self):
        from examples.tool_rl.reward.verifier import _xml_tool_call_spans

        text = "<tool_call>\n<function=get_weather>\n<parameter=city>\"Beijing\"</parameter>\n</function>\n</tool_call>"
        spans = _xml_tool_call_spans(text)
        assert len(spans) == 1
        start, end = spans[0]
        assert "<tool_call>" in text[start:end]
        assert "get_weather" in text[start:end]

    def test_mixed_valid_and_empty(self):
        from examples.tool_rl.reward.verifier import _xml_tool_call_spans

        text = (
            "<tool_call></tool_call>\n"
            "<tool_call><function=get_weather><parameter=city>\"A\"</parameter></function></tool_call>\n"
            "<tool_call></tool_call>"
        )
        assert len(_xml_tool_call_spans(text)) == 1
