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


def _format_score(
    text: str,
    tools: list[dict] | None = TOOLS,
    expects_no_tools: bool = False,
) -> float:
    from examples.tool_rl.reward.verifier import check_format_compliance

    return check_format_compliance(
        _traj(text),
        available_tools=tools,
        expects_no_tools=expects_no_tools,
    )


def _tool_call_format_score(
    text: str,
    label_calls: list[dict] | None = None,
) -> float:
    from examples.tool_rl.reward.verifier import check_tool_call_format

    return check_tool_call_format(_traj(text), label_calls=label_calls)


def _xml_call(name: str = "get_weather", value: str = '"Beijing"') -> str:
    return (
        "<tool_call>\n"
        f"<function={name}>\n"
        f'<parameter=city>{value}</parameter>\n'
        "</function>\n"
        "</tool_call>"
    )


def _inline_call(name: str = "get_weather", args: str = '{"city": "Beijing"}') -> str:
    """Inline JSON style: <tool_call>"name":..., "arguments": {...}</tool_call>"""
    return (
        "<tool_call>\n"
        f'"name": "{name}", "arguments": {args}\n'
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

    def test_repeated_think_zero(self):
        """Repeated think blocks break the strict single-block format → 0.0."""
        text = "<think>a</think>\n<think>b</think>\n" + _xml_call()
        assert _format_score(text) == pytest.approx(0.0)

    def test_unclosed_think_zero(self):
        """An unclosed <think> (think-then-stop collapse) → 0.0."""
        text = "<think>let me check the weather\n" + _xml_call()
        assert _format_score(text) == pytest.approx(0.0)

    def test_think_only_nothing_after_zero(self):
        """A closed think block with no response/tool_call after it → 0.0."""
        text = "<think>deep reasoning</think>"
        assert _format_score(text) == pytest.approx(0.0)
        assert _format_score(text, tools=None) == pytest.approx(0.0)
        assert _format_score(text, expects_no_tools=True) == pytest.approx(0.0)

    def test_no_think_zero(self):
        text = _xml_call()
        assert _format_score(text) == pytest.approx(0.0)

    def test_think_after_call_partial_credit(self):
        """Call before the think: 0.6 lost, half of the calls preceded."""
        text = _xml_call(value='"A"') + "\n<think>x</think>\n" + _xml_call(value='"B"')
        assert _format_score(text) == pytest.approx(0.2)

    def test_interleaved_think_call_pairs_zero(self):
        """Interleaved repeated think blocks break the strict format → 0.0."""
        text = (
            "<think>a</think>\n" + _xml_call(value='"A"') + "\n"
            "<think>b</think>\n" + _xml_call(value='"B"')
        )
        assert _format_score(text) == pytest.approx(0.0)


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


# Real think tags as emitted by the Qwen3 chat template.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class TestNoTools:
    def test_no_thinking_no_calls_no_tools_full(self):
        """A response with no reasoning block is allowed — reasoning is
        optional. A correct no-call answer gets full format credit."""
        text = "nothing to do"
        assert _format_score(text, tools=None) == pytest.approx(1.0)

    def test_no_calls_with_tools_zero(self):
        text = "nothing to do"
        assert _format_score(text, tools=TOOLS) == pytest.approx(0.0)

    def test_no_calls_with_tools_full_when_label_expects_none(self):
        """gt=[] means no tools needed; a no-reasoning, no-call answer is fine."""
        text = "nothing to do"
        assert _format_score(text, tools=TOOLS, expects_no_tools=True) == pytest.approx(1.0)

    def test_empty_think_block_with_answer_allowed(self):
        """An empty ``<think></think>`` followed by an answer is acceptable —
        empty reasoning alone is not a format violation."""
        text = _THINK_OPEN + _THINK_CLOSE + "\nnothing to do"
        assert _format_score(text, tools=None) == pytest.approx(1.0)

    def test_empty_think_block_alone_zero(self):
        """A think block with nothing after it is the think-then-stop
        collapse — a format violation even when no tools are defined."""
        text = _THINK_OPEN + _THINK_CLOSE
        assert _format_score(text, tools=None) == pytest.approx(0.0)

    def test_empty_block_with_tools_full_when_label_expects_none(self):
        """Empty blocks are not calls; with a no-tool label they don't fail."""
        text = "<think>nothing to do</think>\n<tool_call></tool_call>"
        assert _format_score(text, tools=TOOLS, expects_no_tools=True) == pytest.approx(1.0)


LABEL_WEATHER = [{"name": "get_weather", "arguments": {"city": "Beijing"}}]
LABEL_TWO = [
    {"name": "get_weather", "arguments": {"city": "Beijing"}},
    {"name": "get_time", "arguments": {"tz": "UTC"}},
]


class TestToolCallFormatNoLabel:
    """Dim 3 = full score when the label has no tool calls (no detection)."""

    def test_no_label_full_score(self):
        assert _tool_call_format_score("no tools needed") == pytest.approx(1.0)

    def test_no_label_full_score_even_with_calls(self):
        text = " thinkingx response\n" + _xml_call()
        assert _tool_call_format_score(text) == pytest.approx(1.0)

    def test_no_label_full_score_even_with_garbage(self):
        """Without a label there is no detection — think-then-stop collapse,
        wrong tool names, etc. are all ignored (score stays 1.0)."""
        assert _tool_call_format_score(" thinkingdeep reasoning response") == pytest.approx(1.0)
        assert _tool_call_format_score(_THINK_OPEN) == pytest.approx(1.0)
        assert _tool_call_format_score(_xml_call(name="not_a_tool")) == pytest.approx(1.0)


class TestToolCallFormatLabelMatch:
    """Dim 3 matches output calls against ground-truth label calls.

    A call is a complete match when the tool name agrees, the output provides
    exactly the label's parameter names, and every parameter value has a
    compatible type. Values themselves are ignored. Score = matched / N.
    """

    def test_exact_name_match_full(self):
        text = " thinkingx response\n" + _xml_call()  # get_weather(city="Beijing")
        assert _tool_call_format_score(text, label_calls=LABEL_WEATHER) == pytest.approx(1.0)

    def test_wrong_tool_name_zero(self):
        text = " thinkingx response\n" + _xml_call(name="get_other")
        assert _tool_call_format_score(text, label_calls=LABEL_WEATHER) == pytest.approx(0.0)

    def test_wrong_param_name_zero(self):
        text = (
            " thinkingx response\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            '<parameter=city2>"Beijing"</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        assert _tool_call_format_score(text, label_calls=LABEL_WEATHER) == pytest.approx(0.0)

    def test_param_type_mismatch_zero(self):
        """Label expects city as a string; output passes an int → mismatch."""
        text = " thinkingx response\n" + _xml_call(value="123")
        assert _tool_call_format_score(text, label_calls=LABEL_WEATHER) == pytest.approx(0.0)

    def test_value_content_ignored(self):
        """Different values are fine — only param names and types are checked."""
        text = " thinkingx response\n" + _xml_call(value='"Tokyo"')
        assert _tool_call_format_score(text, label_calls=LABEL_WEATHER) == pytest.approx(1.0)

    def test_partial_match_ratio(self):
        """Only get_weather matched → 1/2 of the two label calls."""
        text = " thinkingx response\n" + _xml_call()
        assert _tool_call_format_score(text, label_calls=LABEL_TWO) == pytest.approx(0.5)

    def test_extra_output_call_does_not_add(self):
        """Extra unmatched output calls don't increase the score."""
        text = " thinkingx response\n" + _xml_call() + "\n" + _xml_call(name="other", value='"A"')
        assert _tool_call_format_score(text, label_calls=LABEL_WEATHER) == pytest.approx(1.0)

    def test_missing_param_not_complete_match(self):
        """Label expects two params; output provides only one → not complete."""
        label_two_params = [{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "C"}}]
        text = " thinkingx response\n" + _xml_call()  # only city
        assert _tool_call_format_score(text, label_calls=label_two_params) == pytest.approx(0.0)


class TestValuesMatch:
    """JSON-encoded array/object strings are semantically equal to values."""

    def test_json_encoded_array_string_matches_list(self):
        from examples.tool_rl.reward.verifier import _values_match

        assert _values_match('["session1", "session2"]', ["session1", "session2"])

    def test_json_encoded_array_strings_match_each_other(self):
        from examples.tool_rl.reward.verifier import _values_match

        assert _values_match('["session1", "session2"]', '["session1", "session2"]')

    def test_json_encoded_object_string_matches_dict(self):
        from examples.tool_rl.reward.verifier import _values_match

        assert _values_match('{"a": 1}', {"a": 1})


# ============================================================================
# JSON-fallback tool calls still count
# ============================================================================


class TestJsonFallback:
    def test_json_call_after_think_full(self):
        text = '<think>x</think>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        assert _format_score(text) == pytest.approx(1.0)


# ============================================================================
# Inline JSON tool_call format — <tool_call>"name":..., "arguments": {...}</tool_call>
# ============================================================================

class TestInlineJsonFormat:
    """The inline JSON style is recognized by both Dim 2 and Dim 3."""

    def test_inline_call_after_think_full(self):
        text = "<think>x</think>\n" + _inline_call()
        assert _format_score(text) == pytest.approx(1.0)
        assert _tool_call_format_score(
            text, label_calls=LABEL_WEATHER,
        ) == pytest.approx(1.0)

    def test_inline_call_parsed_args(self):
        from examples.tool_rl.reward.verifier import parse_qwen_tool_calls

        calls = parse_qwen_tool_calls(_inline_call())
        assert calls == [{"name": "get_weather", "arguments": {"city": "Beijing"}}]

    def test_inline_unknown_tool_does_not_match_label_dim3(self):
        text = " thinkingx response\n" + _inline_call(name="not_a_tool")
        assert _tool_call_format_score(
            text, label_calls=LABEL_WEATHER,
        ) == pytest.approx(0.0)

    def test_inline_call_counted_in_spans(self):
        from examples.tool_rl.reward.verifier import _xml_tool_call_spans

        assert len(_xml_tool_call_spans(_inline_call())) == 1


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
