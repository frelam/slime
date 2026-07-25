"""Tests for the structured RM prompt formatting functions in tool_rl reward.

Tests ``_clean_prompt``, ``_format_tools_md``,
``_format_agent_response_structured``, and ``_build_rm_user_message``.
"""

from __future__ import annotations

import json
import re

import pytest


# ============================================================================
# Helpers
# ============================================================================


def _make_trajectory_record(
    turn: int = 0,
    text: str = "",
    finish_reason: str = "stop",
    rtype: str = "turn",
    tool_calls_parsed: list[dict] | None = None,
    tool_call_name: str = "",
) -> dict:
    """Build a single trajectory record matching the generate.py format."""
    rec: dict = {
        "turn": turn,
        "text": text,
        "finish_reason": finish_reason,
        "type": rtype,
    }
    if tool_calls_parsed is not None:
        rec["tool_calls_parsed"] = tool_calls_parsed
    if rtype == "observation":
        rec["tool_call"] = {"name": tool_call_name}
    return rec


# ============================================================================
# _clean_prompt
# ============================================================================


class TestCleanPrompt:
    """Test chat template marker cleaning."""

    def test_qwen_markers(self):
        from examples.tool_rl.reward.reward import _clean_prompt

        text = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nWhat is the weather?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        result = _clean_prompt(text)
        assert "<|im_start|>" not in result
        assert "<|im_end|>" not in result
        assert "### system" in result
        assert "### user" in result
        assert "### assistant" in result
        assert "You are helpful." in result
        assert "What is the weather?" in result

    def test_preserves_content(self):
        from examples.tool_rl.reward.reward import _clean_prompt

        text = "<|im_start|>user\n请帮我查一下北京明天的天气和空气质量。<|im_end|>"
        result = _clean_prompt(text)
        assert "请帮我查一下北京明天的天气和空气质量" in result

    def test_collapses_excess_blank_lines(self):
        from examples.tool_rl.reward.reward import _clean_prompt

        text = "Hello\n\n\n\n\n\nWorld"
        result = _clean_prompt(text)
        # Max 3 consecutive newlines
        assert "\n\n\n\n" not in result


# ============================================================================
# _format_tools_md
# ============================================================================


class TestFormatToolsMd:
    """Test tool-to-markdown formatting."""

    @pytest.fixture
    def sample_tools(self) -> list[dict]:
        return [
            {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name.",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Temperature unit.",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city"],
                },
            },
            {
                "name": "send_email",
                "description": "Send an email.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                },
            },
        ]

    def test_formats_tool_table(self, sample_tools):
        from examples.tool_rl.reward.reward import _format_tools_md

        result = _format_tools_md(sample_tools)
        assert "### `get_weather`" in result
        assert "Get current weather for a city" in result
        assert "### `send_email`" in result
        assert "| Parameter | Type | Required | Description |" in result
        assert "| `city` | string | ✅ | City name. |" in result
        assert "enum: celsius, fahrenheit" in result

    def test_empty_tools(self):
        from examples.tool_rl.reward.reward import _format_tools_md

        assert _format_tools_md(None) == "_No tools available._"
        assert _format_tools_md([]) == "_No tools available._"

    def test_tool_no_params(self):
        from examples.tool_rl.reward.reward import _format_tools_md

        tools = [{"name": "ping", "description": "Ping the server."}]
        result = _format_tools_md(tools)
        assert "### `ping`" in result
        assert "Ping the server" in result
        # No parameter table when there are no properties
        assert "| Parameter |" not in result


# ============================================================================
# _format_agent_response_structured
# ============================================================================


class TestFormatAgentResponseStructured:
    """Test trajectory-to-structured-markdown conversion."""

    def test_single_turn_thinking_and_tool_calls(self):
        from examples.tool_rl.reward.reward import _format_agent_response_structured

        text = (
            "<think>I need to check weather for Beijing.</think>\n"
            "<tool_call>\n"
            "<function=get_weather>\n"
            '<parameter=city>"Beijing"</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        traj = [_make_trajectory_record(text=text)]
        context_md, eval_md = _format_agent_response_structured(traj)

        # Context should be empty for single-turn
        assert context_md == ""

        # Evaluation should have thinking and tool calls
        assert "### Thinking" in eval_md
        assert "I need to check weather for Beijing" in eval_md
        assert "### Tool Calls" in eval_md
        assert "```json" in eval_md

        # Tool calls should be parsed JSON
        json_match = re.search(r"```json\n(.*?)\n```", eval_md, re.DOTALL)
        assert json_match is not None
        calls = json.loads(json_match.group(1))
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"]["city"] == "Beijing"

    def test_uses_pre_parsed_tool_calls(self):
        from examples.tool_rl.reward.reward import _format_agent_response_structured

        pre_parsed = [{"name": "search", "arguments": {"query": "test"}}]
        traj = [
            _make_trajectory_record(
                text="<think>ok</think><tool_call>...</tool_call>",
                tool_calls_parsed=pre_parsed,
            ),
        ]
        _, eval_md = _format_agent_response_structured(traj)
        json_match = re.search(r"```json\n(.*?)\n```", eval_md, re.DOTALL)
        assert json_match is not None
        calls = json.loads(json_match.group(1))
        assert calls == pre_parsed

    def test_no_think_tag(self):
        from examples.tool_rl.reward.reward import _format_agent_response_structured

        text = "<tool_call><function=ping></function></tool_call>"
        traj = [_make_trajectory_record(text=text)]
        _, eval_md = _format_agent_response_structured(traj)
        # Should still have tool calls but no thinking section
        assert "### Thinking" not in eval_md
        assert "### Tool Calls" in eval_md

    def test_no_tool_calls(self):
        from examples.tool_rl.reward.reward import _format_agent_response_structured

        text = "<think>I don't know how to help.</think>"
        traj = [_make_trajectory_record(text=text)]
        _, eval_md = _format_agent_response_structured(traj)
        assert "### Thinking" in eval_md
        assert "no tool calls detected" in eval_md.lower()

    def test_multi_turn_last_only_evaluated(self):
        from examples.tool_rl.reward.reward import _format_agent_response_structured

        traj = [
            _make_trajectory_record(
                turn=0,
                text=(
                    "<think>First I'll search.</think>\n"
                    "<tool_call><function=search>"
                    '<parameter=query>"initial"</parameter>'
                    "</function></tool_call>"
                ),
                tool_calls_parsed=[{"name": "search", "arguments": {"query": "initial"}}],
            ),
            _make_trajectory_record(
                turn=0,
                text='{"results": ["found item"]}',
                rtype="observation",
                tool_call_name="search",
            ),
            _make_trajectory_record(
                turn=1,
                text=(
                    "<think>Now I'll read the file.</think>\n"
                    "<tool_call><function=read_file>"
                    '<parameter=file_path>"/tmp/result.txt"</parameter>'
                    "</function></tool_call>"
                ),
                tool_calls_parsed=[
                    {"name": "read_file", "arguments": {"file_path": "/tmp/result.txt"}},
                ],
            ),
        ]
        context_md, eval_md = _format_agent_response_structured(traj)

        # Previous turn should be in context
        assert "Turn 0" in context_md or "search" in context_md.lower()
        # Observation should be in context
        assert "found item" in context_md

        # Only the LAST turn in evaluation
        assert "read_file" in eval_md
        assert "First I'll search" not in eval_md

    def test_empty_trajectory(self):
        from examples.tool_rl.reward.reward import _format_agent_response_structured

        context_md, eval_md = _format_agent_response_structured([])
        assert eval_md == "(no response)"

    def test_only_observations(self):
        from examples.tool_rl.reward.reward import _format_agent_response_structured

        traj = [
            _make_trajectory_record(
                text='{"temp": 25}',
                rtype="observation",
                tool_call_name="get_weather",
            ),
        ]
        context_md, eval_md = _format_agent_response_structured(traj)
        assert "get_weather" in context_md
        assert eval_md == "(no response)"


# ============================================================================
# _build_rm_user_message
# ============================================================================


class TestBuildRmUserMessage:
    """Test complete RM user message assembly."""

    @pytest.fixture
    def sample_tools(self) -> list[dict]:
        return [
            {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name."},
                    },
                    "required": ["city"],
                },
            },
        ]

    def test_has_three_sections(self, sample_tools):
        from examples.tool_rl.reward.reward import _build_rm_user_message

        task_desc = "<|im_start|>user\nWhat's the weather in Beijing?<|im_end|>"
        traj = [
            _make_trajectory_record(
                text=(
                    "<think>Check weather.</think>\n"
                    "<tool_call><function=get_weather>"
                    '<parameter=city>"Beijing"</parameter>'
                    "</function></tool_call>"
                ),
            ),
        ]

        msg = _build_rm_user_message(
            task_desc,
            traj,
            available_tools=sample_tools,
            ground_truth_label="",
        )

        # Three core sections
        assert "## Conversation Context" in msg
        assert "## Available Tools" in msg
        assert "## Agent Response (to evaluate)" in msg

        # Thinking and tool calls present
        assert "### Thinking" in msg
        assert "Check weather" in msg
        assert "### Tool Calls" in msg
        assert '"name": "get_weather"' in msg

    def test_includes_ground_truth(self, sample_tools):
        from examples.tool_rl.reward.reward import _build_rm_user_message

        task_desc = "User: test"
        traj = [_make_trajectory_record(text="<think>ok</think>")]

        msg = _build_rm_user_message(
            task_desc,
            traj,
            available_tools=sample_tools,
            ground_truth_label="Correct: get_weather",
        )
        assert "## Ground Truth (Reference)" in msg
        assert "Correct: get_weather" in msg

    def test_no_ground_truth(self, sample_tools):
        from examples.tool_rl.reward.reward import _build_rm_user_message

        task_desc = "User: test"
        traj = [_make_trajectory_record(text="<think>ok</think>")]

        msg = _build_rm_user_message(task_desc, traj, available_tools=sample_tools, ground_truth_label="")
        assert "## Ground Truth" not in msg

    def test_no_tools(self):
        from examples.tool_rl.reward.reward import _build_rm_user_message

        task_desc = "User: test"
        traj = [_make_trajectory_record(text="<think>ok</think>")]

        msg = _build_rm_user_message(task_desc, traj, available_tools=None, ground_truth_label="")
        assert "_No tools available._" in msg

    def test_evaluate_instruction_present(self, sample_tools):
        from examples.tool_rl.reward.reward import _build_rm_user_message

        task_desc = "User: test"
        traj = [_make_trajectory_record(text="<think>ok</think>")]

        msg = _build_rm_user_message(
            task_desc,
            traj,
            available_tools=sample_tools,
            ground_truth_label="",
        )
        assert "Agent Response (to evaluate)" in msg
        assert "Output your evaluation as a JSON object" in msg
