"""Tests for tool call loss mask functionality in tool_rl.

Tests ``get_incorrect_tool_call_spans`` and ``_build_tool_aware_loss_mask``.
"""

from __future__ import annotations

import pytest
import torch


# ============================================================================
# get_incorrect_tool_call_spans
# ============================================================================


class TestGetIncorrectToolCallSpans:
    """Test the per-tool-call correctness span detection."""

    @pytest.fixture
    def available_tools(self) -> list[dict]:
        return [
            {
                "name": "search_files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "pattern": {"type": "string"},
                    },
                },
            },
            {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "offset": {"type": "integer"},
                    },
                },
            },
        ]

    def _get_spans(self, text: str, tools=None):
        from examples.tool_rl.reward.verifier import get_incorrect_tool_call_spans

        return get_incorrect_tool_call_spans(text, available_tools=tools)

    # ---- Correct tool calls ----

    def test_all_correct_returns_empty(self, available_tools):
        text = (
            "<think>I need to search files.</think>\n"
            "<tool_call>\n"
            "<function=search_files>\n"
            '<parameter=path>"/src"</parameter>\n'
            '<parameter=pattern>"*.py"</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        spans = self._get_spans(text, available_tools)
        assert spans == []

    def test_multiple_correct_tool_calls(self, available_tools):
        text = (
            "<think>I need to search and read.</think>\n"
            "<tool_call>\n"
            "<function=search_files>\n"
            '<parameter=path>"/src"</parameter>\n'
            '<parameter=pattern>"*.py"</parameter>\n'
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=read_file>\n"
            '<parameter=file_path>"/src/main.py"</parameter>\n'
            "<parameter=offset>0</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        spans = self._get_spans(text, available_tools)
        assert spans == []

    # ---- Incorrect: wrong function name ----

    def test_wrong_function_name_returns_span(self, available_tools):
        text = (
            "<think>I need to use a tool.</think>\n"
            "<tool_call>\n"
            "<function=nonexistent_func>\n"
            '<parameter=path>"/src"</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        spans = self._get_spans(text, available_tools)
        assert len(spans) == 1
        start, end = spans[0]
        # Verify the span corresponds to the <tool_call> block
        assert "<tool_call>" in text[start:end]
        assert "</tool_call>" in text[start:end]

    # ---- Incorrect: wrong parameter name ----

    def test_wrong_param_name_returns_span(self, available_tools):
        text = (
            "<think>Searching.</think>\n"
            "<tool_call>\n"
            "<function=search_files>\n"
            '<parameter=wrong_param>"/src"</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        spans = self._get_spans(text, available_tools)
        assert len(spans) == 1

    # ---- Incorrect: wrong parameter type ----

    def test_wrong_param_type_returns_span(self, available_tools):
        text = (
            "<think>Reading file.</think>\n"
            "<tool_call>\n"
            "<function=read_file>\n"
            '<parameter=file_path>"/src/main.py"</parameter>\n'
            '<parameter=offset>"not_an_integer"</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        spans = self._get_spans(text, available_tools)
        assert len(spans) == 1

    # ---- Edge cases ----

    def test_no_tools_available_returns_empty(self, available_tools):
        """When no tool definitions, all calls are treated as correct."""
        text = (
            "<think>Using tool.</think>\n"
            "<tool_call>\n"
            "<function=unknown_func>\n"
            '<parameter=x>"y"</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        spans = self._get_spans(text, None)
        assert spans == []

    def test_empty_tools_returns_empty(self, available_tools):
        text = (
            "<tool_call>\n"
            "<function=any_func>\n"
            "</function>\n"
            "</tool_call>"
        )
        spans = self._get_spans(text, [])
        assert spans == []

    def test_no_tool_calls_returns_empty(self, available_tools):
        text = "<think>No tools needed for this task.</think>"
        spans = self._get_spans(text, available_tools)
        assert spans == []

    def test_garbled_output_returns_empty(self, available_tools):
        """Garbled text with no valid tool call blocks should return empty."""
        text = "asdfghjkl qwerty 12345 !@#$%"
        spans = self._get_spans(text, available_tools)
        assert spans == []

    # ---- Mixed correct + incorrect ----

    def test_mixed_correct_and_incorrect(self, available_tools):
        text = (
            "<think>Let me search and then read.</think>\n"
            "<tool_call>\n"
            "<function=search_files>\n"
            '<parameter=path>"/src"</parameter>\n'
            '<parameter=pattern>"*.py"</parameter>\n'
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=bad_func>\n"
            '<parameter=x>"y"</parameter>\n'
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=read_file>\n"
            '<parameter=file_path>"/src/main.py"</parameter>\n'
            "<parameter=offset>0</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        spans = self._get_spans(text, available_tools)
        assert len(spans) == 1  # Only the middle "bad_func" is incorrect
        start, end = spans[0]
        assert "bad_func" in text[start:end]


# ============================================================================
# _build_tool_aware_loss_mask
# ============================================================================


class TestBuildToolAwareLossMask:
    """Test the loss mask construction.

    Encoding: 2=normal token, 1=incorrect tool call token.
    """

    @pytest.fixture
    def available_tools(self) -> list[dict]:
        return [
            {
                "name": "search_files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                },
            },
        ]

    def _build_mask(self, text, response_len, tokenizer, tools=None, enable=True):
        from examples.tool_rl.generate import _build_tool_aware_loss_mask

        return _build_tool_aware_loss_mask(
            response_text=text,
            response_len=response_len,
            tokenizer=tokenizer,
            available_tools=tools,
            enable_masking=enable,
        )

    def test_masking_disabled_returns_all_twos(self, available_tools):
        """When enable_masking=False, should return [2] * response_len."""
        text = "some text here"
        mask = self._build_mask(
            text, response_len=3, tokenizer=None, tools=available_tools, enable=False,
        )
        assert mask == [2, 2, 2]

    def test_no_tokenizer_returns_all_twos(self, available_tools):
        """Without tokenizer, fallback to all 2 (normal)."""
        mask = self._build_mask(
            "text", response_len=3, tokenizer=None, tools=available_tools,
        )
        assert mask == [2, 2, 2]

    def test_no_incorrect_calls_returns_all_twos(self, available_tools):
        """When all tool calls are correct, all tokens should be 2."""
        # Use a fake tokenizer that returns offsets
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
                tokens = text.split()
                return {
                    "input_ids": list(range(len(tokens))),
                    "offset_mapping": [(0, 0)] * len(tokens),  # All special tokens — stay as 2
                }

        text = (
            "<think>Planning</think>\n"
            "<tool_call>\n"
            "<function=search_files>\n"
            '<parameter=path>"/src"</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        mask = self._build_mask(
            text, response_len=5, tokenizer=FakeTokenizer(), tools=available_tools,
        )
        assert all(m == 2 for m in mask)
        assert len(mask) == 5


# ============================================================================
# _is_tool_call_correct
# ============================================================================


class TestIsToolCallCorrect:
    """Test individual tool call correctness checks."""

    @pytest.fixture
    def tool_index(self) -> tuple:
        from examples.tool_rl.reward.verifier import _build_tool_index

        tools = [
            {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                },
            },
            {
                "name": "no_params_func",
                "parameters": {"type": "object", "properties": {}},
            },
        ]
        return _build_tool_index(tools)

    def _check(self, call: dict, tool_names, tool_params):
        from examples.tool_rl.reward.verifier import _is_tool_call_correct

        return _is_tool_call_correct(call, tool_names, tool_params)

    def test_correct_call(self, tool_index):
        tool_names, tool_params = tool_index
        assert self._check(
            {"name": "search", "arguments": {"query": "test", "limit": 5}},
            tool_names, tool_params,
        )

    def test_wrong_name(self, tool_index):
        tool_names, tool_params = tool_index
        assert not self._check(
            {"name": "unknown", "arguments": {}},
            tool_names, tool_params,
        )

    def test_wrong_param_name(self, tool_index):
        tool_names, tool_params = tool_index
        assert not self._check(
            {"name": "search", "arguments": {"wrong_key": "val"}},
            tool_names, tool_params,
        )

    def test_wrong_param_type(self, tool_index):
        tool_names, tool_params = tool_index
        assert not self._check(
            {"name": "search", "arguments": {"query": "ok", "limit": "not_int"}},
            tool_names, tool_params,
        )

    def test_extra_param(self, tool_index):
        tool_names, tool_params = tool_index
        assert not self._check(
            {"name": "search", "arguments": {"query": "test", "extra": 123}},
            tool_names, tool_params,
        )

    def test_no_params_expected_but_given(self, tool_index):
        tool_names, tool_params = tool_index
        assert not self._check(
            {"name": "no_params_func", "arguments": {"x": "y"}},
            tool_names, tool_params,
        )

    def test_no_params_expected_none_given(self, tool_index):
        tool_names, tool_params = tool_index
        assert self._check(
            {"name": "no_params_func", "arguments": {}},
            tool_names, tool_params,
        )

    def test_empty_tool_index_returns_true(self, tool_index):
        """With no tool definitions, any call is 'correct' (can't verify)."""
        assert self._check(
            {"name": "anything", "arguments": {"x": "y"}},
            set(), {},
        )


# ============================================================================
# tool_rl_tis_function — advantage-conditioned masking
# ============================================================================


class TestToolRLTisFunction:
    """Test the custom TIS function for advantage-conditioned tool masking."""

    @pytest.fixture
    def mock_args(self):
        """Create a mock args object with the required attributes."""

        class MockArgs:
            mask_failed_tool_calls = True
            mask_failed_tool_calls_adv_conditioned = True

        return MockArgs()

    def _run_tis(
        self,
        args,
        pg_loss: list[float] | None = None,
        loss_masks: list[list[int]] | None = None,
        response_lengths: list[int] | None = None,
    ):
        from examples.tool_rl.tis import tool_rl_tis_function

        pg_loss_t = (
            torch.tensor(pg_loss, dtype=torch.float32)
            if pg_loss
            else torch.tensor([], dtype=torch.float32)
        )
        masks = [torch.tensor(m, dtype=torch.int) for m in (loss_masks or [])]
        train_lp = [torch.zeros_like(m, dtype=torch.float32) for m in masks]
        rollout_lp = [torch.zeros_like(m, dtype=torch.float32) for m in masks]

        return tool_rl_tis_function(
            args,
            pg_loss=pg_loss_t,
            train_log_probs=train_lp,
            rollout_log_probs=rollout_lp,
            loss_masks=masks,
            response_lengths=response_lengths or [],
        )

    def test_feature_disabled_when_flag_false(self):
        """When mask_failed_tool_calls is False, masks pass through unchanged."""

        class OffArgs:
            mask_failed_tool_calls = False
            mask_failed_tool_calls_adv_conditioned = True

        masks = [[2, 1, 2]]
        _, result, _ = self._run_tis(
            OffArgs(),
            pg_loss=[-0.5, -0.3, 0.1],
            loss_masks=masks,
            response_lengths=[3],
        )
        assert result[0].tolist() == [2, 1, 2]

    def test_adv_positive_masks_incorrect_tokens(self, mock_args):
        """Advantage > 0 (pg_loss < 0): 1→0 for incorrect tool tokens."""
        masks = [[2, 1, 2, 1]]
        pg = [-0.5, -0.3, -0.1, -0.2]  # All negative → all adv > 0
        _, result, _ = self._run_tis(
            mock_args,
            pg_loss=pg,
            loss_masks=masks,
            response_lengths=[4],
        )
        assert result[0].tolist() == [2, 0, 2, 0]

    def test_adv_negative_keeps_incorrect_tokens(self, mock_args):
        """Advantage <= 0 (pg_loss >= 0): 1→2 for incorrect tool tokens."""
        masks = [[2, 1, 2, 1]]
        pg = [0.5, 0.3, 0.1, 0.2]  # All positive → all adv <= 0
        _, result, _ = self._run_tis(
            mock_args,
            pg_loss=pg,
            loss_masks=masks,
            response_lengths=[4],
        )
        assert result[0].tolist() == [2, 2, 2, 2]

    def test_mixed_advantage_per_sample(self, mock_args):
        """Sample 1 advantage > 0 (mask), Sample 2 advantage <= 0 (keep)."""
        masks = [[2, 1, 2], [2, 1, 1]]
        pg = [-0.5, -0.3, -0.1, 0.5, 0.3, 0.1]
        _, result, _ = self._run_tis(
            mock_args,
            pg_loss=pg,
            loss_masks=masks,
            response_lengths=[3, 3],
        )
        assert result[0].tolist() == [2, 0, 2]
        assert result[1].tolist() == [2, 2, 2]

    def test_no_tagged_tokens_passes_through(self, mock_args):
        """When all masks are 2 (no incorrect tool calls), nothing changes."""
        masks = [[2, 2, 2], [2, 2]]
        pg = [-0.5, -0.3, -0.1, -0.2, -0.1]
        _, result, _ = self._run_tis(
            mock_args,
            pg_loss=pg,
            loss_masks=masks,
            response_lengths=[3, 2],
        )
        assert result[0].tolist() == [2, 2, 2]
        assert result[1].tolist() == [2, 2]

    def test_adv_conditioned_off_keeps_encoding(self):
        """When adv_conditioned is False but mask_failed_tool_calls is True,
        masks pass through unchanged (the 1 values are not modified)."""

        class NoAdvArgs:
            mask_failed_tool_calls = True
            mask_failed_tool_calls_adv_conditioned = False

        masks = [[2, 1, 2]]
        _, result, _ = self._run_tis(
            NoAdvArgs(),
            pg_loss=[-0.5, -0.3, 0.1],
            loss_masks=masks,
            response_lengths=[3],
        )
        assert result[0].tolist() == [2, 1, 2]

    def test_no_response_lengths_graceful_fallback(self, mock_args):
        """When response_lengths is None, should pass through unchanged."""
        from examples.tool_rl.tis import tool_rl_tis_function

        masks = [torch.tensor([2, 1, 2], dtype=torch.int)]
        train_lp = [torch.zeros(3, dtype=torch.float32)]
        rollout_lp = [torch.zeros(3, dtype=torch.float32)]

        _, result, _ = tool_rl_tis_function(
            mock_args,
            pg_loss=torch.tensor([-0.5, -0.3, 0.1]),
            train_log_probs=train_lp,
            rollout_log_probs=rollout_lp,
            loss_masks=masks,
            response_lengths=None,
        )
        assert result[0].tolist() == [2, 1, 2]
