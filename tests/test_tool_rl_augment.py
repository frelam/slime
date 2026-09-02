"""Tests for the data augmentation stage in examples/tool_rl/data/download_data.py.

Covers the four perturbation strategies applied to a fraction of positive
samples (label has tool calls):

- tool_rename:     rename a label tool in the prompt; label follows the name.
- desc_replace:    swap a label tool's description → negative sample (label emptied).
- param_rename:    rename schema parameter names; label argument keys follow.
- default_shuffle: randomise schema ``default`` values.
"""

from __future__ import annotations

import copy
import json
import random

import pytest


def _make_task() -> dict:
    """A minimal slime-format tool RL task with a labelled tool call."""
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "days": {
                        "type": "integer",
                        "description": "Forecast days",
                        "default": 3,
                    },
                },
                "required": ["city"],
            },
        },
        {
            "name": "send_email",
            "description": "Send an email",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient"},
                },
            },
        },
    ]
    gt = [{"name": "get_weather", "arguments": {"city": "Beijing", "days": 3}}]
    return {
        "messages": [
            {"role": "system", "content": "You can use get_weather when needed."},
            {"role": "user", "content": "What's the weather in Beijing?"},
        ],
        # deep copies so the task carries two independent tool lists,
        # exactly like the loaders produce them
        "tools": copy.deepcopy(tools),
        "label": 'Ground truth:\n  get_weather({"city": "Beijing", "days": 3})',
        "metadata": {
            "benchmark": "tool_rl",
            "source": "test",
            "task_id": "test-0",
            "ground_truth": copy.deepcopy(gt),
            "has_ground_truth": True,
            "tools": copy.deepcopy(tools),
        },
    }


class TestToolRename:
    def test_renames_tool_everywhere_and_label_follows(self):
        from examples.tool_rl.data.download_data import _augment_tool_rename

        task = _make_task()
        rng = random.Random(0)
        result = _augment_tool_rename(task, rng)

        assert result == "tool_rename"
        new = task["metadata"]["augment_detail"]["new"]
        old = task["metadata"]["augment_detail"]["old"]
        assert old == "get_weather"
        assert new != old

        # Both tool copies renamed
        for tool_list in (task["tools"], task["metadata"]["tools"]):
            names = [t["name"] for t in tool_list]
            assert new in names
            assert old not in names
            assert "send_email" in names  # untouched

        # Label ground truth follows the new name, arguments unchanged
        gt = task["metadata"]["ground_truth"]
        assert gt[0]["name"] == new
        assert gt[0]["arguments"] == {"city": "Beijing", "days": 3}

        # Label text + system prompt stay consistent
        assert old not in task["label"]
        assert new in task["label"]
        assert old not in task["messages"][0]["content"]

    def test_returns_none_without_label_candidates(self):
        from examples.tool_rl.data.download_data import _augment_tool_rename

        task = _make_task()
        task["metadata"]["ground_truth"] = []
        assert _augment_tool_rename(task, random.Random(0)) is None


class TestDescReplace:
    def test_empties_label_and_swaps_description(self):
        from examples.tool_rl.data.download_data import (
            _IRRELEVANT_DESCRIPTIONS,
            _augment_desc_replace,
        )

        task = _make_task()
        rng = random.Random(0)
        result = _augment_desc_replace(task, rng)

        assert result == "desc_replace"
        tool_name = task["metadata"]["augment_detail"]["tool"]

        # Negative sample: ground_truth is [] (not None — the reward
        # distinguishes "label says no tools" from "no label")
        assert task["metadata"]["ground_truth"] == []
        assert task["label"] == ""
        assert task["metadata"]["has_ground_truth"] is False

        # Description swapped in both copies, tool name kept
        for tool_list in (task["tools"], task["metadata"]["tools"]):
            t = next(t for t in tool_list if t["name"] == tool_name)
            assert t["description"] in _IRRELEVANT_DESCRIPTIONS

    def test_negative_sample_penalised_by_label_match(self):
        """Reward-side contract: label=[] + model calls a tool → negative dim1."""
        from examples.tool_rl.reward.verifier import match_tool_calls_against_label

        name, param = match_tool_calls_against_label(
            [{"name": "get_weather", "arguments": {"city": "Beijing"}}],
            [],
        )
        assert name < 0
        assert param < 0

        name, param = match_tool_calls_against_label([], [])
        assert (name, param) == (1.0, 1.0)


class TestParamRename:
    def test_renames_param_in_schema_and_label_keys(self):
        from examples.tool_rl.data.download_data import _augment_param_rename

        task = _make_task()
        rng = random.Random(0)
        result = _augment_param_rename(task, rng)

        assert result == "param_rename"
        detail = task["metadata"]["augment_detail"]
        old, new = detail["old"], detail["new"]
        assert old in ("city", "days")
        assert new != old

        # Schema renamed in both copies (+ required list synced)
        for tool_list in (task["tools"], task["metadata"]["tools"]):
            t = next(t for t in tool_list if t["name"] == "get_weather")
            props = t["parameters"]["properties"]
            assert new in props
            assert old not in props
            assert props[new]["description"]  # description preserved
            if old == "city":
                assert new in t["parameters"]["required"]

        # Label argument keys follow, values unchanged
        gt_args = task["metadata"]["ground_truth"][0]["arguments"]
        assert old not in gt_args
        if old == "city":
            assert gt_args[new] == "Beijing"
        else:
            assert gt_args[new] == 3

    def test_label_text_key_renamed(self):
        from examples.tool_rl.data.download_data import _augment_param_rename

        task = _make_task()
        rng = random.Random(0)
        _augment_param_rename(task, rng)
        detail = task["metadata"]["augment_detail"]
        if f'"{detail["old"]}"' in task["label"]:
            pytest.fail("old param name still present in label text")
        assert json.dumps(detail["new"]).strip('"') in task["label"]


class TestDefaultShuffle:
    def test_randomises_defaults_type_compatibly(self):
        from examples.tool_rl.data.download_data import _augment_default_shuffle

        task = _make_task()
        rng = random.Random(0)
        result = _augment_default_shuffle(task, rng)

        assert result == "default_shuffle"
        for tool_list in (task["tools"], task["metadata"]["tools"]):
            t = next(t for t in tool_list if t["name"] == "get_weather")
            default = t["parameters"]["properties"]["days"]["default"]
            assert default != 3
            assert isinstance(default, int)

        # Label ground truth untouched — real values are not defaults
        assert task["metadata"]["ground_truth"][0]["arguments"]["days"] == 3

    def test_returns_none_without_defaults(self):
        from examples.tool_rl.data.download_data import _augment_default_shuffle

        task = _make_task()
        for tool_list in (task["tools"], task["metadata"]["tools"]):
            for t in tool_list:
                for pinfo in t["parameters"]["properties"].values():
                    pinfo.pop("default", None)
        assert _augment_default_shuffle(task, random.Random(0)) is None


class TestAugmentTasks:
    def test_ratio_and_eligibility(self):
        from examples.tool_rl.data.download_data import augment_tasks

        tasks = [_make_task() for _ in range(20)]
        # 5 samples are ineligible (no ground truth calls)
        for i in range(5):
            tasks[i]["metadata"]["ground_truth"] = []

        rng = random.Random(42)
        n = augment_tasks(tasks, 0.15, rng)

        assert n == max(1, round(20 * 0.15))
        augmented = [t for t in tasks if "augmented" in t["metadata"]]
        assert len(augmented) == n
        # Ineligible samples never augmented
        assert all("augmented" not in tasks[i]["metadata"] for i in range(5))
        # Every augmented sample carries a known strategy tag
        assert {t["metadata"]["augmented"] for t in augmented} <= {
            "tool_rename",
            "desc_replace",
            "param_rename",
            "default_shuffle",
        }

    def test_deterministic_with_same_seed(self):
        from examples.tool_rl.data.download_data import augment_tasks

        tasks_a = [_make_task() for _ in range(10)]
        tasks_b = [_make_task() for _ in range(10)]
        augment_tasks(tasks_a, 0.5, random.Random(7))
        augment_tasks(tasks_b, 0.5, random.Random(7))
        assert json.dumps(tasks_a, sort_keys=True) == json.dumps(tasks_b, sort_keys=True)

    def test_zero_ratio_noop(self):
        from examples.tool_rl.data.download_data import augment_tasks

        tasks = [_make_task() for _ in range(5)]
        assert augment_tasks(tasks, 0.0, random.Random(0)) == 0
        assert all("augmented" not in t["metadata"] for t in tasks)
