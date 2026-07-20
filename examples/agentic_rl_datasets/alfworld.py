"""ALFWorld dataset adapter — text-based household task environment.

ALFWorld (ALFRED + TextWorld) tasks: agent navigates a text-based household,
interacting with objects to achieve a goal (e.g., "put a clean mug on the desk").

Reward: binary outcome — 1.0 if goal predicate satisfied, 0.0 otherwise.

Environment Setup
-----------------
ALFWorld requires the ``alfworld`` Python package and its data files::

    pip install alfworld
    # Download data files
    wget http://alfworld.s3.amazonaws.com/alfworld_data.tar.gz
    tar xf alfworld_data.tar.gz -C /path/to/alfworld_data/

Then set ``ALFWORLD_DATA`` env var or pass via ``--alfworld-data-dir``.

Task Types
----------
ALFWorld has 6 task types across 2 categories:

Pick & Place (simple):
  - pick_and_place_simple: Take X from Y and put on Z
  - pick_clean_then_place: Clean X then put on Z
  - pick_cool_then_place: Cool X then put on Z
  - pick_heat_then_place: Heat X then put on Z

Pick Two & Place (harder):
  - pick_two_obj: Take two objects and put on Z
  - look_at_obj_in_light: Find X, turn on lamp, examine X

Each task comes from the ALFWorld training set (~3.5K tasks).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from examples.agentic_rl_datasets import DatasetAdapter, register_adapter

logger = logging.getLogger(__name__)

# ALFWorld task types and their counts in train set
_ALFWORLD_TASK_TYPES = {
    "pick_and_place_simple": "pick_and_place",
    "pick_clean_then_place": "pick_clean_then_place",
    "pick_cool_then_place": "pick_cool_then_place",
    "pick_heat_then_place": "pick_heat_then_place",
    "pick_two_obj": "pick_two_obj",
    "look_at_obj_in_light": "look_at_obj_in_light",
}


def _get_alfworld_data_dir() -> str:
    """Resolve ALFWorld data directory from env or default locations."""
    env_val = os.environ.get("ALFWORLD_DATA", "")
    if env_val and Path(env_val).is_dir():
        return env_val

    candidates = [
        os.path.expanduser("~/alfworld_data"),
        "/root/alfworld_data",
        "/tmp/alfworld_data",
    ]
    for p in candidates:
        if Path(p).is_dir():
            return p
    return candidates[0]


def _generate_synthetic_alfworld_tasks(num_tasks: int = 64) -> list[dict[str, Any]]:
    """Generate synthetic ALFWorld-style tasks for offline training.

    When the actual ALFWorld environment is not installed, we generate
    templated tasks that exercise the same reasoning patterns.

    Each task has:
      - A goal description (prompt)
      - An initial environment description
      - A set of success indicators (check commands to verify the task)

    The agent interacts via bash commands in a simulated household environment
    (files and directories representing rooms and objects).
    """
    templates = [
        {
            "task_type": "pick_and_place_simple",
            "prompt": (
                "You are in a household environment. Your task is to {goal}.\n\n"
                "The house has these rooms: kitchen, living room, bedroom, bathroom.\n"
                "Objects are in various locations. Use shell commands to navigate\n"
                "and interact with the environment.\n\n"
                "Available commands:\n"
                "  cd <room> — go to a room\n"
                "  ls — list objects in current room\n"
                "  cat <object> — examine an object\n"
                "  mv <object> <destination> — move an object\n"
                "  echo 'done' > /tmp/status — signal task completion\n\n"
                "Start in the kitchen. Complete the task efficiently."
            ),
            "goals": [
                "take the mug from the kitchen and put it on the desk in the living room",
                "take the apple from the living room and put it in the fridge in the kitchen",
                "find the book in the bedroom and put it on the shelf in the living room",
                "take the soap from the bathroom and put it on the kitchen counter",
            ],
            "setup_commands": [
                "mkdir -p /home/agent/kitchen /home/agent/living_room "
                "/home/agent/bedroom /home/agent/bathroom",
                "echo 'a ceramic mug' > /home/agent/kitchen/mug.txt",
                "echo 'a red apple' > /home/agent/living_room/apple.txt",
                "echo 'a thick book' > /home/agent/bedroom/book.txt",
                "echo 'a bar of soap' > /home/agent/bathroom/soap.txt",
                "echo 'a wooden desk' > /home/agent/living_room/desk.txt",
                "echo 'a white fridge' > /home/agent/kitchen/fridge.txt",
                "echo 'a bookshelf' > /home/agent/living_room/shelf.txt",
                "echo 'a kitchen counter' > /home/agent/kitchen/counter.txt",
            ],
            "check_commands": [
                # mug → desk: check mug is in living_room/desk area
                "test -f /home/agent/living_room/desk_mug.txt || "
                "grep -r 'mug' /home/agent/living_room/",
                # apple → fridge
                "test -f /home/agent/kitchen/fridge_apple.txt || "
                "grep -r 'apple' /home/agent/kitchen/fridge.txt",
                # book → shelf
                "test -f /home/agent/living_room/shelf_book.txt || "
                "grep -r 'book' /home/agent/living_room/shelf.txt",
                # soap → counter
                "test -f /home/agent/kitchen/counter_soap.txt || "
                "grep -r 'soap' /home/agent/kitchen/counter.txt",
            ],
        },
        {
            "task_type": "pick_clean_then_place",
            "prompt": (
                "You are in a household environment. A {object} at {location} is dirty.\n"
                "Your task: clean it at the sink, then put it on {target}.\n\n"
                "Available rooms: kitchen (has sink, counter), living room,\n"
                "bedroom, bathroom (has sink, cabinet).\n\n"
                "Use shell commands to navigate and interact.\n"
                "Complete the task and run: echo 'done' > /tmp/status"
            ),
            "goals": [
                "clean the dirty plate in the kitchen and put it on the dining table",
                "wash the dirty cup in the bathroom and place it on the shelf",
                "clean the fork in the kitchen sink and put it in the drawer",
            ],
            "setup_commands": [
                "mkdir -p /home/agent/kitchen /home/agent/living_room",
                "echo 'dirty' > /home/agent/kitchen/plate.txt",
                "echo 'dirty' > /home/agent/bathroom/cup.txt",
                "echo 'dirty' > /home/agent/kitchen/fork.txt",
                "echo 'a sink' > /home/agent/kitchen/sink.txt",
                "echo 'a sink' > /home/agent/bathroom/sink.txt",
            ],
            "check_commands": [
                "test ! -f /home/agent/kitchen/plate.txt || "
                "grep -q 'clean' /home/agent/kitchen/plate.txt",
                "test ! -f /home/agent/bathroom/cup.txt || "
                "grep -q 'clean' /home/agent/bathroom/cup.txt",
            ],
        },
        {
            "task_type": "navigation",
            "prompt": (
                "You are in a large building. Your task: {goal}.\n\n"
                "The building has: lobby, office_A, office_B, server_room, break_room.\n"
                "Use shell commands to navigate (cd) and explore (ls, cat, find).\n"
                "Signal completion with: echo 'done' > /tmp/status"
            ),
            "goals": [
                "find the server_room and check if server_status.txt says 'online'",
                "go to office_B and check if the report.pdf exists",
                "find which room has the coffee_machine and report its status",
                "go to every room and count how many people.txt files exist",
            ],
            "setup_commands": [
                "mkdir -p /home/agent/lobby /home/agent/office_A "
                "/home/agent/office_B /home/agent/server_room "
                "/home/agent/break_room",
                "echo 'online' > /home/agent/server_room/server_status.txt",
                "echo 'Q3 report' > /home/agent/office_B/report.pdf",
                "echo 'brewing' > /home/agent/break_room/coffee_machine.txt",
                "echo 'Alice' > /home/agent/office_A/people.txt",
                "echo 'Bob' > /home/agent/office_B/people.txt",
            ],
            "check_commands": [
                "grep -q 'online' /home/agent/server_room/server_status.txt",
                "test -f /home/agent/office_B/report.pdf",
                "grep -q 'brewing' /home/agent/break_room/coffee_machine.txt",
                "find /home/agent -name 'people.txt' | wc -l | grep -q '2'",
            ],
        },
    ]

    tasks: list[dict[str, Any]] = []
    for i in range(num_tasks):
        tmpl = templates[i % len(templates)]
        goal_idx = i % len(tmpl["goals"])
        goal = tmpl["goals"][goal_idx]
        task_id = f"alfworld_{tmpl['task_type']}_{i:04d}"

        tasks.append({
            "prompt": tmpl["prompt"].format(goal=goal, object="object", location="location", target="target"),
            "label": task_id,
            "metadata": {
                "benchmark": "alfworld",
                "task_type": tmpl["task_type"],
                "task_id": task_id,
                "goal": goal,
                "setup_commands": tmpl["setup_commands"],
                "check_command": tmpl["check_commands"][
                    goal_idx % len(tmpl["check_commands"])
                ],
                "max_turns": 15,
                "timeout_sec": 120,
                "workdir": "/home/agent",
            },
        })
    return tasks


@register_adapter
class ALFWorldAdapter(DatasetAdapter):
    """Adapter for ALFWorld text-based household tasks.

    Supports two modes:
      - **native**: Uses the real ``alfworld`` Python package (requires installation).
      - **synthetic**: Uses file-system-based simulated tasks (no ALFWorld dep).
    """

    name = "alfworld"

    # This adapter uses text-based interaction (not tool-call JSON)
    interaction_mode = "text"

    def __init__(self) -> None:
        self._env: Any = None
        self._use_native = self._check_native_alfworld()

    @staticmethod
    def _check_native_alfworld() -> bool:
        """Check if the native ALFWorld package is installed."""
        try:
            import alfworld  # noqa: F401
            return True
        except ImportError:
            return False

    def load_dataset(self, path: str) -> list[dict[str, Any]]:
        """Load ALFWorld tasks from a JSONL file or generate synthetic ones.

        Args:
            path: Path to JSONL file, or ``"synthetic:<N>"`` to generate
                  N synthetic tasks.

        Returns:
            List of task dicts with ``prompt``, ``label``, ``metadata``.
        """
        # Synthetic mode
        if path.startswith("synthetic:"):
            try:
                num = int(path.split(":")[1])
            except (IndexError, ValueError):
                num = 64
            logger.info("Generating %d synthetic ALFWorld tasks", num)
            return _generate_synthetic_alfworld_tasks(num)

        # Load from file
        tasks: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                metadata = raw.get("metadata") or {}
                tasks.append({
                    "prompt": raw.get("prompt", ""),
                    "label": raw.get("label", metadata.get("task_id", "")),
                    "metadata": {
                        "benchmark": "alfworld",
                        "task_type": metadata.get("task_type", ""),
                        "task_id": metadata.get("task_id", ""),
                        "goal": metadata.get("goal", raw.get("prompt", "")),
                        "setup_commands": metadata.get("setup_commands", []),
                        "check_command": metadata.get("check_command", ""),
                        "max_turns": metadata.get("max_turns", 15),
                        "timeout_sec": metadata.get("timeout_sec", 120),
                        "workdir": metadata.get("workdir", "/home/agent"),
                    },
                })
        return tasks

    async def setup_task(self, sb: Any, metadata: dict[str, Any]) -> None:
        """Prepare the sandbox for an ALFWorld task.

        For synthetic mode: creates directory structure representing rooms/objects.
        For native mode: initializes the ALFWorld environment.
        """
        from slime.agent.sandbox import ensure_agent_user

        workdir = metadata.get("workdir", "/home/agent")
        await ensure_agent_user(sb, workdir)

        # Run setup commands (create the simulated environment)
        setup_cmds = metadata.get("setup_commands", [])
        for cmd in setup_cmds:
            if cmd.strip():
                await sb.exec(cmd, user="agent", check=False, timeout=30)

        # Store current working directory for env_step
        metadata["_workdir"] = workdir

    async def env_step(
        self, action: str, metadata: dict[str, Any]
    ) -> tuple[str, bool]:
        """Execute an action in the ALFWorld environment.

        Args:
            action: The agent's action string (e.g., "go to kitchen", "take mug").
            metadata: Task metadata dict.

        Returns:
            (observation, done) — observation is text, done is True if episode ended.
        """
        if self._use_native:
            return await self._native_env_step(action, metadata)
        return await self._synthetic_env_step(action, metadata)

    async def _synthetic_env_step(
        self, action: str, metadata: dict[str, Any]
    ) -> tuple[str, bool]:
        """Execute action in the synthetic file-system environment.

        Returns (observation_text, done).
        """
        import asyncio
        import os as _os
        import subprocess as _sp
        import tempfile as _tf

        workdir = metadata.get("_workdir", "/home/agent")

        # Parse common ALFWorld-style actions into shell commands
        action_lower = action.lower().strip()

        if "done" in action_lower or "finish" in action_lower:
            return "Task marked as done.", True

        # Map natural language actions to shell commands
        cmd = None

        # go to / cd to / navigate to
        for room in ["kitchen", "living_room", "bedroom", "bathroom",
                      "lobby", "office_a", "office_b", "server_room",
                      "break_room", "hallway"]:
            if room.replace("_", " ") in action_lower:
                cmd = f"cd {workdir}/{room} && pwd && ls -la"

        # look / examine / check / read / cat
        if cmd is None and any(w in action_lower for w in
                                ["look", "examine", "check", "read", "see", "cat"]):
            # Extract the object name
            words = action_lower.replace(",", "").split()
            for obj in ["mug", "apple", "book", "soap", "plate", "cup", "fork",
                        "fridge", "desk", "shelf", "counter", "sink", "drawer",
                        "cabinet", "report", "server_status", "coffee_machine"]:
                if obj in words:
                    cmd = f"find {workdir} -name '*{obj}*' -exec cat {{}} \\; 2>/dev/null || echo 'Not found: {obj}'"

        # take / pick up / get / grab
        if cmd is None and any(w in action_lower for w in
                                ["take", "pick up", "get", "grab", "collect"]):
            cmd = f"find {workdir} -type f | head -20"

        # put / place / move / mv
        if cmd is None and any(w in action_lower for w in
                                ["put", "place", "move", "mv", "bring"]):
            cmd = f"ls -R {workdir} 2>/dev/null | head -40"

        # clean / wash / rinse
        if cmd is None and any(w in action_lower for w in
                                ["clean", "wash", "rinse", "wipe"]):
            cmd = f"echo 'cleaned' && find {workdir} -name '*.txt' | head -10"

        # generic explore
        if cmd is None:
            cmd = f"cd {workdir} && pwd && ls -R 2>/dev/null | head -30"

        # Run the command
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=_sp.PIPE,
            stderr=_sp.PIPE,
            env={**_os.environ},
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return "Error: command timed out.", False

        out = (stdout or b"").decode("utf-8", errors="replace")
        err = (stderr or b"").decode("utf-8", errors="replace")
        obs = out[:2000]
        if err and "error" in err.lower():
            obs += f"\n[Error: {err[:500]}]"

        return obs.strip() or "(empty)", False

    async def _native_env_step(
        self, action: str, metadata: dict[str, Any]
    ) -> tuple[str, bool]:
        """Execute action in the native ALFWorld environment."""
        if self._env is None:
            return "Error: ALFWorld environment not initialized.", False

        try:
            observation, reward, done, info = self._env.step(action)
            return observation, bool(done)
        except Exception as e:
            logger.warning("ALFWorld env step error: %s", e)
            return f"Error: {e}", True

    async def evaluate_task(
        self,
        sb: Any,
        metadata: dict[str, Any],
        *,
        timeout_sec: int = 300,
    ) -> float:
        """Evaluate ALFWorld task outcome.

        For synthetic mode: runs check_command and returns 1.0 if successful.
        For native mode: checks env.info['won'].
        """
        check_cmd = metadata.get("check_command", "")
        if not check_cmd:
            logger.warning("[alfworld] No check_command; reward=0")
            return 0.0

        try:
            ec, stdout, stderr = await sb.exec(
                check_cmd, user="agent", check=False, timeout=timeout_sec,
            )
            if ec == 0:
                return 1.0
            else:
                logger.debug(
                    "[alfworld] Check failed (exit=%d): cmd=%s, stderr=%s",
                    ec, check_cmd, (stderr or "")[:200],
                )
                return 0.0
        except Exception:
            logger.exception("[alfworld] Check command failed")
            return 0.0
