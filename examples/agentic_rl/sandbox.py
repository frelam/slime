"""Sandbox lifecycle management for agentic RL.

Provides two sandbox backends selectable at import time:

1. **SubprocessSandbox** — runs tasks in a local subprocess (no E2B dependency).
   Good for development, debugging, and benchmarks that don't need isolation.
2. **E2BSandbox** — wraps ``slime.agent.sandbox.E2BSandbox`` for cloud sandboxes.

Both implement the ``Sandbox`` protocol (``exec`` / ``write_file`` / ``read_file``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from slime.agent.sandbox import Sandbox
from slime.agent.sandbox import exec_and_wait  # noqa: F401  (re-export)

logger = logging.getLogger(__name__)


class SubprocessSandbox:
    """Minimal sandbox backed by local subprocess.

    All commands run as the current user.  For benchmarks that need real
    isolation (SWE-bench, CLI-Gym), use E2BSandbox instead.
    """

    def __init__(self, workdir: str | None = None):
        self._workdir = workdir or tempfile.mkdtemp(prefix="slime_agentic_")

    async def exec(
        self,
        cmd: str,
        user: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        check: bool = True,
    ) -> tuple[int, str, str]:
        """Run ``cmd`` in a subprocess at ``self._workdir``.

        Returns ``(exit_code, stdout, stderr)``.
        """
        merged_env = {**os.environ, **(env or {})}
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._workdir,
            env=merged_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return (-1, "", f"Timeout ({timeout}s)")

        text_stdout = stdout.decode("utf-8", errors="replace") if stdout else ""
        text_stderr = stderr.decode("utf-8", errors="replace") if stderr else ""
        exit_code = proc.returncode or 0

        if check and exit_code != 0:
            logger.warning(
                "SubprocessSandbox exec failed (exit=%d):\n  cmd=%s\n  stderr=%s",
                exit_code, cmd, text_stderr[-500:],
            )
        return exit_code, text_stdout, text_stderr

    async def write_file(self, path: str, content: str | bytes | Path, user: str | None = None) -> None:
        path = str(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(content, Path):
            import shutil
            shutil.copy2(content, path)
        elif isinstance(content, bytes):
            with open(path, "wb") as f:
                f.write(content)
        else:
            with open(path, "w") as f:
                f.write(content)

    async def read_file(self, path: str, user: str | None = None) -> str:
        with open(path) as f:
            return f.read()

    async def __aenter__(self) -> "SubprocessSandbox":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


def create_sandbox(args: Any) -> Sandbox:
    """Factory: return a suitable sandbox for the given args.

    Uses ``E2BSandbox`` when ``--e2b-sandbox-image`` is set; otherwise falls
    back to ``SubprocessSandbox``.
    """
    image = getattr(args, "e2b_sandbox_image", None) or os.environ.get("SLIME_E2B_SANDBOX_IMAGE")
    if image:
        from slime.agent.sandbox import E2BSandbox
        logger.info("Using E2BSandbox with image %s", image)
        return E2BSandbox(image)  # type: ignore[return-value]

    logger.info("Using SubprocessSandbox (no E2B image configured)")
    return SubprocessSandbox()
