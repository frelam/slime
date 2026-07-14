"""Hermes harness.

Hermes (https://hermesagent.org.cn/) is an open-source, self-hosted AI agent
by Nous Research. It uses an OpenAI-compatible API and runs as a CLI tool.

The harness installs Hermes in the sandbox and configures it to use the slime
adapter as its API backend. This works identically to the Codex harness since
both use OpenAI-compatible wire protocols.

Config is written via a base64 round-trip to avoid shell-quoting issues.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
from pathlib import Path

from slime.agent.sandbox import Sandbox

from .common import BaseHarness, HarnessContext, install_npm_cli, run_agent


class HermesHarness(BaseHarness):
    """Harness for the Hermes agent CLI (OpenAI-compatible)."""

    name = "hermes"

    # Host paths and CLI knobs, under the SLIME_AGENT_* prefix.
    node_tarball_env = "SLIME_AGENT_NODE_TARBALL"
    cli_tarball_env = "SLIME_AGENT_HERMES_TARBALL"
    extra_args_env = "SLIME_AGENT_HERMES_EXTRA_ARGS"
    extra_envs_env = "SLIME_AGENT_HERMES_EXTRA_ENVS"

    # Hermes exec flags — use non-interactive mode
    exec_flags = "--no-interactive"

    # Hermes TOML config template. Hermes uses OpenAI-compatible providers.
    # {model} and {base_url} are filled per run in write_config.
    config_toml = (
        'model = "{model}"\n'
        'model_provider = "slime"\n'
        "\n"
        "[model_providers.slime]\n"
        'name = "slime"\n'
        'base_url = "{base_url}"\n'
        'env_key = "OPENAI_API_KEY"\n'
        'wire_api = "chat"\n'
    )

    async def install_cli(self, sb: Sandbox) -> None:
        """Install the Hermes CLI in the sandbox.

        Requires ``SLIME_AGENT_NODE_TARBALL`` and ``SLIME_AGENT_HERMES_TARBALL``
        environment variables pointing to host tarballs.
        """
        await install_npm_cli(
            sb,
            node_runtime=Path(os.environ[self.node_tarball_env]),
            npm_package=Path(os.environ[self.cli_tarball_env]),
            check_cmd="hermes --version",
        )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        """Write the Hermes config file pointing to the slime adapter.

        Uses base64 encoding to avoid shell quoting issues with special
        characters in the config TOML.
        """
        toml = self.config_toml.format(
            model=ctx.model_label,
            base_url=f"{ctx.adapter_url}/v1",
        )
        toml_b64 = base64.b64encode(toml.encode("utf-8")).decode("ascii")

        await sb.exec(
            "mkdir -p /home/agent/.hermes && "
            f"echo {shlex.quote(toml_b64)} | base64 -d > /home/agent/.hermes/config.toml && "
            "chown -R agent:agent /home/agent/.hermes",
            user="root",
            check=True,
            timeout=60,
        )

    async def launch_and_wait(
        self,
        sb: Sandbox,
        ctx: HarnessContext,
        prompt: str,
        time_budget_sec: int,
    ) -> int:
        """Run Hermes in non-interactive mode and wait for completion.

        Hermes reads the API key from ``OPENAI_API_KEY`` and uses it as the
        session ID for the slime adapter.
        """
        # ``hermes exec`` — non-interactive entrypoint
        cmd = f"hermes exec {self.exec_flags} {shlex.quote(prompt)}"

        extra = os.environ.get(self.extra_args_env, "").strip()
        if extra:
            cmd = f"{cmd} {extra}"

        env = {
            # Hermes propagates OPENAI_API_KEY into Authorization: Bearer;
            # the slime adapter resolves the sid from that header.
            "OPENAI_API_KEY": ctx.session_id,
            "OPENAI_BASE_URL": f"{ctx.adapter_url}/v1",
        }

        # Extra env vars as a JSON object, merged last so callers can override
        extra_envs = os.environ.get(self.extra_envs_env, "").strip()
        if extra_envs:
            env.update(json.loads(extra_envs))

        return await run_agent(
            sb,
            workdir=ctx.workdir,
            start_cmd=cmd,
            env=env,
            time_budget_sec=time_budget_sec,
        )
