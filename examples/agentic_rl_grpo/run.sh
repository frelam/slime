#!/bin/bash
# =============================================================================
# GRPO/PPO Training Launcher for Agentic RL
# =============================================================================
#
# Architecture:
#   - General tasks (terminal_bench, cli_gym, tau_bench, api_bank,
#     agent_bench, ...) → Hermes harness + OpenAIAdapter (port 18002)
#   - SWE tasks (swe_gym_lite, r2e_gym) → Claude Code harness +
#     AnthropicAdapter (port 18001)
#
# Reward:
#   - General tasks: multi-dimensional reward (RM + verifier, 7 dims)
#   - SWE tasks: task evaluation reward only (test pass rate)
#
# Usage:
#   bash examples/agentic_rl_grpo/run.sh
#
# Required environment variables:
#   ADAPTER_PUBLIC_HOST         — host IP reachable by E2B sandboxes
#   SLIME_E2B_SANDBOX_IMAGE     — default E2B image for general tasks
#   SLIME_AGENT_NODE_TARBALL    — path to Node.js tarball
#   SLIME_AGENT_HERMES_TARBALL  — path to Hermes CLI tarball
#   SLIME_AGENT_CC_TARBALL      — path to Claude Code CLI tarball (SWE tasks)
# =============================================================================

set -euo pipefail

# ---- Source model config ----
# source scripts/models/qwen2.5-0.5B.sh

# ---- Multi-dimensional reward weights (general tasks only) ----
REWARD_WEIGHTS='{"correctness":0.51,"format":0.15,"tool_params":0.10,"retry":0.05,"planning":0.075,"hallucination":0.075,"tool_count":0.05}'

# ---- Training ----
python train_async.py \
    --advantage-estimator grpo \
    --loss-type policy_loss \
    \
    --custom-generate-function-path examples.agentic_rl_grpo.generate.agentic_grpo_generate \
    --custom-rm-path examples.agentic_rl_grpo.reward.agentic_grpo_reward \
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async \
    --rollout-global-dataset \
    \
    --n-samples-per-prompt 8 \
    --rollout-batch-size 32 \
    --rollout-max-context-len 32768 \
    --rollout-max-response-len 8192 \
    --rollout-temperature 1.0 \
    --rollout-top-p 1.0 \
    \
    --kl-coef 0.001 \
    --kl-loss-type k3 \
    --normalize-advantages \
    \
    --num-rollout 200 \
    --global-batch-size 256 \
    --num-steps-per-rollout 1 \
    --update-weights-interval 1 \
    \
    --rm-model-type sglang \
    --rm-system-prompt-dir examples/agentic_rl_grpo/prompts \
    --reward-weights "${REWARD_WEIGHTS}" \
    \
    --prompt-data /path/to/your/data.jsonl \
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --apply-chat-template \
    --rollout-shuffle \
    "$@"
