#!/bin/bash
# =============================================================================
# GRPO/PPO Training Launcher for Agentic RL
# =============================================================================
#
# Target: 2x V100 32GB with 4B model (colocate, release-train)
# Reward: RM + Verifier mixed (7 dimensions)
#
# Architecture:
#   - General tasks (terminal_bench, cli_gym, tau_bench, api_bank,
#     agent_bench, ...) → Hermes harness + OpenAIAdapter (port 18002)
#   - SWE tasks (swe_gym_lite, r2e_gym) → Claude Code harness +
#     AnthropicAdapter (port 18001)
#
# Agent modes (SLIME_AGENT_MODE env var):
#   - sandbox   (default): E2B sandbox + Hermes/Claude Code harness
#   - sglang_loop:         Local SGLang agent loop, no Docker/E2B needed
#                           (use for NPU smoke testing or machines without Docker)
#
# Reward:
#   - General tasks: multi-dimensional reward (RM + verifier, 7 dims)
#   - SWE tasks: task evaluation reward only (test pass rate)
#
# Memory layout (colocate + release-train):
#   Phase 1: [Megatron] train (~24GB/GPU) --release→ free
#   Phase 2: [SGLang TP=2] rollout + RM (~18GB/GPU KV pool) --offload→ free
#   Phase 3: repeat
#
# Usage:
#   # Production (E2B + Hermes):
#   export SLIME_AGENT_MODE=sandbox
#   bash examples/agentic_rl_grpo/run.sh
#
#   # NPU smoke test / no Docker:
#   export SLIME_AGENT_MODE=sglang_loop
#   bash examples/agentic_rl_grpo/run.sh
#
# Required env vars (sandbox mode):
#   ADAPTER_PUBLIC_HOST, SLIME_E2B_SANDBOX_IMAGE,
#   SLIME_AGENT_NODE_TARBALL, SLIME_AGENT_HERMES_TARBALL
#
# Required env vars (sglang_loop mode):
#   (none — uses local subprocess)
# =============================================================================

set -euo pipefail

# ---- 4B model config ----
source scripts/models/qwen3-4B.sh

# ---- Agent mode ----
AGENT_MODE="${SLIME_AGENT_MODE:-sandbox}"

# ---- Multi-dimensional reward weights (general tasks only) ----
REWARD_WEIGHTS='{"correctness":0.51,"format":0.15,"tool_params":0.10,"retry":0.05,"planning":0.075,"hallucination":0.075,"tool_count":0.05}'

# ---- Per-turn limits (sglang_loop mode) ----
MAX_TURNS="${AGENT_MAX_TURNS:-10}"

# ---- Training ----
# NOTE: colocate 模式不支持 train_async.py，必须用 train.py（同步）
python train.py \
    --advantage-estimator grpo \
    --loss-type policy_loss \
    \
    --custom-generate-function-path examples.agentic_rl_grpo.generate.agentic_grpo_generate \
    --custom-rm-path examples.agentic_rl_grpo.reward.agentic_grpo_reward \
    \
    --n-samples-per-prompt 16 \
    --rollout-batch-size 1 \
    --rollout-max-context-len 40960 \
    --rollout-max-response-len 8192 \
    --rollout-temperature 1.0 \
    --rollout-top-p 1.0 \
    \
    --kl-coef 0.001 \
    --kl-loss-type k3 \
    --normalize-advantages \
    \
    --num-rollout 200 \
    --global-batch-size 8 \
    --num-steps-per-rollout 2 \
    --update-weights-interval 1 \
    \
    --colocate \
    --actor-num-gpus-per-node 2 \
    --num-gpus-per-node 2 \
    --rollout-num-gpus 2 \
    --rollout-num-gpus-per-engine 2 \
    \
    --release-train \
    --update-weight-transport disk \
    --sglang-mem-fraction-static 0.7 \
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
