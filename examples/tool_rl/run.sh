#!/bin/bash
# =============================================================================
# Tool RL GRPO Training Launcher
# =============================================================================
#
# Uses 4-dim reward (RM + Verifier) for tool-use function calling training.
# Designed for datasets: APIGen, ToolACE, Hammer, BFCL.
#
# Reward dimensions:
#   Dim 1 (0.40): Planning & reasoning quality  — RM scored
#   Dim 2 (0.20): Format compliance              — Verifier (rule-based)
#   Dim 3 (0.20): Tool call format correctness   — Verifier (rule-based)
#   Dim 4 (0.20): Hallucination detection        — RM scored
#
# Architecture (colocate + release-train):
#   Phase 1: [Megatron] train → release → free
#   Phase 2: [SGLang TP=2] rollout → offload → free
#   Phase 3: repeat
#
# Usage:
#   # 1. Download data first:
#   python examples/agentic_rl_grpo/download_tool_data.py -o ./data/tool_rl
#
#   # 2. Train:
#   bash examples/agentic_rl_grpo/run_tool_rl.sh
#
#   # 3. Custom data path:
#   bash examples/agentic_rl_grpo/run_tool_rl.sh \
#       --prompt-data ./data/tool_rl/mixed_tool_rl.jsonl
#
#   # 4. Smoke test (few rollouts):
#   bash examples/agentic_rl_grpo/run_tool_rl.sh \
#       --num-rollout 5 --n-samples-per-prompt 4
#
#   # 5. Custom reward weights:
#   bash examples/agentic_rl_grpo/run_tool_rl.sh \
#       --reward-weights '{"planning":0.5,"format":0.15,"tool_call":0.15,"hallucination":0.2}'
#
#   # 6. Use specific RM (DeepSeek API):
#   RM_MODEL_TYPE=deepseek RM_API_KEY=sk-xxx \
#   bash examples/agentic_rl_grpo/run_tool_rl.sh
# =============================================================================

set -euo pipefail

# ---- Model config (default: Qwen3-4B) ----
MODEL_CONFIG="${MODEL_CONFIG:-scripts/models/qwen3-4B.sh}"
if [ -f "${MODEL_CONFIG}" ]; then
    source "${MODEL_CONFIG}"
else
    echo "[run_tool_rl] WARNING: Model config not found: ${MODEL_CONFIG}"
    echo "[run_tool_rl] Set MODEL_CONFIG to a valid model config script."
    echo "[run_tool_rl] Available: $(ls scripts/models/*.sh 2>/dev/null | tr '\n' ' ')"
    exit 1
fi

# ---- Agent mode ----
export SLIME_AGENT_MODE="${SLIME_AGENT_MODE:-sglang_loop}"

# ---- Per-turn limits ----
export AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-5}"

# ---- Mock execution mode: "generic" (default) or "match" ----
export TOOL_RL_MOCK_MODE="${TOOL_RL_MOCK_MODE:-generic}"

# ---- RM configuration ----
# RM model type: sglang (local RM on SGLang) or deepseek (external API)
RM_MODEL_TYPE="${RM_MODEL_TYPE:-sglang}"
# RM endpoint (leave empty to use SGLang router as RM)
RM_MODEL_ENDPOINT="${RM_MODEL_ENDPOINT:-}"
# RM API key is read from RM_API_KEY env var directly by tool_rl_reward.py
# (NOT passed as CLI arg — avoids exposure in /proc and ps output)

# ---- KL Loss (regularization toward reference model) ----
# Prevents the policy from diverging too far from the reference during GRPO.
# k3 = forward KL with low-variance estimator (recommended for GRPO).
KL_COEF="${KL_COEF:-0.001}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-k3}"

# ---- Optimizer (Muon + Adam, chained) ----
OPTIMIZER_ARGS=(
    --optimizer muon
    --lr 3e-4
    --lr-decay-style constant
    --weight-decay 0.01
    --muon-momentum 0.95
    --muon-use-nesterov
    --muon-scale-mode spectral
    --muon-num-ns-steps 5
)

# ---- Dynamic Sampling (std-based filtering) ----
# Keeps only prompt groups where the rollout rewards have non-zero std.
# This ensures every training batch has meaningful advantage signals.
# Tuning knobs:
#   DYNAMIC_SAMPLING_FILTER — filter function path
#   OVER_SAMPLING_BATCH_SIZE — oversample factor (2 = generate 2x, keep best)
DYNAMIC_SAMPLING="${DYNAMIC_SAMPLING:-true}"
DYNAMIC_SAMPLING_FILTER="${DYNAMIC_SAMPLING_FILTER:-slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-2}"
USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-true}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"

if [ "${DYNAMIC_SAMPLING}" = "true" ]; then
    DAPO_ARGS=(
        --dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER}"
        --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
        --use-dynamic-batch-size
        --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    )
else
    DAPO_ARGS=()
fi

# ---- Default data path ----
DATA_DIR="${TOOL_RL_DATA_DIR:-./data/tool_rl}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/mixed_tool_rl.jsonl}"

# ---- Generate data if not exists ----
if [ ! -f "${PROMPT_DATA}" ]; then
    echo "============================================"
    echo "[run_tool_rl] Data not found at ${PROMPT_DATA}"
    echo "[run_tool_rl] Downloading tool-use datasets..."
    echo "============================================"
    python examples/tool_rl/data/download_data.py \
        -o "${DATA_DIR}" --max-samples 5000

    if [ ! -f "${PROMPT_DATA}" ]; then
        echo "[run_tool_rl] ERROR: Failed to download data."
        echo "[run_tool_rl] Make sure you have:"
        echo "  1. pip install datasets huggingface_hub"
        echo "  2. huggingface-cli login"
        echo "  3. Accepted APIGen license at https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k"
        exit 1
    fi
fi

echo "============================================"
echo "[run_tool_rl] Tool RL GRPO Training"
echo "============================================"
echo "  Agent mode:       ${SLIME_AGENT_MODE}"
echo "  Max turns:        ${AGENT_MAX_TURNS}"
echo "  Mock mode:        ${TOOL_RL_MOCK_MODE}"
echo "  Data:             ${PROMPT_DATA}"
echo "  RM type:          ${RM_MODEL_TYPE}"
echo "  Reward:           4-dim (planning + format + tool_call + hallucination)"
echo "  KL loss:          coef=${KL_COEF}, type=${KL_LOSS_TYPE}"
echo "  Dynamic sampling: ${DYNAMIC_SAMPLING} (filter=${DYNAMIC_SAMPLING_FILTER##*.})"
echo "============================================"

# ---- Build RM args ----
RM_ARGS=()
if [ -n "${RM_MODEL_TYPE:-}" ]; then
    RM_ARGS+=(--rm-model-type "${RM_MODEL_TYPE}")
fi
if [ -n "${RM_MODEL_ENDPOINT:-}" ]; then
    RM_ARGS+=(--rm-model-endpoint "${RM_MODEL_ENDPOINT}")
fi
# NOTE: RM_API_KEY is NOT passed via CLI. The Python code reads it
# directly from os.environ["RM_API_KEY"] to avoid /proc exposure.

# ---- Training ----
# NOTE: colocate mode requires train.py (sync), not train_async.py
python train.py \
    --advantage-estimator grpo \
    --loss-type policy_loss \
    \
    --custom-generate-function-path examples.tool_rl.generate.tool_rl_grpo_generate \
    --custom-rm-path examples.tool_rl.reward.reward.tool_rl_reward \
    \
    --n-samples-per-prompt 16 \
    --rollout-batch-size 4 \
    --rollout-max-context-len 40960 \
    --rollout-max-response-len 8192 \
    --rollout-temperature 1.0 \
    --rollout-top-p 1.0 \
    \
    --kl-coef "${KL_COEF}" \
    --kl-loss-type "${KL_LOSS_TYPE}" \
    --entropy-coef 0.001 \
    --normalize-advantages \
    \
    --num-rollout 500 \
    --global-batch-size 1 \
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
    --prompt-data "${PROMPT_DATA}" \
    --input-key messages \
    --label-key label \
    --metadata-key metadata \
    --tool-key tools \
    --apply-chat-template \
    --hf-checkpoint "${HF_CHECKPOINT:-/home/charles/workspace/qwen3-4b-gdpo-step300}" \
    --rollout-shuffle \
    \
    "${OPTIMIZER_ARGS[@]}" \
    "${DAPO_ARGS[@]}" \
    "${RM_ARGS[@]}" \
    "$@"
