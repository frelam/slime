#!/bin/bash
# =============================================================================
# Simple GRPO Training Launcher for Agentic RL (Easy Datasets)
# =============================================================================
#
# 与 run.sh 不同，此脚本使用**简单数据集**进行 agent RL GRPO 训练。
# 不需要 Docker/E2B，使用本地 sglang_loop 模式。
#
# 数据集 (全部是 outcome reward，无需 multi-dim RM):
#   - simple_shell:  Shell 命令任务 (terminal-bench 风格，简单级别)
#   - simple_math:   数学问题 (GSM8K 风格，Python 工具求解)
#   - simple_code:   编程任务 (测试驱动的代码验证)
#   - alfworld:       文本家务环境 (模拟文件系统操作)
#   - terminal_bench: 既有 terminal-bench 任务
#
# Reward 策略:
#   - simple_shell/terminal_bench: check_command exit code → 0 or 1
#   - simple_math:   answer matching → 0 or 1
#   - simple_code:   test pass rate → 0 or 1
#   - alfworld:      task success check → 0 or 1
#   - 开放性问题:     LLM judge → [0, 1] (可选)
#
# 运行模式: sglang_loop (本地 subprocess，无需 Docker)
#
# Architecture (colocate + release-train, 与 run.sh 相同):
#   Phase 1: [Megatron] train (~24GB/GPU) --release→ free
#   Phase 2: [SGLang TP=2] rollout (~18GB/GPU KV pool) --offload→ free
#   Phase 3: repeat
#
# Usage:
#   # 生成简单数据集:
#   python examples/agentic_rl_grpo/download_simple_data.py -o ./data/simple
#
#   # 训练:
#   bash examples/agentic_rl_grpo/run_simple.sh
#
#   # 自定义数据路径:
#   bash examples/agentic_rl_grpo/run_simple.sh \
#       --prompt-data ./data/simple/mixed_simple_rl.jsonl
#
#   # Smoke test (极少 rollout):
#   bash examples/agentic_rl_grpo/run_simple.sh \
#       --num-rollout 5 --n-samples-per-prompt 4
# =============================================================================

set -euo pipefail

# ---- 4B model config ----
source scripts/models/qwen3-4B.sh

# ---- sglang_loop mode (无需 Docker/E2B) ----
export SLIME_AGENT_MODE="${SLIME_AGENT_MODE:-sglang_loop}"

# ---- Per-turn limits ----
export AGENT_MAX_TURNS="${AGENT_MAX_TURNS:-10}"

# ---- Simple outcome reward (无 multi-dim RM) ----
# 不需要复杂的 reward weights，只使用纯 outcome reward

# ---- Optimizer (Muon + Adam chained, 与 run.sh 相同) ----
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

# ---- DAPO-style Dynamic Sampling (与 run.sh 相同) ----
DAPO_ARGS=(
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --over-sampling-batch-size 2
    --use-dynamic-batch-size
    --max-tokens-per-gpu 9216
)

# ---- Default data path ----
DATA_DIR="${SIMPLE_DATA_DIR:-./data/simple}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/mixed_simple_rl.jsonl}"

# ---- Generate data if not exists ----
if [ ! -f "${PROMPT_DATA}" ]; then
    echo "[run_simple] Data not found at ${PROMPT_DATA}, generating..."
    python examples/agentic_rl_grpo/download_simple_data.py \
        -o "${DATA_DIR}" --max-samples 64
fi

echo "============================================"
echo "[run_simple] Simple Agentic RL GRPO Training"
echo "============================================"
echo "  Agent mode:    ${SLIME_AGENT_MODE}"
echo "  Max turns:     ${AGENT_MAX_TURNS}"
echo "  Data:          ${PROMPT_DATA}"
echo "  Model:         ${MODEL_ARGS[0]:-Qwen3-4B}"
echo "============================================"

# ---- Training ----
# NOTE: colocate 模式不支持 train_async.py，必须用 train.py（同步）
python train.py \
    --advantage-estimator grpo \
    --loss-type policy_loss \
    \
    --custom-generate-function-path examples.agentic_rl_grpo.simple_generate.simple_grpo_generate \
    --custom-rm-path examples.agentic_rl_grpo.simple_reward.simple_outcome_reward \
    \
    --n-samples-per-prompt 16 \
    --rollout-batch-size 4 \
    --rollout-max-context-len 40960 \
    --rollout-max-response-len 8192 \
    --rollout-temperature 1.0 \
    --rollout-top-p 1.0 \
    \
    --kl-coef 0.001 \
    --kl-loss-type k3 \
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
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --apply-chat-template \
    --rollout-shuffle \
    \
    "${OPTIMIZER_ARGS[@]}" \
    "${DAPO_ARGS[@]}" \
    "$@"
