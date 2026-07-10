#!/bin/bash
# Agentic RL + Online DPO training across multiple benchmarks.
#
# Trains on a mixed dataset (all benchmarks combined) or one at a time.
# Usage:
#   bash run-all.sh                   # train on all benchmarks combined
#   bash run-all.sh swe_gym_lite      # train on SWE-Gym-Lite only
#   bash run-all.sh tau_bench         # train on τ-bench only
#   bash run-all.sh terminal_bench    # train on Terminal-Bench only

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd "$SCRIPT_DIR/../../.." &>/dev/null && pwd)"

source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

MODEL_CKPT="/root/models/Qwen3-4B-Instruct-2507"
SAVE_DIR="/tmp/slime_agentic_rl"

BENCHMARK="${1:-all}"

case "${BENCHMARK}" in
  swe_gym_lite|swe)
    DATA_PATH="/root/datasets/swe_gym_lite/train.jsonl"
    ;;
  tau_bench|tau)
    DATA_PATH="/root/datasets/tau_bench/train.jsonl"
    ;;
  terminal_bench|terminal)
    DATA_PATH="/root/datasets/terminal_bench/train.jsonl"
    ;;
  cli_gym|cli)
    DATA_PATH="/root/datasets/cli_gym/train.jsonl"
    ;;
  api_bank|api)
    DATA_PATH="/root/datasets/api_bank/train.jsonl"
    ;;
  r2e_gym|r2e)
    DATA_PATH="/root/datasets/r2e_gym/train.jsonl"
    ;;
  agent_bench|agent)
    DATA_PATH="/root/datasets/agent_bench/train.jsonl"
    ;;
  all|mixed)
    # Concatenate all datasets (one JSONL per line)
    DATA_PATH="/root/datasets/mixed_agentic_rl/train.jsonl"
    ;;
  *)
    echo "Unknown benchmark: ${BENCHMARK}"
    echo "Usage: $0 [swe_gym_lite|tau_bench|terminal_bench|cli_gym|api_bank|r2e_gym|agent_bench|all]"
    exit 1
    ;;
esac

python "${SLIME_DIR}/train.py" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_CKPT}" \
  --ref-load "${MODEL_CKPT}" \
  --save "${SAVE_DIR}" \
  --save-interval 20 \
  \
  --loss-type custom_loss \
  --custom-loss-function-path examples.agentic_rl.dpo_loss.dpo_loss \
  --dpo-beta 0.1 \
  --compute-advantages-and-returns \
  \
  --custom-generate-function-path examples.agentic_rl.generate.agentic_generate \
  --custom-rm-path examples.agentic_rl.reward.agentic_reward \
  --prompt-data "${DATA_PATH}" \
  --input-key prompt \
  --label-key label \
  --metadata-key metadata \
  --apply-chat-template \
  --rollout-shuffle \
  \
  --num-rollout 200 \
  --rollout-batch-size 4 \
  --n-samples-per-prompt 2 \
  --global-batch-size 8 \
  --num-steps-per-rollout 1 \
  --rollout-max-context-len 32768 \
  --rollout-max-response-len 8192 \
  --rollout-temperature 1.0 \
  \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  \
  "$@"
