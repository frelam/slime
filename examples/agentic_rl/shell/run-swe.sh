#!/bin/bash
# Agentic RL + Online DPO training on SWE-Gym-Lite
# Prerequisites: Qwen3-4B (or other) model checkpoint, SWE-Gym-Lite dataset

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd "$SCRIPT_DIR/../../.." &>/dev/null && pwd)"

source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

MODEL_CKPT="/root/models/Qwen3-4B-Instruct-2507"
DATA_PATH="/root/datasets/swe_gym_lite/train.jsonl"
SAVE_DIR="/tmp/slime_agentic_swe"

python "${SLIME_DIR}/train.py" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_CKPT}" \
  --ref-load "${MODEL_CKPT}" \
  --save "${SAVE_DIR}" \
  --save-interval 10 \
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
  --num-rollout 100 \
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
