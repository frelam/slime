# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Is slime

slime is an LLM post-training framework for **RL scaling**, connecting **Megatron** (training backend) with **SGLang** (rollout/inference backend) via **Ray** (orchestration). It powers the RL training behind GLM-4.5 through GLM-5.2 and supports Qwen, DeepSeek, Llama, Moonlight, Kimi-K2, MiniMax-M2, and other model families.

The core loop: train → sync weights → rollout (generate + reward) → buffer → train.

## Commands

### Run Training (Sync)
```bash
# Single script that launches Ray, training actors, and SGLang rollout
bash scripts/run-qwen2.5-0.5B-gb10-smoke.sh
```

### Run Training (Async / Disaggregated)
```bash
python train_async.py <all args>
```

### Debug: Rollout Only (no training)
```bash
python train.py --num-rollout 1 --num-train-steps-per-rollout 0 ...
```

### Debug: Train Only (replay saved rollout data)
```bash
bash tests/test_qwen2.5_0.5B_debug_rollout_then_train.sh
```

### Run GPU Tests (4 GPUs required)
```bash
# Single test
python tests/ci/gpu_lock_exec.py --target-env-name CUDA_VISIBLE_DEVICES \
  -- cmd python tests/test_qwen2.5_0.5B_short.py

# All GPU tests
ls tests/test_*.py | xargs -I{} python tests/ci/gpu_lock_exec.py \
  --target-env-name CUDA_VISIBLE_DEVICES -- cmd python {}
```

### Run CPU Unit Tests
```bash
pytest tests/utils/ tests/test_agent/ tests/plugin_contracts/ -n auto
```

### Run a Specific Unit Test
```bash
pytest tests/test_agent/test_trajectory_manager_branching.py -v
pytest tests/utils/test_mask_utils.py -v -k "test_name"
```

### Lint & Format
```bash
ruff check slime/ slime_plugins/ tests/ examples/
```

### Model Config Pattern
```bash
# Load model architecture config, then launch:
source scripts/models/qwen2.5-0.5B.sh
python train.py "${MODEL_ARGS[@]}" <ckpt_args> <rollout_args> <grpo_args> <sglang_args> ...
```

## Code Architecture

### Entry Points
- **`train.py`** — Synchronous train loop: generate → train → save (colocated or disaggregated)
- **`train_async.py`** — Async train loop: overlaps rollout generation with training via future pipelining (for disaggregated setups where gen is slow)
- Both import from `slime.ray.placement_group` (`create_placement_groups`, `create_rollout_manager`, `create_training_models`)

### Key Packages

- `slime/ray/` — Ray orchestration: RolloutManager, placement groups, train actor groups
- `slime/rollout/` — Rollout logic: SGLang generation, data sources, RM hub, filter hub
- `slime/agent/` — Agentic RL harnesses: trajectory management, sandbox execution, API adapters
- `slime/backends/` — Backend integrations: Megatron (model, loss, checkpoint, weight sync) + SGLang utils
- `slime/utils/` — Shared utilities: arguments (90k+ lines, 600+ args), PPO math, data loading, packing
- `slime_plugins/` — Model-specific bridges: mbridge, model implementations, rollout buffer

### Data Flow (Sync Train Loop)

```
train.py
  ├── create_placement_groups(args)     # Reserve GPUs
  ├── create_rollout_manager(args, pg)  # Launch SGLang engine(s) in Ray actor(s)
  ├── create_training_models(args, pg)  # Launch Megatron training actor(s)
  ├── actor_model.update_weights()      # Push initial weights to SGLang
  │
  └── for rollout_id in range(num_rollout):
      ├── rollout_manager.generate.remote(rollout_id)  # SGLang inference + reward
      │   └── sglang_rollout.generate_rollout()
      │       ├── DataSource.get_samples()   → prompts
      │       ├── SGLang /generate            → responses
      │       ├── RM hub (math/gpqa/f1/…)    → rewards
      │       └── DataSource.add_samples()   → store in buffer
      ├── actor_model.train(rollout_data)    # Megatron forward/backward
      │   └── backends/megatron_utils/loss.py  # PPO/GRPO loss
      ├── actor_model.update_weights()       # Sync updated weights → SGLang
      └── rollout_manager.eval.remote()      # Optional eval
```

### Launch Configuration

There is **no single config file**. Everything is CLI arguments passed to `train.py` or `train_async.py`:

| Argument Group | Prefix/Pattern | Source |
|---|---|---|
| Megatron args | Direct (e.g., `--num-layers 24`) | `slime/utils/arguments.py` via `parse_args()` |
| SGLang args | `--sglang-` prefix (e.g., `--sglang-mem-fraction-static 0.8`) | SGLang's own parser, injected by `slime/backends/sglang_utils/arguments.py` |
| SGLang Config | `--sglang-config path/to/config.yaml` | Optional YAML for multi-model, PD disaggregation, heterogeneous groups |
| RL algorithm args | `--advantage-estimator grpo/gae/...` | `slime/utils/arguments.py` |
| Custom rollout fn | `--custom-rollout-fn-path module.fn` | Any Python function; loaded via `load_function()` |
| Custom generate fn | `--custom-generate-fn-path module.fn` | Async generate wrapper for agentic workflows |

### Model Support Pattern

Each model family follows a consistent pattern:
1. **Model config script** (`scripts/models/qwen2.5-0.5B.sh`) — architecture hyperparams
2. **Megatron→HF converter** (`slime/backends/megatron_utils/megatron_to_hf/qwen2.py`)
3. **Optional model bridge** (`slime_plugins/mbridge/qwen3_next.py`)
4. **Optional model implementation** (`slime_plugins/models/`)

### Tests Structure

- `tests/test_<model>_<feature>.py` — GPU integration tests (4+ GPUs). Smoke test: `test_qwen2.5_0.5B_short.py`
- `tests/test_agent/` — Agentic RL CPU tests
- `tests/plugin_contracts/` — Plugin API contract tests
- `tests/utils/` — Utility function tests
- `tests/gemma4/` — Gemma4-specific tests

### Key Design Patterns

- **CLI-driven**: All configuration via `argparse` (~600+ args), no config file framework. Model architecture defined in sourced shell scripts.
- **`load_function()`**: Plugin system via dotted-path function references (e.g., `--custom-rollout-fn-path my_module.my_fn`). Used for custom rollout, generate, reward, and filter functions.
- **Reward Models** in `slime/rollout/rm_hub/`: Each RM is a callable registered via `--rm-type` (math, gpqa, f1, deepscaler, ifbench). Math uses DAPO-style verifier; GPQA uses LLM judge.
- **Weight Sync**: Training → inference weight transfer supports tensor-passing, disk, disk-delta, distributed modes. Enable via `--weight-update-mode`.
- **Dynamic Packing**: Sequences packed into fixed `--global-batch-size` units via `dp_schedule.py`, honoring `--max-tokens-per-gpu`.
- **PD Disaggregation**: Separate prefill/decode server groups via `--sglang-config` YAML, enabling different GPU counts and SGLang args per group.
- **Advantage Estimators**: GRPO, GAE (PPO), chunked GAE, RLOO, and more — selected via `--advantage-estimator`.

### Commonly Modified Files

| File | Purpose |
|---|---|
| `slime/utils/arguments.py` | Add/modify CLI arguments (90k file, be careful) |
| `slime/backends/megatron_utils/loss.py` | Add new RL loss functions |
| `slime/utils/ppo_utils.py` | Advantage estimation, KL computation, PPO math |
| `slime/rollout/rm_hub/` | Add new reward model |
| `slime/rollout/data_source.py` | Modify prompt sampling, data loading |
| `slime/rollout/sglang_rollout.py` | Modify rollout generation flow |
| `slime/ray/rollout.py` | Modify rollout manager orchestration |
| `slime/rollout/filter_hub/` | Add dynamic sampling filters |
| `scripts/models/` | Add new model architecture config |
