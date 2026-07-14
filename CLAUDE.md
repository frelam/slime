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

```
slime/
├── ray/                    # Ray-based orchestration layer
│   ├── rollout.py          # RolloutManager (Ray actor) — manages SGLang engines, weight sync, data gathering
│   ├── placement_group.py  # GPU provisioning: creates Ray placement groups for train/rollout actors
│   ├── actor_group.py      # RayTrainGroup — wraps Megatron training on a Ray actor group
│   ├── train_actor.py      # TrainActor (Ray actor) — single training worker
│   └── rollout_validation.py  # Validates server-group GPU indices
│
├── rollout/                # Rollout logic & data flow
│   ├── sglang_rollout.py   # Core SGLang rollout — generate_rollout(), reward computation, RM integration
│   ├── sglang_streaming_rollout.py  # Streaming partial rollout (for long generations)
│   ├── fully_async_rollout.py       # Rollout + train fully async (no blocking)
│   ├── data_source.py      # DataSource base class + RolloutDataSource (prompt management, sampling)
│   ├── base_types.py       # RolloutFnTrainOutput, RolloutFnEvalOutput
│   ├── filter_hub/         # Dynamic sampling filters (e.g., top-p, temperature)
│   └── rm_hub/             # Reward model integrations: math, GPQA, F1, DeepScaler, IFBench
│
├── agent/                  # Agentic RL / coding-agent harnesses
│   ├── trajectory.py       # TrajectoryManager — multi-turn conversation tree → training samples
│   ├── sandbox.py          # Sandbox execution for coding agent tasks
│   ├── adapters/           # API adapters: OpenAI, Anthropic
│   ├── harness/            # Coding agent harnesses: ClaudeCode, Codex (common/base)
│   └── parsing.py          # Agent output parsing utilities
│
├── backends/               # Training & inference backend integrations
│   ├── megatron_utils/     # Megatron integration: model provider, actor, loss, checkpoint, CP, update_weight/
│   │   ├── model_provider.py   # Model definition (GPTModel wrapper)
│   │   ├── actor.py             # Training actor logic (forward/backward, PPO loss)
│   │   ├── loss.py              # Loss functions (GRPO, PPO, KLOffline, etc.)
│   │   ├── checkpoint.py        # Checkpoint save/load
│   │   ├── update_weight/       # Weight sync to SGLang (tensor, disk, disk-delta, distributed)
│   │   ├── megatron_to_hf/     # Per-model Megatron→HF checkpoint converters
│   │   └── hf_checkpoint_saver.py
│   └── sglang_utils/       # SGLang integration: config parsing, engine control
│
├── utils/                  # Shared utilities (90k+ lines in arguments.py alone)
│   ├── arguments.py        # All CLI argument definitions (600+ args)
│   ├── ppo_utils.py        # PPO math: advantage estimation, loss computation, KL, GAE
│   ├── types.py            # Core data types: Sample, MultimodalTypes
│   ├── data.py             # Dataset loading (JSONL, Parquet, JSON), tokenization utils
│   ├── dp_schedule.py      # Dynamic packing schedule for variable-length sequences
│   ├── seqlen_balancing.py # Sequence length balancing for efficient packing
│   ├── mask_utils.py       # Loss mask generation for PPO/GRPO
│   ├── trace_utils.py      # Trace/tracing integration for observability
│   ├── tensorboard_utils.py / wandb_utils.py # Experiment tracking
│   └── external_utils/     # External CLI helpers (typer-based)
│
slime_plugins/              # Model-specific bridges and extensions
├── megatron_bridge/        # Megatron bridge for GLM-4V MoE
├── mbridge/                # Model bridge configs (GLM, Qwen, DeepSeek, etc.)
├── models/                 # Model implementations (GLM5, Gemma4, Qwen3, etc.)
└── rollout_buffer/         # Rollout buffer for experience replay
```

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

```
tests/
├── test_<model>_<feature>.py    # GPU integration tests (4+ GPUs)
├── test_qwen2.5_0.5B_short.py   # Minimal smoke test (fastest GPU test)
├── test_release_train.py         # Full workflow test
├── test_loss_cp_invariance.py    # Context parallelism loss correctness
├── test_metric_report.py         # Metric reporting correctness
├── test_sample.py                # Data sampling correctness
├── test_chunked_gae.py           # Chunked GAE advantage estimation
├── test_cispo_loss.py            # CISPO loss
├── test_ppo_logprob_entropy.py   # PPO logprob/entropy CPU test
├── test_ppo_logprob_entropy_gpu.py  # PPO logprob/entropy GPU test
├── test_agent/                   # Agentic RL CPU tests
├── plugin_contracts/             # Plugin API contract tests
├── utils/                        # Utility function tests
└── gemma4/                       # Gemma4-specific tests
```

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
