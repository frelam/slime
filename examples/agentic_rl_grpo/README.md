# Agentic RL — GRPO/PPO Training

使用 GRPO/PPO 对 agentic RL 任务进行强化学习训练。

## 架构

```
train_async.py  (rollout N+1 与 train N 并行)
  │
  ├── rollout_manager.generate.remote(rollout_id)
  │   └── agentic_grpo_generate()  ← 每个 sample 调用一次
  │       │
  │       ├── [数据集适配器] 检测 benchmark 类型
  │       │
  │       ├── [通用任务] terminal_bench, cli_gym, tau_bench, …
  │       │   ├── E2B sandbox + Hermes CLI
  │       │   ├── Hermes → OpenAIAdapter (port 18002) → SGLang
  │       │   │   └── 自动捕获 logprobs
  │       │   ├── adapter.evaluate_task() → 规则化评估
  │       │   └── 多维度 reward (RM + verifier, 7 维)
  │       │
  │       └── [SWE 任务] swe_gym_lite, r2e_gym
  │           ├── E2B sandbox + Claude Code CLI
  │           ├── Claude Code → AnthropicAdapter (port 18001) → SGLang
  │           │   └── 自动捕获 logprobs
  │           └── 测试通过率 reward (暂不启用多维度打分)
  │
  └── actor_model.async_train(rollout_id, rollout_data)
      └── 内建 GRPO loss: group-reward norm → advantage → policy gradient
```

## 任务路由

`generate.py` 根据 `sample.metadata["benchmark"]` 自动选择 harness：

| benchmark | 任务类型 | Harness | Adapter | Port |
|-----------|---------|---------|---------|------|
| `swe_gym_lite` | SWE | Claude Code | Anthropic | 18001 |
| `r2e_gym` | SWE | Claude Code | Anthropic | 18001 |
| 其他所有 | 通用 | **Hermes** | OpenAI | 18002 |

`benchmark` 字段通常由数据集 JSONL 中的 `metadata.benchmark` 提供。
如果缺失，`_auto_detect_benchmark()` 会根据其他 metadata 字段自动推断。

## Reward 规则

### 通用任务（已注册）

**Verifier 维度**（规则化，无需 LLM）：

| 维度 | 权重 | 说明 | 打分规则 |
|------|------|------|---------|
| 4.2 格式合规 | 0.15 | 输出格式是否正确 | 格式正确→1, 否则→0 |
| 4.3 工具参数正确 | 0.10 | 工具调用是否成功 | 无失败→1, 有失败→0 |
| 4.4 重试行为 | 0.05 | 失败后是否重试 | 无失败→1, 失败+重试→0.5, 放弃→0 |
| 4.7 调用次数惩罚 | 0.05 | 过度调用惩罚 | 答错→0, 答对→max(0, 1-count/1000) |

**RM 维度**（调用 Reward Model 打分）：

| 维度 | 权重 | 说明 | 打分规则 |
|------|------|------|---------|
| 4.1 答案正确性 | 0.51 | 是否完成任务 | RM 判断, 0 或 1 |
| 4.5 规划质量 | 0.075 | 思路是否清晰 | RM 判断, 0.0–1.0 |
| 4.6 是否有臆想 | 0.075 | 是否编造信息 | RM 判断, 0 或 1 |

**最终 reward**: `Σ(weight_i × score_i)`, 范围 [0, 1]。

RM 的 system prompt 按任务类型从 `prompts/{task_type}.md` 加载，用户可直接编辑 `.md` 文件调优评测标准，无需修改代码。

### SWE 任务（待注册）

当前 SWE 任务使用原始的测试通过率作为 reward，不启用多维度打分。
SWE 的多维度 reward 规则后续补充。

## 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `ADAPTER_PUBLIC_HOST` | **是** | 宿主机 IP，sandbox 可通过此 IP 回调 adapter |
| `ADAPTER_BIND_HOST` | 否 | adapter 监听地址，默认 `0.0.0.0` |
| `ADAPTER_PORT_ANTHROPIC` | 否 | Anthropic adapter 端口，默认 `18001` |
| `ADAPTER_PORT_OPENAI` | 否 | OpenAI adapter 端口，默认 `18002` |
| `SLIME_E2B_SANDBOX_IMAGE` | **是** | 通用任务的默认 E2B 镜像 |
| `SLIME_AGENT_NODE_TARBALL` | **是** | Node.js 22 安装包路径（.tar.xz） |
| `SLIME_AGENT_HERMES_TARBALL` | **是** | Hermes CLI 安装包路径（.tgz） |
| `SLIME_AGENT_CC_TARBALL` | SWE 必需 | Claude Code CLI 安装包路径（.tgz） |
| `SWE_AGENT_TIME_BUDGET_SEC` | 否 | 每个 agent 任务的时间预算（秒），默认 `1800` |
| `SWE_EVAL_TIMEOUT_SEC` | 否 | 评估超时（秒），默认 `600` |
| `SWE_ROLLOUT_GUARD_SEC` | 否 | 单次 rollout 的 wall-clock 上限（秒），默认 `agent_budget + eval + 180` |
| `SWE_BOOT_CONCURRENCY` | 否 | sandbox 并发启动数，默认 `16` |
| `SWE_BOOT_RETRIES` | 否 | sandbox 启动重试次数，默认 `2` |
| `SWE_CC_PROMPT` | 否 | Claude Code 的 prompt 模板 |
| `SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS` | 否 | fork-merge 阈值 |

## CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--rm-model-type` | RM 后端类型: `sglang` 或 `deepseek` | `sglang` |
| `--rm-model-endpoint` | RM API 端点 URL | SGLang router |
| `--rm-api-key` | 外部 RM 的 API Key（DeepSeek 用） | 无 |
| `--rm-system-prompt-dir` | RM system prompt `.md` 文件目录 | `examples/agentic_rl_grpo/prompts` |
| `--reward-weights` | 维度权重的 JSON 字符串 | 见上表 |

## 快速开始

### 1. 准备数据

数据集为 JSONL 格式，每行一个任务：

```jsonl
{"prompt": "列出 /etc 下所有 .conf 文件", "metadata": {"benchmark": "terminal_bench", "check_command": "ls /etc/*.conf", "expected_exit_code": 0}}
{"prompt": "...", "metadata": {"benchmark": "swe_gym_lite", "instance_id": "django__django-12345", "image": "swebench/sweb.eval.x86_64.django__django-12345:latest", "repo": "django/django", "base_commit": "abc1234"}}
```

### 2. 设置环境变量

```bash
export ADAPTER_PUBLIC_HOST=10.0.0.1          # 你的宿主机 IP
export SLIME_E2B_SANDBOX_IMAGE=your-image    # 默认 E2B 镜像
export SLIME_AGENT_NODE_TARBALL=/path/to/node-v22.tar.xz
export SLIME_AGENT_HERMES_TARBALL=/path/to/hermes.tgz
export SLIME_AGENT_CC_TARBALL=/path/to/claude-code.tgz  # SWE 用
```

### 3. 调整 RM prompt（可选）

编辑 `examples/agentic_rl_grpo/prompts/{task_type}.md` 调整 RM 评测维度，无需改代码。

### 4. 启动训练

```bash
bash examples/agentic_rl_grpo/run.sh \
    --hf-checkpoint /path/to/model \
    --prompt-data /path/to/data.jsonl \
    "${MODEL_ARGS[@]}"
```

## 文件结构

```
examples/agentic_rl_grpo/
├── __init__.py              # Package doc
├── generate.py              # 主生成函数 (--custom-generate-function-path)
├── reward.py                # 多维度 reward 组合器
├── reward_model.py          # RM API 客户端 (Qwen/DeepSeek)
├── verifier.py              # 规则化 verifier 维度 (4.2/4.3/4.4/4.7)
├── traj_analysis.py         # 轨迹解析工具
├── prompts/                 # RM system prompt .md 文件
│   ├── terminal_bench.md
│   ├── cli_gym.md
│   ├── swe_gym_lite.md
│   ├── r2e_gym.md
│   ├── tau_bench.md
│   ├── api_bank.md
│   └── agent_bench.md
└── run.sh                   # 训练启动脚本

slime/agent/harness/
├── hermes.py                # Hermes harness (OpenAI 兼容)
└── __init__.py              # 注册 HermesHarness
```

## 常见问题

### Q: 如何新增任务类型？

1. 在 `examples/agentic_rl_datasets/` 中创建新的 `DatasetAdapter` 子类
2. 在 `examples/agentic_rl_grpo/prompts/` 中添加 `{task_type}.md`
3. 如果任务有专用 harness，添加到 task-routing 映射

### Q: 如何调整 reward 权重？

通过 `--reward-weights` 传入 JSON 字符串（支持部分覆盖）：

```bash
--reward-weights '{"correctness":0.6,"format":0.1}'
```

### Q: RM 调用失败怎么办？

RM 调用失败时返回中性分数（各项 0.5），并 fallback 到任务评估 reward 作为 correctness 维度。不影响训练继续。

### Q: 如何只用一种 harness？

如果要所有任务都用 Hermes，将 `_SWE_TASK_TYPES` 设为空集合即可。反之亦然。
