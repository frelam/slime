# Tool Call Loss Masking

将错误 tool call token 从 loss 中 mask，避免强化错误的函数名/参数名/类型。

## 使用

```bash
# 基础：mask 所有错误 tool call（不论 advantage 符号）
bash examples/tool_rl/run.sh --mask-failed-tool-calls

# Advantage-conditioned：仅 advantage>0 时 mask，advantage≤0 保留以 unlearn
bash examples/tool_rl/run.sh \
    --mask-failed-tool-calls \
    --mask-failed-tool-calls-adv-conditioned \
    --custom-tis-function-path examples.tool_rl.tis.tool_rl_tis_function
```

## 机制

| loss_mask 值 | 含义 |
|:---:|:---|
| 2 | 正常 token（reasoning、正确 tool call） |
| 1 | 错误 tool call token |
| 0 | TIS 将 adv>0 样本的 `1` 改为 `0` |

**参数：** `--mask-failed-tool-calls`（默认关）、`--mask-failed-tool-calls-adv-conditioned`（需配合 `--custom-tis-function-path examples.tool_rl.tis.tool_rl_tis_function`）。
