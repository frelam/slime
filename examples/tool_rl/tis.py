"""Custom TIS (Truncated Importance Sampling) function for tool RL.

Provides advantage-conditioned masking of incorrect tool call tokens.

How it works
------------
The generate function (``tool_rl_grpo_generate``) encodes tool-call correctness
into ``loss_mask`` values:

- ``2`` = normal token (correct tool call, reasoning, or non-tool text)
- ``1`` = incorrect tool call token

This TIS function reads the per-sample advantage sign from ``pg_loss`` and
decides whether to mask incorrect tool call tokens:

- **advantage > 0** (sample is better than group mean):
   Change ``loss_mask=1`` → ``0``, i.e. mask out incorrect tool call tokens
   so the policy does NOT get reinforced for wrong format.

- **advantage <= 0** (sample is worse or equal to group mean):
   Keep incorrect tool call tokens in loss (change ``1`` → ``2``), so the
   policy can unlearn the wrong format.

Usage
-----
.. code-block:: bash

    python train.py \\
        --custom-generate-function-path examples.tool_rl.generate.tool_rl_grpo_generate \\
        --custom-rm-path examples.tool_rl.reward.reward.tool_rl_reward \\
        --custom-tis-function-path examples.tool_rl.tis.tool_rl_tis_function \\
        --mask-failed-tool-calls \\
        --mask-failed-tool-calls-adv-conditioned \\
        ...
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def tool_rl_tis_function(
    args: Any,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    """TIS function with advantage-conditioned tool call token masking.

    Only active when both ``--mask-failed-tool-calls`` and
    ``--mask-failed-tool-calls-adv-conditioned`` are set.

    Args:
        args: Training args (must have ``mask_failed_tool_calls`` and
            ``mask_failed_tool_calls_adv_conditioned`` attributes).
        pg_loss: Per-token policy gradient loss (flat, concatenated).
        train_log_probs: Log probs from training forward pass.
        rollout_log_probs: Log probs from rollout engine.
        loss_masks: Per-sample loss masks.
        response_lengths: Per-sample response lengths (for splitting pg_loss).
        **kwargs: Additional arguments from the TIS dispatch.

    Returns:
        Tuple of ``(pg_loss, modified_response_masks, metrics_dict)``.
    """
    # Early exit: feature not enabled
    if not getattr(args, "mask_failed_tool_calls", False):
        return pg_loss, loss_masks, {}

    if not getattr(args, "mask_failed_tool_calls_adv_conditioned", False):
        return pg_loss, loss_masks, {}

    if response_lengths is None:
        logger.warning(
            "[tool_rl-tis] response_lengths not provided — "
            "cannot determine per-sample advantage sign, skipping"
        )
        return pg_loss, loss_masks, {}

    # Quick check: do any masks contain value 1 (incorrect tool call tokens)?
    has_tagged = any((mask == 1).any() for mask in loss_masks)
    if not has_tagged:
        return pg_loss, loss_masks, {}

    modified_masks: list[torch.Tensor] = []
    offset = 0
    total_masked = 0
    total_kept = 0

    for mask, resp_len in zip(loss_masks, response_lengths, strict=False):
        sample_pg_loss = pg_loss[offset : offset + resp_len]
        offset += resp_len

        if sample_pg_loss.numel() == 0:
            modified_masks.append(mask)
            continue

        # Mean pg_loss sign → advantage sign:
        # pg_loss = -ratio.clip(...) * advantage, ratio > 0
        # So: pg_loss < 0 → advantage > 0 → reinforce
        #     pg_loss >= 0 → advantage <= 0 → unlearn
        mean_pg = sample_pg_loss.mean().item()

        modified = mask.clone()

        # Count incorrect tool call tokens in this sample
        incorrect_mask = modified == 1

        if mean_pg < 0:
            # Advantage > 0: mask incorrect tool call tokens (don't reinforce)
            modified[incorrect_mask] = 0
            total_masked += int(incorrect_mask.sum())
        else:
            # Advantage <= 0: convert to normal (let them unlearn)
            modified[incorrect_mask] = 2
            total_kept += int(incorrect_mask.sum())

        modified_masks.append(modified)

    logger.info(
        "[tool_rl-tis] adv-conditioned mask: masked=%d tokens (adv>0), "
        "kept=%d tokens (adv<=0)",
        total_masked, total_kept,
    )

    # Standard TIS metrics (not modified by our masking)
    rollout_log_probs_cat = torch.cat(rollout_log_probs, dim=0)
    old_log_probs_cat = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs_cat - rollout_log_probs_cat)
    metrics: dict[str, torch.Tensor] = {
        "tis": tis.clone().detach(),
        "tool_rl_masked_tokens": torch.tensor(total_masked, dtype=torch.float32),
        "tool_rl_kept_tokens": torch.tensor(total_kept, dtype=torch.float32),
    }

    return pg_loss, modified_masks, metrics
