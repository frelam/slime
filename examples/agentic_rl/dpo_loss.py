"""Online DPO loss function for agentic RL.

Implements the standard DPO (Direct Preference Optimization) loss:

    L_DPO = -log σ(β * ((logπ_θ(y_w) - logπ_ref(y_w)) - (logπ_θ(y_l) - logπ_ref(y_l))))

where y_w is the chosen (preferred) trajectory and y_l is the rejected one.

This module provides:

1. :func:`dpo_loss` — entry point for ``--custom-loss-function-path``.
2. :func:`dpo_convert_samples_to_train_data` — optional custom data converter
   (for ``--custom-convert-samples-to-train-data-path``) that groups pairs and
   stores ``pair_indices`` in the batch.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn.functional as F

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ===========================================================================
# DPO data pipeline helper
# ===========================================================================

def dpo_tag_samples(samples: list[Sample]) -> list[Sample]:
    """Tag samples with ``pair_index`` and ``is_chosen`` based on rewards.

    Groups samples by their ``group_index``, then within each group assigns
    the higher-reward sample as "chosen".

    This is called inside the generate function; for the data pipeline, we
    propagate these tags through ``train_metadata``.
    """
    # Group by group_index
    groups: dict[int | None, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.group_index].append(s)

    tagged = []
    for gidx, group in groups.items():
        # Sort by reward descending
        group.sort(key=lambda s: s.reward if s.reward is not None else -1.0, reverse=True)
        for i, s in enumerate(group):
            if s.train_metadata is None:
                s.train_metadata = {}
            s.train_metadata["is_chosen"] = (i == 0)
            s.train_metadata["pair_index"] = gidx if gidx is not None else hash(str(s.prompt))
            tagged.append(s)
    return tagged


# ===========================================================================
# Custom samples → train-data converter
# ===========================================================================

def dpo_convert_samples_to_train_data(
    args: Any,
    samples: list[Sample],
) -> dict[str, Any] | None:
    """Custom converter for ``--custom-convert-samples-to-train-data-path``.

    Extends the default conversion by adding ``pair_indices`` and ``is_chosen``
    fields to the training batch, so the DPO loss function can identify pairs.

    Call this as a wrapper around the default conversion, then add the
    extra keys.
    """
    # Let the default converter run first
    from slime.ray.rollout import _convert_samples_to_train_data as default_convert

    train_data = default_convert(args, samples)
    if train_data is None:
        return None

    # Add pair_indices and is_chosen from train_metadata
    train_data["pair_indices"] = []
    train_data["is_chosen"] = []

    for s in samples:
        md = s.train_metadata or {}
        pair_idx = md.get("pair_index", -1)
        is_chosen = md.get("is_chosen", True)
        train_data["pair_indices"].append(pair_idx)
        train_data["is_chosen"].append(is_chosen)

    logger.debug(
        "DPO data: %d samples, %d unique pairs",
        len(samples),
        len(set(train_data["pair_indices"])),
    )
    return train_data


# ===========================================================================
# DPO Loss Function
# ===========================================================================

def dpo_loss(
    args: Any,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Online DPO loss function.

    Designed for ``--custom-loss-function-path`` with ``--loss-type custom_loss``.

    Args:
        args: Slime training args (must have ``dpo_beta``).
        batch: Batch dict with keys:
            - ``tokens``, ``loss_masks``, ``log_probs`` (rollout old log-probs)
            - ``ref_log_probs`` (reference model log-probs per token)
            - ``pair_indices`` (list[int], one per sample)
            - ``is_chosen`` (list[bool], one per sample)
        logits: Model output logits ``[1, T, V]`` (or ``[1, T, 1]`` for value).
        sum_of_sample_mean: Callable to reduce per-sample losses to scalar.

    Returns:
        ``(loss, metrics_dict)``.
    """
    # 1. Compute current policy log-probs
    # Reuse the existing helper from slime's loss module
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    log_probs, entropy = get_log_probs_and_entropy(logits, batch)

    beta = getattr(args, "dpo_beta", 0.1)

    pair_indices = batch.get("pair_indices", [])
    is_chosen = batch.get("is_chosen", [])

    # Handle the case where pair_indices/is_chosen are lists or tensors
    if torch.is_tensor(pair_indices):
        pair_indices = pair_indices.tolist()
    if torch.is_tensor(is_chosen):
        is_chosen = is_chosen.tolist()

    # 2. Compute accuracy metric (current policy: chosen > rejected)
    chosen_acc_metric = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    pair_count = 0
    total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    if pair_indices and is_chosen and len(pair_indices) == len(is_chosen) == log_probs.shape[0]:
        # Group by pair_index
        pairs: dict[int, dict[str, int | float]] = {}
        for i in range(len(pair_indices)):
            pid = int(pair_indices[i])
            if pid not in pairs:
                pairs[pid] = {}
            key = "chosen" if is_chosen[i] else "rejected"
            pairs[pid][key] = i  # type: ignore[typeddict-item]

        losses = []
        for pid, pair in pairs.items():
            c_idx = pair.get("chosen")
            r_idx = pair.get("rejected")
            if c_idx is None or r_idx is None:
                # Incomplete pair in this micro-batch — skip (will be handled
                # in another micro-batch or is an eval sample)
                continue
            c_idx = int(c_idx)
            r_idx = int(r_idx)

            # Sum log-probs over response tokens only
            # loss_masks shape: [batch_size, seq_len]
            c_mask = batch["loss_masks"][c_idx]  # type: ignore[index]
            r_mask = batch["loss_masks"][r_idx]  # type: ignore[index]

            if c_mask.sum() < 1 or r_mask.sum() < 1:
                logger.debug("Skipping pair %s: empty loss mask (chosen=%s, rejected=%s)", pid, c_mask.sum(), r_mask.sum())
                continue

            # Current policy log-probs for response tokens
            c_logprob = (log_probs[c_idx] * c_mask).sum()
            r_logprob = (log_probs[r_idx] * r_mask).sum()

            # Reference log-probs (already computed by train_actor)
            c_ref_logprob = (batch["ref_log_probs"][c_idx] * c_mask).sum()  # type: ignore[index]
            r_ref_logprob = (batch["ref_log_probs"][r_idx] * r_mask).sum()  # type: ignore[index]

            # DPO log-ratio: (π_θ(y_w) - π_ref(y_w)) - (π_θ(y_l) - π_ref(y_l))
            pi_log_ratio = (c_logprob - c_ref_logprob) - (r_logprob - r_ref_logprob)

            # DPO loss = -log σ(β · pi_log_ratio)
            per_pair_loss = -F.logsigmoid(beta * pi_log_ratio)
            losses.append(per_pair_loss)

            # Accuracy: check if π_θ implicitly prefers the chosen trajectory
            implicit_reward_c = c_logprob - c_ref_logprob
            implicit_reward_r = r_logprob - r_ref_logprob
            if implicit_reward_c > implicit_reward_r:
                chosen_acc_metric += 1.0
            pair_count += 1

        if losses:
            total_loss = torch.stack(losses).sum()
    else:
        # Fallback: no pairing info — apply a simple SFT-like loss instead
        logger.debug("No DPO pair info in batch; falling back to negative log-likelihood")
        total_loss = -(log_probs * batch["loss_masks"]).sum()  # type: ignore[index]
        pair_count = batch["loss_masks"].shape[0]  # type: ignore[index]

    # Use sum_of_sample_mean to reduce to scalar (handles CP, DP correctly)
    loss = sum_of_sample_mean(total_loss.unsqueeze(0))

    # Metrics
    metrics = {
        "dpo/loss": loss.detach(),
        "dpo/beta": torch.tensor(beta, device=logits.device, dtype=logits.dtype),
    }
    if pair_count > 0:
        metrics["dpo/accuracy"] = (chosen_acc_metric / max(pair_count, 1)).detach()

    return loss, metrics
