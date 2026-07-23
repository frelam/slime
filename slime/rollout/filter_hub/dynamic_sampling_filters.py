import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    # Flatten nested list if generate function returns list[list[Sample]]
    flat = []
    for s in samples:
        if isinstance(s, list):
            flat.extend(s)
        else:
            flat.append(s)
    rewards = [s.get_reward_value(args) for s in flat]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
