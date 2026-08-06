import logging

import torch

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
        Qwen3VLMoETextRotaryEmbedding,
        Qwen3VLTextRotaryEmbedding,
    )

    def patch_rotary_embedding(cls):
        _original_forward = cls.forward

        def _patched_forward(self, *args, packed_seq_params=None, **kwargs):
            return _original_forward(self, *args, **kwargs)

        cls.forward = _patched_forward

    patch_rotary_embedding(Qwen3VLTextRotaryEmbedding)
    patch_rotary_embedding(Qwen3VLMoETextRotaryEmbedding)
except ImportError:
    pass

# Patch DotProductAttention to reject packed_seq_params gracefully
# (local transformer impl doesn't support THD format; this is only
#  hit when the training data pipeline creates packed batches)
try:
    from megatron.core.transformer.dot_product_attention import DotProductAttention

    _original_dpa_forward = DotProductAttention.forward

    def _patched_dpa_forward(self, query, key, value, attention_mask, attn_mask_type=None, attention_bias=None, packed_seq_params=None):
        # DotProductAttention doesn't support packed THD format.
        # If packed_seq_params are provided, unpack sequences and
        # process them individually.
        if packed_seq_params is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            outputs = []
            q_split = query.split(seqlens)
            k_split = key.split(seqlens)
            v_split = value.split(seqlens)
            for q, k, v in zip(q_split, k_split, v_split, strict=False):
                # DotProductAttention expects [sq, b, np, hn]; THD gives
                # [sq, np, hn] per sequence, so the batch dim goes at position 1.
                out = _original_dpa_forward(self, q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
                                            attention_mask, attn_mask_type, attention_bias, None)
                outputs.append(out.squeeze(1))
            return torch.cat(outputs, dim=0)
        return _original_dpa_forward(self, query, key, value, attention_mask,
                                      attn_mask_type, attention_bias, None)

    DotProductAttention.forward = _patched_dpa_forward
    logging.getLogger(__name__).info("Patched DotProductAttention to handle packed_seq_params")
except (ImportError, AttributeError) as e:
    logging.getLogger(__name__).warning("Could not patch DotProductAttention: %s", e)

# Patch torch.nn.RMSNorm to compute in fp32. On fp16 inputs the stock forward
# computes mean(x^2) in half precision, which overflows to inf for rows whose
# L2 norm exceeds ~sqrt(65504 * H) (e.g. ~12948 for H=2560). Overflowing rows
# get rstd=0 -> the whole row is zeroed -> near-uniform logits -> entropy ~
# ln(vocab), huge logprob diffs, ppo_kl spikes and NaN gradients. Long
# tool-RL prompts produce such rows (observed L2 norms up to ~16760 at the
# final-block input). fp32 compute is exact for every representable fp16 row
# and costs negligible time. Covers input_layernorm / pre_mlp_layernorm /
# qk_layernorm / final_layernorm (local impl builds all of these as
# torch.nn.RMSNorm via WrappedTorchNorm). Module structure is unchanged so
# checkpoints stay compatible.
try:
    _orig_rmsnorm_forward = torch.nn.RMSNorm.forward

    def _rmsnorm_fp32_forward(self, input: torch.Tensor) -> torch.Tensor:
        return _orig_rmsnorm_forward(self, input.float()).to(input.dtype)

    torch.nn.RMSNorm.forward = _rmsnorm_fp32_forward
    logging.getLogger(__name__).info("Patched torch.nn.RMSNorm to compute in fp32 (overflow-safe)")
except Exception as e:
    logging.getLogger(__name__).warning("Could not patch torch.nn.RMSNorm: %s", e)

logging.getLogger("megatron").setLevel(logging.WARNING)

from . import megatron_patch  # noqa: F401, E402
