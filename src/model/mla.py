import math
from typing import Any, Dict, Literal

import torch
import torch.nn as nn

from flash_attn import flash_attn_func

from .fope import FourierPositionEmbedding

PositionEmbeddingType = Literal["rope", "fope", "none"]
PositionEmbeddingSettings = Dict[str, Any]


class NoEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k


# arXiv:2502.07864
class MultiheadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        latent_dim_rank: int,
        bias: bool = False,
        position_embedding: PositionEmbeddingType = "fope",
        position_embedding_settings: PositionEmbeddingSettings = {},
        context_len: int = None,
        dropout: float = 0.0,
        softcap: float = 20.0,
        softmax_scale: float = None,
        # arXiv:2501.19399
        learn_softmax: bool = True,
        softmax_bias: bool = False,
        softmax_scale_init: float = 0.43,
        # arXiv:2505.15548
        longshort: bool = True,
        long_ratio: float = 0.3333333333333333,
        short_range: int = 512,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_kv_heads <= n_heads, "n_kv_heads must be less than or equal to n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert not longshort or (
            long_ratio > 0 and long_ratio < 1
        ), "long_ratio must be between 0 and 1 if longshort is True"
        assert (
            not longshort or (long_ratio * n_heads).is_integer()
        ), "long_ratio * n_heads must be an integer if longshort is True"
        assert (
            not longshort or (long_ratio * n_kv_heads).is_integer()
        ), "long_ratio * n_kv_heads must be an integer if longshort is True"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        self.longshort = longshort
        self.n_long_heads = int(n_heads * long_ratio) if longshort else n_heads
        self.n_short_heads = n_heads - self.n_long_heads
        self.n_long_kv_heads = int(n_kv_heads * long_ratio) if longshort else n_kv_heads
        self.n_short_kv_heads = n_kv_heads - self.n_long_kv_heads
        swh = short_range // 2
        self.short_window = (swh, swh)

        self.d_head = d_model // n_heads
        ratio = n_heads // n_kv_heads

        self.w_q = nn.Linear(d_model, d_model, bias=bias)

        latent_dim_rank = latent_dim_rank * 2
        self.w_kv_compress = nn.Linear(d_model, latent_dim_rank, bias=bias)
        self.w_kv_decompress = nn.Linear(
            latent_dim_rank, 2 * (d_model // ratio), bias=bias
        )

        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        if position_embedding == "fope":
            self.emb = FourierPositionEmbedding(
                d_head=self.d_head,
                context_len=context_len,
                **position_embedding_settings,
            )
        elif position_embedding == "rope":
            pass
            # self.emb = RotaryPositionalEmbedding(

            # )
        else:
            self.emb = NoEmbedding()
        self.dropout = nn.Dropout(dropout)
        self.softcap = softcap

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(self.d_head)

        if learn_softmax:
            self.softmax_scale = nn.Parameter(
                torch.full((self.n_heads,), softmax_scale_init)
            )
            if softmax_bias:
                self.softmax_bias = nn.Parameter(torch.full((self.n_heads,), 0.0))

        self.scale = softmax_scale
        self.learn_softmax = learn_softmax
        self.softmax_has_bias = softmax_bias

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # value is ignored 'cause we don't use it
        B, L_q, _ = query.shape
        _, L_k, _ = kv.shape

        q = self.w_q(query)
        kv_input = kv
        latent_kv = self.w_kv_compress(kv_input)
        k, v = self.w_kv_decompress(latent_kv).chunk(2, dim=-1)

        q = q.view(B, L_q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L_k, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L_k, self.n_kv_heads, self.d_head).transpose(1, 2)

        q, k = self.emb(q=q, k=k)

        if self.learn_softmax:
            dtype = q.dtype
            n_val = torch.tensor(L_q, dtype=dtype, device=q.device)
            log_n = torch.log(n_val)

            scale_per_head = self.softmax_scale.view(1, self.n_heads, 1, 1)
            scale_per_head = scale_per_head * log_n

            if self.softmax_has_bias:
                bias_per_head = self.softmax_bias.view(1, self.n_heads, 1, 1)
                scale_per_head = scale_per_head + bias_per_head

            q = q * scale_per_head

        if not self.longshort:
            output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
                # Attention in gemma2 was trained using a softcap of 20, which improved
                # long range dependencies in attention, so might as well have this in here,
                # given its minimal performance cost.
                softcap=self.softcap,
                # arXiv:2501.19399
                # Scaling softmax with a learnable scale (according to the paper mentioned above)
                # improves performance in long sequences. Long sequences are all we have, so this
                # is generally a good thing to have, right?
                softmax_scale=self.scale,
            )
        else:
            output_local = flash_attn_func(
                q[:, : self.n_short_heads, :, :],
                k[:, : self.n_short_kv_heads, :, :],
                v[:, : self.n_short_kv_heads, :, :],
                dropout_p=self.dropout.p if self.training else 0.0,
                softcap=self.softcap,
                softmax_scale=self.scale,
                # Short attention heads focus on local dependencies
                window_size=self.short_window,
            )

            output_long = flash_attn_func(
                q[:, self.n_short_heads :, :, :],
                k[:, self.n_short_kv_heads :, :, :],
                v[:, self.n_short_kv_heads :, :, :],
                dropout_p=self.dropout.p if self.training else 0.0,
                softcap=self.softcap,
                softmax_scale=self.scale,
            )
            output = torch.cat([output_local, output_long], dim=1)

        output = output.transpose(1, 2).contiguous().view(B, L_q, -1)
        return self.w_o(output)
