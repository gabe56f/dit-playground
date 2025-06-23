import math
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn.modules.normalization import RMSNorm

from .mla import MultiheadLatentAttention


class SwiGLUMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        use_smooth_swiglu: bool = False,
        act: Literal["swish", "gelu", "gelu-approx"] = "swish",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, 2 * hidden_features, bias=False)
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "gelu-approx":
            self.act = nn.GELU(approximate="tanh")
        else:
            self.act = nn.SiLU()

        # arXiv:2409.12517
        if use_smooth_swiglu:
            self.gate_norm = RMSNorm(
                hidden_features, eps=1e-8, elementwise_affine=False
            )
        else:
            self.gate_norm = nn.Identity()

        self.w2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project and split into value (x) and gate (g)
        x, g = self.w1(x).chunk(2, dim=-1)

        g = self.gate_norm(g)
        x = x * self.act(g)

        # Final projection
        x = self.w2(x)
        return x


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        use_smooth_swiglu: bool = False,
        act: Literal["swish", "gelu", "gelu-approx"] = "swish",
    ):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        exponent = (
            -math.log(10000.0)
            * torch.arange(0, half_dim, dtype=torch.float32)
            / half_dim
        )
        self.register_buffer("exponent", exponent, persistent=False)
        self.mlp = SwiGLUMlp(
            d_model,
            int(d_model * mlp_ratio),
            use_smooth_swiglu=use_smooth_swiglu,
            act=act,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(self.mlp.w1.weight.device)
        args = t[:, None] * self.exponent[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class MMDitBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.1,
        softcap: float = 20.0,
        learn_softmax: bool = True,
        softmax_bias: bool = False,
        softmax_scale_init: float = 0.43,
        context_len: int = 1024,
        use_fourier_embedding: bool = True,
        fourier_terms: int = 4,
        fourier_sigma: float = 0.1,
        mla_dim_rank: int = None,
        use_smooth_swiglu: bool = False,
        scale_rmsnorm: bool = False,
        act: Literal["swish", "gelu", "gelu-approx"] = "swish",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.self_norm = RMSNorm(
            d_model,
            eps=eps,
            elementwise_affine=scale_rmsnorm,
        )
        self.cross_norm = RMSNorm(
            d_model,
            eps=eps,
            elementwise_affine=scale_rmsnorm,
        )
        self.ffn_norm = RMSNorm(
            d_model,
            eps=eps,
            elementwise_affine=scale_rmsnorm,
        )

        self.self_attention = MultiheadLatentAttention(
            d_model,
            n_heads,
            n_kv_heads,
            bias=False,
            use_fourier_embedding=use_fourier_embedding,
            context_len=context_len,
            fourier_terms=fourier_terms,
            fourier_sigma=fourier_sigma,
            latent_dim_rank=mla_dim_rank,
            dropout=attn_dropout,
            softcap=softcap,
            learn_softmax=learn_softmax,
            softmax_bias=softmax_bias,
            softmax_scale_init=softmax_scale_init,
        )
        self.cross_attention = MultiheadLatentAttention(
            d_model,
            n_heads,
            n_kv_heads,
            bias=False,
            use_fourier_embedding=use_fourier_embedding,
            context_len=context_len,
            fourier_terms=fourier_terms,
            fourier_sigma=fourier_sigma,
            latent_dim_rank=mla_dim_rank,
            dropout=attn_dropout,
            softcap=softcap,
            learn_softmax=learn_softmax,
            softmax_bias=softmax_bias,
            softmax_scale_init=softmax_scale_init,
        )

        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = SwiGLUMlp(
            d_model, hidden_dim, use_smooth_swiglu=use_smooth_swiglu, act=act
        )

        self.adaln = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True)
        )
        # zero out adaln
        nn.init.constant_(self.adaln[-1].weight, 0)
        nn.init.constant_(self.adaln[-1].bias, 0)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca = self.adaln(
            t_emb
        ).chunk(6, dim=1)

        x_normed = self.self_norm(x) * (
            1 + scale_msa.unsqueeze(1)
        ) + shift_msa.unsqueeze(1)
        attn_out = self.self_attention(x_normed, x_normed)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_normed = self.cross_norm(x) * (
            1 + scale_mca.unsqueeze(1)
        ) + shift_mca.unsqueeze(1)
        attn_out = self.cross_attention(x_normed, c)
        x = x + gate_mca.unsqueeze(1) * attn_out

        x = x + self.mlp(self.ffn_norm(x))
        return x


class MMDiT(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        depth: int = 32,
        context_depth: int = 4,
        context_len: int = 4096,
        in_channels: int = 6,
        mlp_ratio: float = 6.0,
        dac_codebooks: int = 18,
        dac_vocab: int = 1024,  # [0, 1023]
        n_heads: int = 16,
        n_kv_heads: int = 8,
        softcap: float = 20.0,
        mla_dim_rank: int = 32,
        fourier_terms: int = 4,
        fourier_sigma: float = 0.1,
        dropout: float = 0.1,
        learn_softmax: bool = True,
        softmax_bias: bool = False,
        softmax_scale_init: float = 0.43,
        use_smooth_swiglu: bool = False,
        scale_rmsnorm: bool = False,
        eps: float = 1e-8,
        context_mlp_ratio: float = 2.0,
        activation_func: Literal["swish", "gelu", "gelu-approx"] = "swish",
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model

        self.use_checkpointing = use_checkpointing

        self.x_embedder = nn.Linear(in_channels, d_model)
        self.c_embedders = nn.ModuleList(
            [nn.Embedding(dac_vocab, d_model) for _ in range(dac_codebooks)]
        )
        self.t_embedder = TimestepEmbedding(
            d_model,
            mlp_ratio=mlp_ratio,
            use_smooth_swiglu=use_smooth_swiglu,
            act=activation_func,
        )

        context_dim = int(d_model * context_mlp_ratio)
        self.context_refiners = nn.ModuleList(
            [
                SwiGLUMlp(
                    d_model,
                    context_dim,
                    use_smooth_swiglu=use_smooth_swiglu,
                    act=activation_func,
                )
                for _ in range(context_depth)
            ]
        )

        self.cross_blocks = nn.ModuleList(
            [
                MMDitBlock(
                    d_model,
                    n_heads,
                    n_kv_heads,
                    mlp_ratio=mlp_ratio,
                    attn_dropout=dropout,
                    context_len=context_len,
                    softcap=softcap,
                    learn_softmax=learn_softmax,
                    softmax_bias=softmax_bias,
                    softmax_scale_init=softmax_scale_init,
                    fourier_terms=fourier_terms,
                    fourier_sigma=fourier_sigma,
                    mla_dim_rank=mla_dim_rank,
                    use_smooth_swiglu=use_smooth_swiglu,
                    scale_rmsnorm=scale_rmsnorm,
                    act=activation_func,
                    eps=eps,
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = RMSNorm(
            d_model,
            eps=eps,
            elementwise_affine=scale_rmsnorm,
        )
        self.final_proj = nn.Linear(d_model, in_channels)
        nn.init.constant_(self.final_proj.weight, 0)
        nn.init.constant_(self.final_proj.bias, 0)

    def forward(
        self, x: torch.FloatTensor, c: torch.ShortTensor, t: torch.LongTensor
    ) -> torch.FloatTensor:
        x = self.x_embedder(x.transpose(1, 2))

        c_list = []
        for i, emb in enumerate(self.c_embedders):
            c_list.append(emb(c[:, i, :]))
        c = torch.stack(c_list, dim=0).sum(dim=0)

        for block in self.context_refiners:
            c = block(c)

        t_emb = self.t_embedder(t)

        for block in self.cross_blocks:
            if self.training and self.use_checkpointing:
                x = checkpoint(
                    block, x, c, t_emb, use_reentrant=False, determinism_check="none"
                )
            else:
                x = block(x, c, t_emb)

        x = self.final_norm(x)
        pred_noise = self.final_proj(x).transpose(1, 2)
        return pred_noise
