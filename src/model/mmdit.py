import math
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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
            self.gate_norm = nn.LayerNorm(
                hidden_features, elementwise_affine=False, eps=1e-6
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
        context_len: int = 1024,
        use_fourier_embedding: bool = True,
        fourier_terms: int = 4,
        fourier_sigma: float = 0.1,
        use_mla: bool = True,
        mla_dim_rank: int = None,
        use_smooth_swiglu: bool = False,
        act: Literal["swish", "gelu", "gelu-approx"] = "swish",
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.cross_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        if use_mla:
            self.self_attention = MultiheadLatentAttention(
                d_model,
                n_heads,
                n_kv_heads,
                use_fourier_embedding=use_fourier_embedding,
                context_len=context_len,
                fourier_terms=fourier_terms,
                fourier_sigma=fourier_sigma,
                latent_dim_rank=mla_dim_rank,
                dropout=attn_dropout,
            )
            self.cross_attention = MultiheadLatentAttention(
                d_model,
                n_heads,
                n_kv_heads,
                use_fourier_embedding=use_fourier_embedding,
                context_len=context_len,
                fourier_terms=fourier_terms,
                fourier_sigma=fourier_sigma,
                latent_dim_rank=mla_dim_rank,
                dropout=attn_dropout,
            )
        else:
            self.self_attention = nn.MultiheadAttention(
                d_model,
                n_heads,
                batch_first=True,
                need_weights=False,
                dropout=attn_dropout,
            )
            self.cross_attention = nn.MultiheadAttention(
                d_model,
                n_heads,
                batch_first=True,
                need_weights=False,
                dropout=attn_dropout,
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
        attn_out = self.self_attention(x_normed, x_normed, x_normed)[0]
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_normed = self.cross_norm(x) * (
            1 + scale_mca.unsqueeze(1)
        ) + shift_mca.unsqueeze(1)
        attn_out = self.cross_attention(x_normed, c, c)[0]
        x = x + gate_mca.unsqueeze(1) * attn_out

        x = x + self.mlp(self.ffn_norm(x))
        return x


class MMDiT(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        depth: int = 32,
        context_len: int = 4096,
        in_channels: int = 6,
        mlp_ratio: float = 6.0,
        dac_codebooks: int = 18,
        dac_vocab: int = 1024,  # [0, 1023]
        n_heads: int = 12,
        n_kv_heads: int = 6,
        use_mla: bool = True,
        mla_dim_rank: int = 32,
        fourier_terms: int = 4,
        fourier_sigma: float = 0.1,
        dropout: float = 0.1,
        use_smooth_swiglu: bool = False,
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

        self.blocks = nn.ModuleList(
            [
                MMDitBlock(
                    d_model,
                    n_heads,
                    n_kv_heads,
                    mlp_ratio,
                    dropout,
                    context_len,
                    fourier_terms=fourier_terms,
                    fourier_sigma=fourier_sigma,
                    use_mla=use_mla,
                    mla_dim_rank=mla_dim_rank,
                    use_smooth_swiglu=use_smooth_swiglu,
                    act=activation_func,
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_proj = nn.Linear(d_model, in_channels)
        nn.init.constant_(self.final_proj.weight, 0)
        nn.init.constant_(self.final_proj.bias, 0)

    def forward(
        self, x: torch.FloatTensor, c: torch.ShortTensor, t: torch.LongTensor
    ) -> torch.FloatTensor:
        x = self.x_embedder(x.transpose(1, 2))
        # no idea if this works with torch.compile, maybe rewrite as a for loop?
        c_list = [emb(c[:, i, :]) for i, emb in enumerate(self.c_embedders)]
        c = torch.stack(c_list, dim=0).sum(dim=0)
        t_emb = self.t_embedder(t)

        for block in self.blocks:
            if self.training and self.use_checkpointing:
                x = checkpoint(
                    block, x, c, t_emb, use_reentrant=False, determinism_check="none"
                )
            else:
                x = block(x, c, t_emb)

        x = self.final_norm(x)
        pred_noise = self.final_proj(x).transpose(1, 2)
        return pred_noise
