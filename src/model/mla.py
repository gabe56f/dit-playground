import math

import torch
import torch.nn as nn

from flash_attn import flash_attn_func


class FourierPositionEmbedding(nn.Module):
    def __init__(
        self,
        d_head: int,
        context_len: int,
        n_fourier_terms: int = 4,
        fourier_sigma: float = 0.1,
        fourier_learnable: bool = False,
    ):
        super().__init__()
        self.d_head = d_head
        self.context_len = context_len
        self.n_fourier_terms = n_fourier_terms
        dominant_freqs = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("dominant_freqs", dominant_freqs, persistent=False)
        fourier_freqs = torch.linspace(0, math.pi, n_fourier_terms)
        self.register_buffer("fourier_freqs", fourier_freqs, persistent=False)
        self.fourier_weights = nn.Parameter(
            torch.randn(2, d_head // 2, n_fourier_terms) * fourier_sigma,
            requires_grad=fourier_learnable,
        )
        self.floor_freq = 2 * math.pi / context_len
        self._fourier_terms_cache = None

    def _compute_fourier_terms(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        if (
            self._fourier_terms_cache is not None
            and self._fourier_terms_cache.shape[1] >= seq_len
        ):
            return self._fourier_terms_cache[:, :seq_len, :].to(device)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # Calculate Dominant Term: e^(i * ω_m * n)
        dom_freqs_grid = torch.outer(positions, self.dominant_freqs)
        freq_mask = self.dominant_freqs > self.floor_freq
        dom_freqs_grid[:, ~freq_mask] = 0.0
        dom_cos, dom_sin = torch.cos(dom_freqs_grid), torch.sin(dom_freqs_grid)
        # Calculate Fourier Series Term: Σ a_ω * e^(i * ω * n)
        fourier_freqs_grid = torch.outer(positions, self.fourier_freqs)
        fourier_cos, fourier_sin = torch.cos(fourier_freqs_grid), torch.sin(
            fourier_freqs_grid
        )
        # Calculate the sum using the weights: a_ω * cos(ωn) and a_ω * sin(ωn)
        # [d_head/2, n_terms] @ [n_terms, seq_len] -> [d_head/2, seq_len] -> [seq_len, d_head/2]
        weights = self.fourier_weights.float()
        fourier_sum_cos = (weights[0] @ fourier_cos.T).T
        fourier_sum_sin = (weights[1] @ fourier_sin.T).T
        # Total real part: cos(ω_m*n) + Σ a_cos * cos(ω*n)
        # Total imag part: sin(ω_m*n) + Σ a_sin * sin(ω*n)
        final_cos, final_sin = dom_cos + fourier_sum_cos, dom_sin + fourier_sum_sin
        self._fourier_terms_cache = torch.stack(
            (final_cos, final_sin), dim=-1
        ).unsqueeze(0)
        return self._fourier_terms_cache.to(device)

    def apply_fourier_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        B, n_heads, L, d_head = x.shape
        dtype = x.dtype
        x_complex = x.float().reshape(B, n_heads, L, d_head // 2, 2)
        x_real, x_imag = x_complex.unbind(-1)
        fourier_terms = self._compute_fourier_terms(L, x.device)
        f_cos, f_sin = fourier_terms.squeeze(0).unbind(-1)
        # Apply complex multiplication: (x_r + i*x_i) * (f_c + i*f_s)
        # = (x_r*f_c - x_i*f_s) + i*(x_r*f_s + x_i*f_c)
        x_rotated_real = x_real * f_cos - x_imag * f_sin
        x_rotated_imag = x_real * f_sin + x_imag * f_cos
        # Combine back and reshape
        x_rotated = torch.stack((x_rotated_real, x_rotated_imag), dim=-1)
        return x_rotated.flatten(3).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_fourier_pos_emb(x)


class MultiheadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        latent_dim_rank: int,
        use_fourier_embedding: bool = True,
        context_len: int = None,
        fourier_terms: int = 4,
        fourier_sigma: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_kv_heads <= n_heads, "n_kv_heads must be less than or equal to n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        ratio = n_heads // n_kv_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv_compress = nn.Linear(d_model, 2 * latent_dim_rank, bias=False)
        self.w_k_decompress = nn.Linear(latent_dim_rank, d_model // ratio, bias=False)
        self.w_v_decompress = nn.Linear(latent_dim_rank, d_model // ratio, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        if use_fourier_embedding:
            self.emb = FourierPositionEmbedding(
                d_head=self.d_head,
                context_len=context_len,
                n_fourier_terms=fourier_terms,
                fourier_sigma=fourier_sigma,
            )
        else:
            self.emb = nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # value is ignored 'cause we don't use it
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape

        q = self.w_q(query)
        kv_input = key
        latent_kv = self.w_kv_compress(kv_input)
        latent_k, latent_v = latent_kv.chunk(2, dim=-1)
        k = self.w_k_decompress(latent_k)
        v = self.w_v_decompress(latent_v)

        q = q.view(B, L_q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L_k, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L_k, self.n_kv_heads, self.d_head).transpose(1, 2)

        q = self.emb(q)
        k = self.emb(k)

        output = flash_attn_func(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0
        )

        # output = sageattn(q, k, v)

        # output = F.scaled_dot_product_attention(
        #     q, k, v, dropout_p=self.dropout.p if self.training else 0.0
        # )
        output = output.transpose(1, 2).contiguous().view(B, L_q, -1)
        return self.w_o(output), None
