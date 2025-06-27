import math
from typing import Literal, NamedTuple

import einops
import torch
from torch import nn

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True

    @triton.jit
    def rope_fused_kernel(
        IN_ptr,
        FREQ_ptr,
        OUT_ptr,
        in_stride_b,
        in_stride_h,
        in_stride_l,
        seq_len: tl.int32,
        BLOCK_SIZE_L: tl.constexpr,
        D_HEAD_HALF: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_l_block = tl.program_id(2)

        d_half_offsets = tl.arange(0, D_HEAD_HALF)
        l_offsets = pid_l_block * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
        l_mask = l_offsets < seq_len

        freqs = tl.load(FREQ_ptr + d_half_offsets)
        theta = l_offsets[:, None] * freqs[None, :]
        cos_theta = tl.cos(theta)
        sin_theta = tl.sin(theta)

        in1_ptr = (
            IN_ptr
            + (pid_b * in_stride_b + pid_h * in_stride_h)
            + (l_offsets[:, None] * in_stride_l + d_half_offsets[None, :])
        )
        in2_ptr = in1_ptr + D_HEAD_HALF

        out1_ptr = (
            OUT_ptr
            + (pid_b * in_stride_b + pid_h * in_stride_h)
            + (l_offsets[:, None] * in_stride_l + d_half_offsets[None, :])
        )
        out2_ptr = out1_ptr + D_HEAD_HALF

        in1 = tl.load(in1_ptr, mask=l_mask[:, None], other=0.0)
        in2 = tl.load(in2_ptr, mask=l_mask[:, None], other=0.0)

        # (x*c) + (rotate_half(x)*s)
        out1 = in1 * cos_theta - in2 * sin_theta
        out2 = in2 * cos_theta + in1 * sin_theta

        tl.store(out1_ptr, out1, mask=l_mask[:, None])
        tl.store(out2_ptr, out2, mask=l_mask[:, None])

    @triton.jit
    def rope_fused_backward_kernel(
        DOUT_grad_ptr,
        DIN_grad_ptr,
        FREQ_ptr,
        in_stride_b,
        in_stride_h,
        in_stride_l,
        seq_len: tl.int32,
        BLOCK_SIZE_L: tl.constexpr,
        D_HEAD_HALF: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_l_block = tl.program_id(2)

        d_half_offsets = tl.arange(0, D_HEAD_HALF)
        l_offsets = pid_l_block * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
        l_mask = l_offsets < seq_len

        # Recompute the frequencies instead of storing them
        freqs = tl.load(FREQ_ptr + d_half_offsets)
        theta = l_offsets[:, None] * freqs[None, :]
        cos_theta = tl.cos(theta)
        sin_theta = tl.sin(theta)

        dout1_grad_ptr = (
            DOUT_grad_ptr
            + (pid_b * in_stride_b + pid_h * in_stride_h)
            + (l_offsets[:, None] * in_stride_l + d_half_offsets[None, :])
        )
        dout2_grad_ptr = dout1_grad_ptr + D_HEAD_HALF

        din1_grad_ptr = (
            DIN_grad_ptr
            + (pid_b * in_stride_b + pid_h * in_stride_h)
            + (l_offsets[:, None] * in_stride_l + d_half_offsets[None, :])
        )
        din2_grad_ptr = din1_grad_ptr + D_HEAD_HALF

        dout1_grad = tl.load(dout1_grad_ptr, mask=l_mask[:, None], other=0.0)
        dout2_grad = tl.load(dout2_grad_ptr, mask=l_mask[:, None], other=0.0)

        # Apply the inverse rotation
        din1_grad = dout1_grad * cos_theta + dout2_grad * sin_theta
        din2_grad = dout2_grad * cos_theta - dout1_grad * sin_theta

        tl.store(din1_grad_ptr, din1_grad, mask=l_mask[:, None])
        tl.store(din2_grad_ptr, din2_grad, mask=l_mask[:, None])

    class _ApplyRoPEEmbTritonFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, freqs):
            q, k, freqs = (
                q.contiguous(),
                k.contiguous(),
                freqs.contiguous(),
            )
            batch_size, n_heads, seq_len, d_head = q.shape
            _, n_kv_heads, _, _ = k.shape

            q_rot = torch.empty_like(q)
            k_rot = torch.empty_like(k)

            D_HEAD_HALF = d_head // 2
            BLOCK_SIZE_L = 64 if seq_len > 64 else triton.next_power_of_2(seq_len)

            grid_q = (batch_size, n_heads, triton.cdiv(seq_len, BLOCK_SIZE_L))
            rope_fused_kernel[grid_q](
                q,
                freqs,
                q_rot,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                seq_len,
                BLOCK_SIZE_L=BLOCK_SIZE_L,
                D_HEAD_HALF=D_HEAD_HALF,
            )

            grid_k = (batch_size, n_kv_heads, triton.cdiv(seq_len, BLOCK_SIZE_L))
            rope_fused_kernel[grid_k](
                k,
                freqs,
                k_rot,
                k.stride(0),
                k.stride(1),
                k.stride(2),
                seq_len,
                BLOCK_SIZE_L=BLOCK_SIZE_L,
                D_HEAD_HALF=D_HEAD_HALF,
            )

            ctx.save_for_backward(freqs)
            ctx.q_shape, ctx.k_shape = q.shape, k.shape
            ctx.BLOCK_SIZE_L = BLOCK_SIZE_L
            return q_rot, k_rot

        @staticmethod
        def backward(ctx, dq_rot_grad, dk_rot_grad):
            (freqs,) = ctx.saved_tensors
            batch_size, n_heads, seq_len, d_head = ctx.q_shape
            _, n_kv_heads, _, _ = ctx.k_shape

            dq_rot_grad, dk_rot_grad = (
                dq_rot_grad.contiguous(),
                dk_rot_grad.contiguous(),
            )
            dq_grad = torch.empty_like(dq_rot_grad)
            dk_grad = torch.empty_like(dk_rot_grad)

            D_HEAD_HALF = d_head // 2
            BLOCK_SIZE_L = ctx.BLOCK_SIZE_L

            grid_q = (batch_size, n_heads, triton.cdiv(seq_len, BLOCK_SIZE_L))
            rope_fused_backward_kernel[grid_q](
                dq_rot_grad,
                dq_grad,
                freqs,
                dq_rot_grad.stride(0),
                dq_rot_grad.stride(1),
                dq_rot_grad.stride(2),
                seq_len,
                BLOCK_SIZE_L=BLOCK_SIZE_L,
                D_HEAD_HALF=D_HEAD_HALF,
            )

            grid_k = (batch_size, n_kv_heads, triton.cdiv(seq_len, BLOCK_SIZE_L))
            rope_fused_backward_kernel[grid_k](
                dk_rot_grad,
                dk_grad,
                freqs,
                dk_rot_grad.stride(0),
                dk_rot_grad.stride(1),
                dk_rot_grad.stride(2),
                seq_len,
                BLOCK_SIZE_L=BLOCK_SIZE_L,
                D_HEAD_HALF=D_HEAD_HALF,
            )

            return dq_grad, dk_grad, None

except ImportError:
    TRITON_AVAILABLE = False


class Terms(NamedTuple):
    cos: torch.Tensor
    sin: torch.Tensor


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        d_head: int,
        context_len: int,
        base: int = 10000,
        frequency_init: Literal["base", "resonance"] = "base",
        extending: Literal["none", "linear", "ntk"] = "ntk",
        scaling_factor: float = 1.0,
        **_,
    ):
        super().__init__()

        self.d_head = d_head
        self.context_len = context_len
        self.extending = extending
        self.base = base
        self.scaling_factor = scaling_factor
        self.frequency_init = frequency_init

        self.triton_fast_path = TRITON_AVAILABLE and frequency_init != "resonance"

        self._compute_inv_freq()

    def _compute_inv_freq(self, seq_len: int = None) -> None:
        base = self.base
        if seq_len is not None:
            # Recompute for NTK
            base = base * (self.scaling_factor * seq_len / self.context_len) - (
                self.scaling_factor - 1
            ) ** (self.d_head / (self.d_head - 2))

        freq = torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head
        inv_freq = 1.0 / (base**freq)

        # arXiv:2403.00071
        if self.frequency_init == "resonance":
            r_wavelengths = torch.round(2 * torch.pi / inv_freq)
            r_inv_freq = 2 * math.pi / r_wavelengths
            self.register_buffer("inv_freq", r_inv_freq, persistent=False)
            self.register_buffer("wavelengths", r_wavelengths, persistent=False)
        else:
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_terms(self, x: torch.Tensor, seq_len: int) -> Terms:
        if self.extending == "ntk" and seq_len > self.context_len:
            self._compute_inv_freq(seq_len=seq_len)

        # arXiv:2403.00071
        if self.frequency_init == "resonance":
            freqs_list = []
            for i in range(self.d_head // 2):
                if seq_len >= self.r_wavelengths[i].item():
                    positions_i = torch.arange(
                        self.r_wavelengths[i], device=x.device, dtype=torch.float32
                    )

                    if self.extending == "linear":
                        positions_i = positions_i / self.scaling_factor

                    current_freq = einops.repeat(
                        positions_i * self.inv_freq[i],
                        "l -> (repeat l)",
                        repeat=math.ceil(seq_len / self.r_wavelengths[i].item()),
                    ).reshape(-1)[None, :seq_len, None]
                else:
                    positions_i = torch.arange(
                        seq_len, device=x.device, dtype=torch.float32
                    )

                    if self.extending == "linear":
                        positions_i = positions_i / self.scaling_factor

                    current_freq = positions_i * self.inv_freq[None, i, None]
                freqs_list.append(current_freq)
            freqs = torch.stack(freqs_list, dim=-1)
        else:
            positions = torch.arange(0, seq_len, dtype=torch.float32, device=x.device)
            if self.extending == "linear":
                positions = positions / self.scaling_factor
            inv_freq = self.inv_freq[None, :, None].to(x.device).expand(seq_len, -1, 1)
            freqs = (inv_freq @ positions[:, None, :]).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1).type_as(x)
        sin = emb.sin().unsqueeze(1).type_as(x)

        return Terms(cos, sin)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, L, d_head_input = q.shape

        assert (
            d_head_input == self.d_head
        ), f"Expected d_head_input {self.d_head}, but got {d_head_input}."

        if self.extending == "ntk" and L > self.context_len:
            self._compute_inv_freq(seq_len=L)

        if self.triton_fast_path:
            freqs = self.inv_freq.to(q.device)
            if self.extending == "linear":
                freqs = freqs / self.scaling_factor

            return _ApplyRoPEEmbTritonFn.apply(q, k, freqs)

        q_terms = self._compute_terms(q, L)
        k_terms = self._compute_terms(k, L)

        q_rot = (q * q_terms.cos) + (self.rotate_half(q) * q_terms.sin)
        k_rot = (k * k_terms.cos) + (self.rotate_half(k) * k_terms.sin)

        return q_rot, k_rot
