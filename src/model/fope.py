import math

import torch
from torch import nn

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True

    def _configs_compute():
        conf = []
        for warps in [4, 8]:
            for block_size_seq in [8, 16, 32]:
                for block_size_d_half in [16, 32, 64]:
                    conf.append(
                        triton.Config(
                            {
                                "BLOCK_SIZE_SEQ": block_size_seq,
                                "BLOCK_SIZE_D_HALF": block_size_d_half,
                            },
                            num_warps=warps,
                        )
                    )
        return conf

    @triton.autotune(
        configs=_configs_compute(),
        key=["SEQ_LEN", "D_HEAD_HALF", "N_FOURIER_TERMS"],
    )
    @triton.jit
    def _compute_fourier_terms_kernel_forward(
        output_terms_ptr,  # [seq_len, d_head_half, 2]
        positions_ptr,  # [seq_len,]
        dominant_freqs_ptr,  # [d_head_half,]
        fourier_freqs_ptr,  # [N_FOURIER_TERMS,]
        fourier_weights_ptr,  # [2, d_head_half, N_FOURIER_TERMS]
        floor_freq,
        SEQ_LEN: tl.constexpr,
        D_HEAD_HALF: tl.constexpr,
        N_FOURIER_TERMS: tl.constexpr,
        stride_out_seq,
        stride_out_d,
        stride_out_complex,
        stride_pos_seq,
        stride_df_d,
        stride_ff_n,
        stride_fw_type,
        stride_fw_d,
        stride_fw_n,
        BLOCK_SIZE_SEQ: tl.constexpr,
        BLOCK_SIZE_D_HALF: tl.constexpr,
    ):
        pid_seq = tl.program_id(0)
        pid_d_half = tl.program_id(1)

        offsets_seq_tile = tl.arange(0, BLOCK_SIZE_SEQ)
        offsets_d_tile = tl.arange(0, BLOCK_SIZE_D_HALF)

        actual_seq_indices = pid_seq * BLOCK_SIZE_SEQ + offsets_seq_tile
        actual_d_indices = pid_d_half * BLOCK_SIZE_D_HALF + offsets_d_tile

        mask_seq = actual_seq_indices < SEQ_LEN
        mask_d = actual_d_indices < D_HEAD_HALF

        pos = tl.load(
            positions_ptr + actual_seq_indices * stride_pos_seq,
            mask=mask_seq,
            other=0.0,
        )

        dom_freq = tl.load(
            dominant_freqs_ptr + actual_d_indices * stride_df_d, mask=mask_d, other=0.0
        )

        fourier_sum_cos_acc = tl.zeros(
            (BLOCK_SIZE_SEQ, BLOCK_SIZE_D_HALF), dtype=tl.float32
        )
        fourier_sum_sin_acc = tl.zeros(
            (BLOCK_SIZE_SEQ, BLOCK_SIZE_D_HALF), dtype=tl.float32
        )

        # Accumulate Fourier series terms
        for k in range(N_FOURIER_TERMS):
            f_freq_k = tl.load(fourier_freqs_ptr + k * stride_ff_n)

            weight_cos_k = tl.load(
                fourier_weights_ptr
                + 0 * stride_fw_type
                + actual_d_indices * stride_fw_d
                + k * stride_fw_n,
                mask=mask_d,
                other=0.0,
            )
            weight_sin_k = tl.load(
                fourier_weights_ptr
                + 1 * stride_fw_type
                + actual_d_indices * stride_fw_d
                + k * stride_fw_n,
                mask=mask_d,
                other=0.0,
            )

            # Compute k-th fourier term: pos changes along seq, f_freq_k is fixed for this k
            # fourier_freqs_grid_val_k will be [BLOCK_SIZE_SEQ, 1] after broadcasting pos
            fourier_freqs_grid_val_k = pos[:, None] * f_freq_k
            f_cos_k = tl.cos(fourier_freqs_grid_val_k)
            f_sin_k = tl.sin(fourier_freqs_grid_val_k)

            # Accumulate: ([BLOCK_SIZE_SEQ, 1] * [1, BLOCK_SIZE_D_HALF]) -> [BLOCK_SIZE_SEQ, BLOCK_SIZE_D_HALF]
            fourier_sum_cos_acc += f_cos_k * weight_cos_k[None, :]
            fourier_sum_sin_acc += f_sin_k * weight_sin_k[None, :]

        dom_freqs_grid_val = pos[:, None] * dom_freq[None, :]

        dom_freq_active_mask = dom_freq > floor_freq
        dom_freqs_grid_val = tl.where(
            dom_freq_active_mask[None, :],
            dom_freqs_grid_val,
            tl.zeros_like(dom_freqs_grid_val),
        )

        dom_cos = tl.cos(dom_freqs_grid_val)
        dom_sin = tl.sin(dom_freqs_grid_val)

        final_cos = dom_cos + fourier_sum_cos_acc
        final_sin = dom_sin + fourier_sum_sin_acc

        out_ptr_offset_seq = (
            pid_seq * BLOCK_SIZE_SEQ + offsets_seq_tile[:, None]
        ) * stride_out_seq
        out_ptr_offset_d = (
            pid_d_half * BLOCK_SIZE_D_HALF + offsets_d_tile[None, :]
        ) * stride_out_d

        out_ptr_cos_base = (
            output_terms_ptr
            + out_ptr_offset_seq
            + out_ptr_offset_d
            + 0 * stride_out_complex
        )
        out_ptr_sin_base = (
            output_terms_ptr
            + out_ptr_offset_seq
            + out_ptr_offset_d
            + 1 * stride_out_complex
        )

        mask_seq_d = mask_seq[:, None] & mask_d[None, :]

        tl.store(out_ptr_cos_base, final_cos, mask=mask_seq_d)
        tl.store(out_ptr_sin_base, final_sin, mask=mask_seq_d)

    def _configs_apply():
        conf = []
        for warps in [4, 8]:
            for block_size_l in [4, 8, 16]:
                for block_size_d_half in [8, 16, 32]:
                    conf.append(
                        triton.Config(
                            {
                                "BLOCK_SIZE_L": block_size_l,
                                "BLOCK_SIZE_D_HALF": block_size_d_half,
                            },
                            num_warps=warps,
                        )
                    )
        return conf

    @triton.autotune(
        configs=_configs_apply(),
        key=[
            "L",
            "D_HEAD_HALF",
            "B",
            "N_HEADS",
            "IS_FORWARD",
            "NEEDS_GRAD_X",
            "NEEDS_GRAD_F",
        ],
    )
    @triton.jit
    def _apply_fourier_kernel(
        out_real_ptr,
        out_imag_ptr,
        in_real_ptr,
        in_imag_ptr,
        f_cos_ptr,
        f_sin_ptr,
        grad_out_real_ptr,
        grad_out_imag_ptr,
        grad_in_real_ptr,
        grad_in_imag_ptr,
        grad_f_cos_ptr,
        grad_f_sin_ptr,
        B: tl.constexpr,
        N_HEADS: tl.constexpr,
        L: tl.constexpr,
        D_HEAD_HALF: tl.constexpr,
        stride_x_b,
        stride_x_h,
        stride_x_l,
        stride_x_d,
        stride_f_l,
        stride_f_d,
        IS_FORWARD: tl.constexpr,
        NEEDS_GRAD_X: tl.constexpr,
        NEEDS_GRAD_F: tl.constexpr,
        BLOCK_SIZE_L: tl.constexpr,
        BLOCK_SIZE_D_HALF: tl.constexpr,
    ):
        pid_batch_head = tl.program_id(0)
        pid_l_tile = tl.program_id(1)
        pid_d_tile = tl.program_id(2)

        current_b = pid_batch_head // N_HEADS
        current_h = pid_batch_head % N_HEADS

        offsets_l_in_tile = tl.arange(0, BLOCK_SIZE_L)
        offsets_d_in_tile = tl.arange(0, BLOCK_SIZE_D_HALF)

        actual_l_indices = pid_l_tile * BLOCK_SIZE_L + offsets_l_in_tile
        actual_d_indices = pid_d_tile * BLOCK_SIZE_D_HALF + offsets_d_in_tile

        mask_l = actual_l_indices < L
        mask_d = actual_d_indices < D_HEAD_HALF

        x_base_offset = current_b * stride_x_b + current_h * stride_x_h

        x_tile_offset_l = actual_l_indices[:, None] * stride_x_l
        x_tile_offset_d = actual_d_indices[None, :] * stride_x_d
        x_tile_offset = x_base_offset + x_tile_offset_l + x_tile_offset_d

        f_tile_offset_l = actual_l_indices[:, None] * stride_f_l
        f_tile_offset_d = actual_d_indices[None, :] * stride_f_d
        f_tile_offset = f_tile_offset_l + f_tile_offset_d

        mask_ld = mask_l[:, None] & mask_d[None, :]

        f_c = tl.load(f_cos_ptr + f_tile_offset, mask=mask_ld, other=0.0)
        f_s = tl.load(f_sin_ptr + f_tile_offset, mask=mask_ld, other=0.0)

        if IS_FORWARD:
            x_r = tl.load(in_real_ptr + x_tile_offset, mask=mask_ld, other=0.0)
            x_i = tl.load(in_imag_ptr + x_tile_offset, mask=mask_ld, other=0.0)

            # Complex multiplication: (x_r + i*x_i) * (f_c + i*f_s)
            rot_r = x_r * f_c - x_i * f_s
            rot_i = x_r * f_s + x_i * f_c

            # Store rotated output
            tl.store(out_real_ptr + x_tile_offset, rot_r, mask=mask_ld)
            tl.store(out_imag_ptr + x_tile_offset, rot_i, mask=mask_ld)
        else:
            # Load grad_output (dL/d rot_r, dL/d rot_i)
            grad_rot_r = tl.load(
                grad_out_real_ptr + x_tile_offset, mask=mask_ld, other=0.0
            )
            grad_rot_i = tl.load(
                grad_out_imag_ptr + x_tile_offset, mask=mask_ld, other=0.0
            )

            if NEEDS_GRAD_X:
                # For dL/dx: x_r, x_i are not strictly needed here, but usually passed if needed by dL/df
                # dL/dx_r = dL/d rot_r * f_c + dL/d rot_i * f_s
                # dL/dx_i = -dL/d rot_r * f_s + dL/d rot_i * f_c
                grad_x_r = grad_rot_r * f_c + grad_rot_i * f_s
                grad_x_i = -grad_rot_r * f_s + grad_rot_i * f_c
                tl.store(grad_in_real_ptr + x_tile_offset, grad_x_r, mask=mask_ld)
                tl.store(grad_in_imag_ptr + x_tile_offset, grad_x_i, mask=mask_ld)

            if NEEDS_GRAD_F:
                x_r = tl.load(in_real_ptr + x_tile_offset, mask=mask_ld, other=0.0)
                x_i = tl.load(in_imag_ptr + x_tile_offset, mask=mask_ld, other=0.0)

                # dL/df_c = dL/d rot_r * x_r + dL/d rot_i * x_i
                # dL/df_s = -dL/d rot_r * x_i + dL/d rot_i * x_r
                grad_f_c_val = grad_rot_r * x_r + grad_rot_i * x_i
                grad_f_s_val = -grad_rot_r * x_i + grad_rot_i * x_r

                # Atomic add for gradients of f_cos and f_sin as they are shared across B, N_HEADS
                tl.atomic_add(
                    grad_f_cos_ptr + f_tile_offset, grad_f_c_val, mask=mask_ld
                )
                tl.atomic_add(
                    grad_f_sin_ptr + f_tile_offset, grad_f_s_val, mask=mask_ld
                )

    class _ComputeFourierTermsTritonFn(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            positions,
            dominant_freqs,
            fourier_freqs,
            fourier_weights,
            floor_freq,
            seq_len,
            d_head_half,
            n_fourier_terms,
        ):

            output_terms = torch.empty(
                (seq_len, d_head_half, 2), device=positions.device, dtype=torch.float32
            )

            def grid(meta):
                return (
                    triton.cdiv(seq_len, meta["BLOCK_SIZE_SEQ"]),
                    triton.cdiv(d_head_half, meta["BLOCK_SIZE_D_HALF"]),
                )

            _compute_fourier_terms_kernel_forward[grid](
                output_terms,
                positions,
                dominant_freqs,
                fourier_freqs,
                fourier_weights,
                floor_freq,
                seq_len,
                d_head_half,
                n_fourier_terms,
                output_terms.stride(0),
                output_terms.stride(1),
                output_terms.stride(2),
                positions.stride(0),
                dominant_freqs.stride(0),
                fourier_freqs.stride(0),
                fourier_weights.stride(0),
                fourier_weights.stride(1),
                fourier_weights.stride(2),
            )

            ctx.save_for_backward(positions, fourier_freqs, fourier_weights)
            ctx.floor_freq = floor_freq
            ctx.seq_len = seq_len
            ctx.d_head_half = d_head_half
            ctx.n_fourier_terms = n_fourier_terms

            return output_terms

    class _ApplyFourierEmbTritonFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, fourier_terms_combined, B, n_heads, L, d_head_half):
            if fourier_terms_combined.dim() == 4:
                fourier_terms_squeezed = fourier_terms_combined.squeeze(0)
            else:
                fourier_terms_squeezed = fourier_terms_combined

            x_dtype = x.dtype
            x_float = x.float()  # Calculations in float32
            x_reshaped = x_float.reshape(B, n_heads, L, d_head_half, 2)
            x_real = x_reshaped[..., 0].contiguous()
            x_imag = x_reshaped[..., 1].contiguous()

            f_cos = fourier_terms_squeezed[..., 0].contiguous()  # [L, d_head_half]
            f_sin = fourier_terms_squeezed[..., 1].contiguous()  # [L, d_head_half]

            out_real = torch.empty_like(x_real)
            out_imag = torch.empty_like(x_imag)

            def grid(meta):
                return (
                    B * n_heads,
                    triton.cdiv(L, meta["BLOCK_SIZE_L"]),
                    triton.cdiv(d_head_half, meta["BLOCK_SIZE_D_HALF"]),
                )

            _apply_fourier_kernel[grid](
                out_real,
                out_imag,
                x_real,
                x_imag,
                f_cos,
                f_sin,
                None,
                None,
                None,
                None,
                None,
                None,  # No grad ptrs for forward
                B,
                n_heads,
                L,
                d_head_half,
                x_real.stride(0),
                x_real.stride(1),
                x_real.stride(2),
                x_real.stride(3),
                f_cos.stride(0),
                f_cos.stride(1),
                IS_FORWARD=True,
                NEEDS_GRAD_X=False,
                NEEDS_GRAD_F=False,
            )

            x_rotated = torch.stack((out_real, out_imag), dim=-1).flatten(start_dim=3)

            ctx.save_for_backward(x_real, x_imag, f_cos, f_sin)
            ctx.B, ctx.n_heads, ctx.L, ctx.d_head_half = B, n_heads, L, d_head_half
            ctx.x_dtype = x_dtype
            ctx.needs_grad_x = x.requires_grad
            ctx.needs_grad_fourier_terms = fourier_terms_combined.requires_grad

            return x_rotated.to(x_dtype)

        @staticmethod
        def backward(ctx, grad_x_rotated):
            x_real, x_imag, f_cos, f_sin = ctx.saved_tensors
            B, n_heads, L, d_head_half = ctx.B, ctx.n_heads, ctx.L, ctx.d_head_half
            x_dtype = ctx.x_dtype

            grad_x_rotated_float = grad_x_rotated.float()
            grad_x_rot_reshaped = grad_x_rotated_float.reshape(
                B, n_heads, L, d_head_half, 2
            )
            grad_out_real = grad_x_rot_reshaped[..., 0].contiguous()
            grad_out_imag = grad_x_rot_reshaped[..., 1].contiguous()

            grad_x = None
            grad_fourier_terms_combined = None  # [L, d_head_half, 2]

            grad_in_real = torch.empty_like(x_real) if ctx.needs_grad_x else None
            grad_in_imag = torch.empty_like(x_imag) if ctx.needs_grad_x else None

            # Gradients for f_cos, f_sin need to be initialized to zero for atomic_add
            grad_f_cos = (
                torch.zeros_like(f_cos) if ctx.needs_grad_fourier_terms else None
            )
            grad_f_sin = (
                torch.zeros_like(f_sin) if ctx.needs_grad_fourier_terms else None
            )

            def grid(meta):
                return (
                    B * n_heads,
                    triton.cdiv(L, meta["BLOCK_SIZE_L"]),
                    triton.cdiv(d_head_half, meta["BLOCK_SIZE_D_HALF"]),
                )

            _apply_fourier_kernel[grid](
                None,
                None,
                x_real,
                x_imag,
                f_cos,
                f_sin,
                grad_out_real,
                grad_out_imag,
                grad_in_real,
                grad_in_imag,
                grad_f_cos,
                grad_f_sin,
                B,
                n_heads,
                L,
                d_head_half,
                x_real.stride(0),
                x_real.stride(1),
                x_real.stride(2),
                x_real.stride(3),
                f_cos.stride(0),
                f_cos.stride(1),
                IS_FORWARD=False,
                NEEDS_GRAD_X=ctx.needs_grad_x,
                NEEDS_GRAD_F=ctx.needs_grad_fourier_terms,
            )

            if ctx.needs_grad_x:
                grad_x = (
                    torch.stack((grad_in_real, grad_in_imag), dim=-1)
                    .flatten(start_dim=3)
                    .to(x_dtype)
                )

            if ctx.needs_grad_fourier_terms:
                # Result from kernel is [L, d_head_half, 2]
                # Match original fourier_terms_combined shape, usually [1, L, d_head_half, 2]
                grad_fourier_terms_combined = torch.stack(
                    (grad_f_cos, grad_f_sin), dim=-1
                ).unsqueeze(0)

            return (
                grad_x,
                grad_fourier_terms_combined,
                None,
                None,
                None,
                None,
            )  # For B, n_heads, L, d_head_half

except ImportError:
    TRITON_AVAILABLE = False


class FourierPositionEmbedding(nn.Module):
    def __init__(
        self,
        d_head: int,
        context_len: int,
        n_fourier_terms: int = 4,
        fourier_sigma: float = 0.1,
        fourier_learnable: bool = False,
        triton_override: bool = False,
    ):
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError("d_head must be even for Fourier Position Embedding.")
        assert (
            TRITON_AVAILABLE and not triton_override
        ) and not fourier_learnable, (
            "Backwards not implemented for fourier weights in Triton."
        )

        self.d_head = d_head
        self.d_head_half = d_head // 2
        self.context_len = context_len
        self.n_fourier_terms = n_fourier_terms

        dominant_freqs = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("dominant_freqs", dominant_freqs, persistent=False)

        fourier_freqs = torch.linspace(0, math.pi, n_fourier_terms)
        self.register_buffer("fourier_freqs", fourier_freqs, persistent=False)

        self.fourier_weights = nn.Parameter(
            torch.randn(2, self.d_head_half, n_fourier_terms) * fourier_sigma,
            requires_grad=fourier_learnable,
        )
        self.floor_freq = 2 * math.pi / context_len
        self._fourier_terms_cache = None
        self.triton_override = triton_override

    def _compute_fourier_terms_triton(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        if (
            self._fourier_terms_cache is not None
            and self._fourier_terms_cache.shape[1] >= seq_len
            and self._fourier_terms_cache.device == device
        ):
            return self._fourier_terms_cache[:, :seq_len, :, :]

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)

        dominant_freqs = self.dominant_freqs.to(device=device, dtype=torch.float32)
        fourier_freqs = self.fourier_freqs.to(device=device, dtype=torch.float32)

        current_fourier_weights = self.fourier_weights.to(
            device=device, dtype=torch.float32
        ).detach()

        fourier_terms = _ComputeFourierTermsTritonFn.apply(
            positions,
            dominant_freqs,
            fourier_freqs,
            current_fourier_weights,
            self.floor_freq,
            seq_len,
            self.d_head_half,
            self.n_fourier_terms,
        )

        fourier_terms_combined = fourier_terms.unsqueeze(0)
        self._fourier_terms_cache = fourier_terms_combined.to(device)
        return self._fourier_terms_cache

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
        if not TRITON_AVAILABLE or self.triton_override:
            return self.apply_fourier_pos_emb(x)

        B, n_heads, L, d_head_input = x.shape
        if d_head_input != self.d_head:
            raise ValueError(
                f"Input d_head ({d_head_input}) does not match module d_head ({self.d_head})"
            )

        fourier_terms_combined = self._compute_fourier_terms_triton(L, x.device)

        x_rotated = _ApplyFourierEmbTritonFn.apply(
            x,
            fourier_terms_combined,
            B,
            n_heads,
            L,
            self.d_head_half,
        )
        return x_rotated
