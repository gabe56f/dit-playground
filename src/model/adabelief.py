from typing import Optional

import torch
from torchao.optim.adam import (
    _AdamBase,
    _fp32_to_bf16_sr,
    OptimState8bit,
    OptimState4bit,
    OptimStateFp8,
)


def single_param_adabelief(
    p: torch.Tensor,
    grad: torch.Tensor,
    step: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    max_exp_avg_sq: Optional[torch.Tensor],
    lr: torch.Tensor,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    IS_ADAMW: bool,
    BF16_STOCHASTIC_ROUND: bool,
):
    # compute in FP32 for accurate calculations
    p_f32 = p.float()
    grad_f32 = grad.float()

    if IS_ADAMW:
        p_f32 = p_f32 - lr * weight_decay * p_f32
    else:
        grad_f32 = grad_f32 + weight_decay * p_f32

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    exp_avg_f32 = exp_avg.float().lerp(grad_f32, 1 - beta1)

    # The "belief" in the gradient is (grad - exp_avg)
    grad_residual = grad_f32 - exp_avg_f32
    exp_avg_sq_f32 = exp_avg_sq.float().lerp(grad_residual.square(), 1 - beta2)

    exp_avg.copy_(exp_avg_f32)
    exp_avg_sq.copy_(exp_avg_sq_f32)

    if max_exp_avg_sq is not None:
        max_exp_avg_sq_f32 = torch.maximum(max_exp_avg_sq.float(), exp_avg_sq_f32)
        max_exp_avg_sq.copy_(max_exp_avg_sq_f32)
        denom = (max_exp_avg_sq_f32.sqrt() / bias_correction2.sqrt()) + eps
    else:
        denom = (exp_avg_sq_f32.sqrt() / bias_correction2.sqrt()) + eps

    step_size = lr / bias_correction1
    p_f32 = p_f32 - step_size * exp_avg_f32 / denom

    if BF16_STOCHASTIC_ROUND:
        p.copy_(_fp32_to_bf16_sr(p_f32))
    else:
        p.copy_(p_f32)


class _AdaBelief(_AdamBase):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        weight_decouple: bool = True,
        *,
        block_size: int,
        bf16_stochastic_round: bool = False,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
            is_adamw=weight_decouple,
        )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch._dynamo.utils.disable_cache_limit():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError(
                            "AdaBelief does not support sparse gradients."
                        )

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = torch.tensor(0.0)
                        state["exp_avg"] = self._new_buffer(p, True)
                        state["exp_avg_sq"] = self._new_buffer(p, False)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = self._new_buffer(p, False)

                    state["step"] += 1

                    if not isinstance(group["lr"], torch.Tensor):
                        raise RuntimeError(
                            "lr was changed to a non-Tensor object. If you want to update lr, please use "
                            "optim.param_groups[0]['lr'].fill_(new_lr)"
                        )

                    torch.compile(
                        single_param_adabelief, fullgraph=True, dynamic=False
                    )(
                        p.detach(),
                        grad,
                        state["step"],
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state.get("max_exp_avg_sq", None),
                        group["lr"],
                        group["betas"][0],
                        group["betas"][1],
                        group["weight_decay"],
                        group["eps"],
                        self.is_adamw,
                        self.bf16_stochastic_round and p.dtype is torch.bfloat16,
                    )

        return loss


class AdaBelief8bit(_AdaBelief):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        weight_decouple: bool = True,
        *,
        block_size: int = 256,
        bf16_stochastic_round: bool = False,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            weight_decouple=weight_decouple,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
        )

    @staticmethod
    def _subclass_zeros(p: torch.Tensor, signed: bool, block_size: int):
        return OptimState8bit.zeros(p.shape, signed, block_size, p.device)


class AdaBelief4bit(_AdaBelief):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        weight_decouple: bool = True,
        *,
        block_size: int = 128,
        bf16_stochastic_round: bool = False,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            weight_decouple=weight_decouple,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
        )

    @staticmethod
    def _subclass_zeros(p: torch.Tensor, signed: bool, block_size: int):
        return OptimState4bit.zeros(p.shape, signed, block_size, p.device)


class AdaBeliefFp8(_AdaBelief):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        weight_decouple: bool = True,
        *,
        block_size: int = 256,
        bf16_stochastic_round: bool = False,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            weight_decouple=weight_decouple,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
        )

    @staticmethod
    def _subclass_zeros(p: torch.Tensor, signed: bool, block_size: int):
        return OptimStateFp8.zeros(p.shape, signed, block_size, p.device)
