import math
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm.auto import tqdm
from wandb.wandb_run import Run

from .model.adabelief import AdaBelief8bit
from .model.mmdit import MMDiT
from .beatmaps.encode import CursorSignals


class MapDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        context_len: int,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.context_len = context_len
        self.index = []

        for artist_dir in self.dataset_dir.iterdir():
            if not artist_dir.is_dir():
                continue

            for audio_dir in artist_dir.iterdir():
                if not audio_dir.is_dir():
                    continue

                try:
                    spec_path = audio_dir / "spec.pt"
                    map_paths = list(audio_dir.glob("*.map.pt"))

                    if not spec_path.exists() or not map_paths:
                        continue

                    if self.context_len > 0:
                        s_shape = np.load(spec_path, mmap_mode="r").shape

                        num_chunks = math.ceil(s_shape[-1] / self.context_len)

                        for map_path in map_paths:
                            for i in range(num_chunks):
                                self.index.append((map_path, spec_path, i))
                    else:
                        # Load whole file:
                        for map_path in map_paths:
                            self.index.append((map_path, spec_path, 0))

                except Exception as e:
                    print(f"Error indexing {artist_dir.name}/{audio_dir.name}: {e}")

        print(f"Index built. Found {len(self.index)} samples.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        map_path, spec_path, chunk_idx = self.index[idx]

        try:
            # Use mmap_mode='r' to avoid loading the whole file into RAM
            spec_np: np.ndarray = np.load(spec_path, mmap_mode="r")
            map_np: np.ndarray = np.load(map_path, mmap_mode="r")
        except EOFError:
            print(f"EOFError loading {spec_path}. Skipping.")
            return None, None
        except FileNotFoundError:
            print(f"FileNotFoundError loading {spec_path}. Skipping.")
            return None, None

        if self.context_len > 0:
            # Calculate padding and slice coordinates
            start = chunk_idx * self.context_len
            end = start + self.context_len
            original_len = spec_np.shape[-1]

            spec_chunk_unpadded = spec_np[..., start : min(end, original_len)]
            map_chunk_unpadded = map_np[..., start : min(end, original_len)]

            pad_amount = max(0, end - original_len)
            if pad_amount > 0:
                spec_chunk_padded = np.pad(
                    spec_chunk_unpadded,
                    ((0, 0), (0, pad_amount)),
                )
                map_chunk_padded = np.pad(
                    map_chunk_unpadded,
                    ((0, 0), (0, pad_amount)),
                )
            else:
                spec_chunk_padded = spec_chunk_unpadded
                map_chunk_padded = map_chunk_unpadded

            spec_tensor = torch.from_numpy(spec_chunk_padded.copy()).int()
            map_tensor = torch.from_numpy(map_chunk_padded.copy()).bfloat16()
            del spec_chunk_padded, map_chunk_padded

            return map_tensor, spec_tensor
        else:
            spec_tensor = torch.from_numpy(spec_np.copy()).int()
            map_tensor = torch.from_numpy(map_np.copy()).bfloat16()

            return map_tensor, spec_tensor


class FlowMatchingScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        sigma_min: float = 1e-4,
        loss_gamma: float = 5.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.sigma_min = sigma_min
        self.loss_gamma = loss_gamma

    def _alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_t = t
        sigma_t = torch.clamp(1 - t, min=self.sigma_min)
        return alpha_t, sigma_t

    def add_noise(
        self,
        x0: torch.Tensor,
        eps: torch.Tensor,
        timesteps: torch.Tensor,
        return_snr: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timesteps.max() >= self.num_train_timesteps or timesteps.min() < 0:
            raise ValueError(f"Timesteps must be in [0, {self.num_train_timesteps-1}]")
        t_c = (timesteps.bfloat16() + 1) / self.num_train_timesteps
        alpha_t, sigma_t = self._alpha_sigma(t_c)
        view_shape = (x0.shape[0],) + (1,) * (x0.dim() - 1)
        alpha_t = alpha_t.view(view_shape)
        sigma_t = sigma_t.view(view_shape)

        if return_snr:
            snr = (alpha_t**2) / (sigma_t**2 + 1e-8)
            snr = snr.nan_to_num(0.0)
            snr = torch.clamp(snr, max=self.loss_gamma)

            return alpha_t * x0 + sigma_t * eps, snr
        else:
            return alpha_t * x0 + sigma_t * eps


class Trainer:
    def __init__(self, config, run: Run):
        self.config = config
        self.run = run

        self.dtype = torch.bfloat16
        self.device = torch.device(config["device"])
        self.model = MMDiT(use_checkpointing=True, **config["model_config"])
        self.model.to(device=self.device, dtype=self.dtype)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            f"Model initialized with {total_params / 1e6:.2f}M parameters, of which {trainable_params / 1e6:.2f}M are trainable."
        )

        self.optimizer = AdaBelief8bit(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            bf16_stochastic_round=True,
        )

        self.noise_scheduler = FlowMatchingScheduler(
            num_train_timesteps=self.config["num_timesteps"]
        )

        self._setup_dataloaders()
        T_max = config["epochs"]
        if config["learning_rate_stepping"] == "step":
            T_max *= len(self.train_dataloader) // config["gradient_accumulation"]

        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_max, eta_min=1e-12
        )

        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # tensor so torch.compile can (i think) more easier inline it
        self.cursor_factor = torch.tensor(0.05, device=self.device, dtype=self.dtype)

    # Can't use cudagraphs, cause of low-bit ao compiler
    @torch.compile(fullgraph=False, mode="max-autotune-no-cudagraphs")
    def _step(self):
        # Just here in case gradients explode and a whole epoch is ruined
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

    @torch.compile(fullgraph=True, mode="max-autotune")
    def _loss(
        self,
        xp: torch.Tensor,
        xp_cursors: torch.Tensor,
        x0: torch.Tensor,
        x0_cursors: torch.Tensor,
        snr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # don't reduce since we have per-batch snr weights
        pixel_loss = F.huber_loss(xp, x0, reduction="none", delta=1.0)
        pixel_loss = pixel_loss.mean(dim=tuple(range(1, pixel_loss.dim())))
        pixel_loss = (snr * pixel_loss).mean()

        cursor_diff = x0_cursors[:, :, 1:] - x0_cursors[:, :, :-1]
        pred_cursor_diff = xp_cursors[:, :, 1:] - xp_cursors[:, :, :-1]
        cursor_diff = torch.tanh(cursor_diff * 20)
        pred_cursor_diff = torch.tanh(pred_cursor_diff * 20)

        # don't reduce since we have per-batch snr weights
        cursor_loss = F.mse_loss(cursor_diff, pred_cursor_diff, reduction="none")
        cursor_loss = cursor_loss.mean(dim=tuple(range(1, cursor_loss.dim())))
        cursor_loss = (snr * cursor_loss).mean()

        loss = pixel_loss + self.cursor_factor * cursor_loss
        return pixel_loss, cursor_loss, loss

    def _load_checkpoint(self, ckpt: str):
        checkpoint = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device=self.device, dtype=torch.bfloat16)
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except KeyError:
            pass
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        torch.cuda.empty_cache()
        print(f"Loaded checkpoint from {ckpt}")

    def _setup_dataloaders(self):
        dataset = MapDataset(**self.config["dataset_config"])

        def override_collate(batch):
            batch = list(filter(lambda x: x[0] is not None, batch))
            return default_collate(batch)

        self.train_dataloader = DataLoader(
            dataset,
            collate_fn=override_collate,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=6,
            pin_memory=True,
        )
        print(
            f"Effective batch size is: {self.config['batch_size'] * self.config['gradient_accumulation']}. (batch * gradient_accum)"
        )

    def _train_one_epoch(self, epoch):
        self.model.train()
        progress = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        samples = len(self.train_dataloader)
        total_loss = 0.0

        for step, (x0, c) in enumerate(progress):
            # no whining from cudagraph
            torch.compiler.cudagraph_mark_step_begin()

            x0 = x0.to(device=self.device, dtype=self.dtype)
            c = c.to(device=self.device, dtype=torch.int)
            B = x0.shape[0]

            # Sample timesteps and add noise
            timesteps = torch.randint(
                0, self.config["num_timesteps"], (B,), device=self.device
            )
            eps = torch.randn_like(x0, device=self.device, dtype=x0.dtype)

            # SNR for min-snr-γ loss weighting
            xt, snr = self.noise_scheduler.add_noise(x0, eps, timesteps)
            # print(snr.mean().item(), snr.max().item(), snr.min().item())
            xp = self.model(xt, c, timesteps)

            xp_cursors = xp[:, CursorSignals, :]
            x0_cursors = x0[:, CursorSignals, :]
            pixel_loss, cursor_loss, loss = self._loss(
                xp, xp_cursors, x0, x0_cursors, snr
            )
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

            loss: torch.Tensor = loss / self.config["gradient_accumulation"]
            total_loss += loss.item()
            loss.backward()

            if ((step + 1) % self.config["gradient_accumulation"] == 0) or (
                step + 1 == samples
            ):
                if self.config["learning_rate_stepping"] == "step":
                    self.lr_scheduler.step()
                self.run.log(
                    {
                        "loss/total": total_loss,
                        "loss/cursor": cursor_loss.detach(),
                        "loss/pixel": pixel_loss.detach(),
                        "step": step + epoch * samples,
                        "epoch": epoch + step / samples,
                        **(
                            {
                                "lr": self.optimizer.param_groups[0]["lr"],
                            }
                            if self.config["learning_rate_stepping"] == "step"
                            else {}
                        ),
                    }
                )
                self._step()
                self.optimizer.zero_grad()
                total_loss = 0.0
            del x0, c, xp_cursors, x0_cursors

    def train(self):
        print("Starting training...")
        self.run.log({"step": 0, "lr": self.optimizer.param_groups[0]["lr"]})
        for epoch in range(self.config["epochs"]):
            self._train_one_epoch(epoch)

            if self.config["learning_rate_stepping"] == "epoch":
                self.lr_scheduler.step()
                self.run.log(
                    {
                        "step": epoch * len(self.train_dataloader),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            if (epoch + 1) % self.config["save_every"] == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                    "config": self.config,
                }
                ckpt_path = self.checkpoint_dir / f"ckpt_epoch_{epoch+1}.pt"
                torch.save(checkpoint, ckpt_path)
                del checkpoint

                print(f"Saved checkpoint to {ckpt_path.as_posix()}")
        print("Training finished.")
