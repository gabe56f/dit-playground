import math

from einops import rearrange
import numpy as np
import torch
from torch import nn


def WNConv1d(*args, **kwargs) -> torch.Tensor:
    return nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))


@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    eps = 1e-9
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + eps).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = (6 * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        model_dim: int = 64,
        strides: list[int] = [2, 4, 8, 8],
        latent_dim: int = 64,
    ):
        super().__init__()
        self.block = [WNConv1d(1, model_dim, kernel_size=7, padding=3)]
        for stride in strides:
            model_dim *= 2
            self.block += [EncoderBlock(model_dim, stride=stride)]

        self.block += [
            Snake1d(model_dim),
            WNConv1d(model_dim, latent_dim, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*self.block)
        self.enc_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VectorQuantize(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)
        z_q = z_e + (z_q - z_e).detach()
        return self.out_proj(z_q), indices

    def embed_code(self, embed_id: int) -> torch.Tensor:
        return nn.functional.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: int) -> torch.Tensor:
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        encodings = nn.functional.normalize(encodings)
        codebook = nn.functional.normalize(codebook)
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList(
            [VectorQuantize(input_dim, codebook_size, dim) for dim in codebook_dim]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        residual = z
        z_q = 0
        indices = []
        for i, quantizer in enumerate(self.quantizers):
            z_q_i, ind_i = quantizer(residual)
            z_q = z_q + z_q_i
            residual = residual - z_q_i
            indices.append(ind_i)
        return torch.stack(indices, dim=1)


class DAC(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list[int] = [2, 4, 8, 8],
        latent_dim=128,
        n_codebooks: int = 18,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.sample_rate = sample_rate
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )
        self.delay = self.get_delay()

    def get_delay(self):
        l_out = self.get_output_length(0)
        L = l_out

        layers = []
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(layer)

        for layer in reversed(layers):
            d = layer.dilation[0]
            k = layer.kernel_size[0]
            s = layer.stride[0]

            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                L = (L - 1) * s + d * (k - 1) + 1
            L = math.ceil(L)
        l_in = L
        return (l_in - l_out) // 2

    def get_output_length(self, input_length: int) -> int:
        L = input_length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]
                k = layer.kernel_size[0]
                s = layer.stride[0]

                if isinstance(layer, nn.ConvTranspose1d):
                    L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.Conv1d):
                    L = (L - 1) * s + d * (k - 1) + 1
                L = math.floor(L)
        return L

    def preprocess(self, audio_data: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data

    def encode(self, audio_data: torch.Tensor) -> torch.Tensor:
        z = self.encoder(audio_data)
        return self.quantizer(z)

    def chunked_encode(
        self, waveform: torch.Tensor, sample_rate: int, chunk_size: int = 2
    ) -> torch.Tensor:
        x = waveform.unsqueeze(1)
        chunk_size = int(chunk_size * sample_rate)
        remainer = chunk_size % self.hop_length
        chunk_size = chunk_size - remainer
        audio_length = x.shape[-1]
        s_list = []
        for start in range(0, audio_length, chunk_size):
            end = start + chunk_size
            chunk = x[:, :, start:end]
            chunk = self.preprocess(chunk, sample_rate)
            s = self.encode(chunk.to(self.device)).cpu()
            s_list.append(s)
        return torch.cat(s_list, dim=2).short()
