from safetensors.torch import load_file
import torchaudio
import torch
import numpy as np
import librosa

from .dac import DAC


def load_audio(file: str) -> tuple[torch.Tensor, int]:
    return torchaudio.load(file)


def resample_audio(
    waveform: torch.Tensor, original_sr: int, target_sr: int
) -> torch.Tensor:
    if original_sr != target_sr:
        waveform = torchaudio.transforms.Resample(
            orig_freq=original_sr, new_freq=target_sr
        )(waveform)
    return waveform


def to_mono_channel(waveform: torch.Tensor) -> torch.Tensor:
    n_channels = waveform.shape[0]
    if n_channels > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def load_dac(file: str, sr: int = 44100, device: str = "cuda:0") -> DAC:
    dac = DAC(sample_rate=sr)
    dac.load_state_dict(load_file(file, device=device), strict=False)
    dac = dac.to(device, torch.bfloat16)
    dac.device = device
    return dac


def encode(dac: DAC, file: str, chunk_size: int = 2) -> torch.Tensor:
    waveform, sample_rate = load_audio(file)
    waveform = resample_audio(waveform, sample_rate, dac.sample_rate)
    waveform = to_mono_channel(waveform)
    waveform = waveform.to(dac.device, torch.bfloat16)
    encoded = dac.chunked_encode(
        waveform, sample_rate=dac.sample_rate, chunk_size=chunk_size
    )
    return encoded


def get_frame_times(spec, dac: DAC) -> np.ndarray:
    return (
        librosa.frames_to_time(
            np.arange(spec.shape[-1]),
            sr=dac.sample_rate,
            hop_length=dac.hop_length,
        )
        * 1000
    )
