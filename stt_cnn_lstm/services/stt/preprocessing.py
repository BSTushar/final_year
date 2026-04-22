from typing import Dict, Tuple

import torch


def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    peak = torch.max(torch.abs(waveform)).item()
    if peak <= 1e-8:
        return waveform
    return torch.clamp(waveform / peak, -1.0, 1.0)


def suppress_background_noise(waveform: torch.Tensor, gate_db: float = -42.0) -> torch.Tensor:
    gate_linear = 10.0 ** (gate_db / 20.0)
    # Soft noise gate to reduce very low-energy background components.
    return torch.where(torch.abs(waveform) < gate_linear, waveform * 0.2, waveform)


def voice_activity_ratio(waveform: torch.Tensor, threshold_db: float = -35.0) -> float:
    threshold = 10.0 ** (threshold_db / 20.0)
    active = (torch.abs(waveform) > threshold).float()
    return float(active.mean().item())


def preprocess_audio(waveform: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    normalized = normalize_audio(waveform)
    denoised = suppress_background_noise(normalized)
    vad_ratio = voice_activity_ratio(denoised)
    rms = torch.sqrt(torch.mean(denoised**2) + 1e-8)
    noise_db = float(20.0 * torch.log10(rms).item())
    return denoised, {
        "vad_ratio": vad_ratio,
        "noise_db": noise_db,
    }

