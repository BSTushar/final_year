import torch
import torchaudio
from torch import Tensor


SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
WINDOW_FN = torch.hann_window


class LogMelFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_rate = SAMPLE_RATE
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            window_fn=WINDOW_FN,
            center=True,
            power=2.0,
            normalized=False,
            f_min=0.0,
            f_max=None,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.mel(waveform)
        mel = torch.log(mel + 1e-9)

        mean = mel.mean(dim=-1, keepdim=True)
        std = mel.std(dim=-1, keepdim=True)
        std = torch.clamp(std, min=1e-9)
        mel = (mel - mean) / std

        return mel


def extract_log_mel(batch_waveforms: Tensor) -> Tensor:
    """
    Convenience function for batched extraction.

    Args:
        batch_waveforms: list/tuple of Tensors or a padded Tensor (B, T)
    Returns:
        batch_features: Tensor (B, 1, n_mels, time)
    """
    extractor = LogMelFeatureExtractor()
    features = []
    for wav in batch_waveforms:
        feat = extractor(wav)
        features.append(feat)

    max_time = max(f.shape[-1] for f in features)
    batch = torch.zeros(
        len(features),
        1,
        N_MELS,
        max_time,
        dtype=features[0].dtype,
    )
    for i, f in enumerate(features):
        t = f.shape[-1]
        batch[i, :, :, :t] = f
    return batch


