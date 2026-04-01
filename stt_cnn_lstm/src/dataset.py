import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from .features import LogMelFeatureExtractor, SAMPLE_RATE
from .utils import build_vocab, text_to_indices


@dataclass
class Sample:
    path: str
    text: str


class AudioAugmentation:
    """
    Data augmentation for speech recognition.
    Inspired by techniques from the referenced project.
    
    Includes:
    - SpecAugment (time/frequency masking)
    - Speed perturbation
    - Optional noise injection
    """
    
    def __init__(self, 
                 apply_specaugment: bool = True,
                 apply_speed: bool = True,
                 apply_noise: bool = False,
                 noise_factor: float = 0.01):
        self.apply_specaugment = apply_specaugment
        self.apply_speed = apply_speed
        self.apply_noise = apply_noise
        self.noise_factor = noise_factor
        
        # SpecAugment parameters (time and frequency masking)
        if self.apply_specaugment:
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=15)
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
    
    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Augment raw waveform with speed perturbation and noise."""
        # Speed perturbation (0.9x, 1.0x, or 1.1x speed)
        if self.apply_speed and random.random() < 0.5:
            speed_factor = random.choice([0.9, 1.0, 1.1])
            # speed() returns (waveform, lengths) tuple, we only need waveform
            waveform, _ = torchaudio.functional.speed(
                waveform.unsqueeze(0), 
                orig_freq=SAMPLE_RATE, 
                factor=speed_factor
            )
            waveform = waveform.squeeze(0)
        
        # Noise injection (optional, can be enabled if needed)
        if self.apply_noise and random.random() < 0.3:
            noise = torch.randn_like(waveform) * self.noise_factor
            waveform = waveform + noise
        
        return waveform
    
    def augment_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Augment spectrogram with SpecAugment (time/frequency masking).
        
        Args:
            spec: (1, n_mels, time) or (n_mels, time) spectrogram
            
        Returns:
            Augmented spectrogram
        """
        if not self.apply_specaugment:
            return spec
        
        if random.random() < 0.7:  # Apply 70% of the time
            # Ensure correct shape: (1, n_mels, time)
            if spec.dim() == 2:
                spec = spec.unsqueeze(0)
            
            # Apply time masking (mask consecutive time steps)
            spec = self.time_mask(spec)
            
            # Apply frequency masking (mask consecutive frequency bins)
            spec = self.freq_mask(spec)
            
            # Remove batch dimension if it was added
            if spec.shape[0] == 1 and spec.dim() == 3:
                spec = spec.squeeze(0)
        
        return spec


class SpeechDataset(Dataset):
    def __init__(self, csv_path: str, augment: bool = False):
        """
        Args:
            csv_path: Path to CSV manifest file
            augment: If True, apply data augmentation (for training)
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if "path" not in df.columns or "text" not in df.columns:
            raise ValueError("CSV must contain 'path' and 'text' columns")

        self.samples: List[Sample] = []
        for _, row in df.iterrows():
            path = str(row["path"])
            text = str(row["text"])
            if isinstance(path, str) and isinstance(text, str):
                self.samples.append(Sample(path=path, text=text))

        self.char2idx, self.idx2char = build_vocab()
        self.extractor = LogMelFeatureExtractor()
        self.augment = augment
        
        # Initialize augmentation if enabled
        if augment:
            self.audio_aug = AudioAugmentation(
                apply_specaugment=True,
                apply_speed=True,
                apply_noise=False  # Can enable if you have noise files
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        full_path = sample.path
        if not os.path.isabs(full_path):
            full_path = os.path.join(os.getcwd(), full_path)
        try:
            wav, sr = torchaudio.load(full_path)
        except Exception:
            return None

        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)

        if wav.dim() > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if wav.numel() < 400:
            return None

        wav_mono = wav.squeeze(0) if wav.dim() > 1 else wav
        
        # Apply waveform augmentation (speed perturbation, noise)
        if self.augment:
            wav_mono = self.audio_aug.augment_waveform(wav_mono)
        
        # Extract features
        feats = self.extractor(wav_mono)
        
        # Apply spectrogram augmentation (SpecAugment)
        if self.augment:
            feats = self.audio_aug.augment_spectrogram(feats)
        
        target_indices = text_to_indices(sample.text, self.char2idx)
        if len(target_indices) == 0:
            return None

        return {
            "features": feats,
            "target": torch.tensor(target_indices, dtype=torch.long),
            "text": sample.text,
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    feats = [b["features"] for b in batch]
    for i, f in enumerate(feats):
        if f.dim() == 2:
            f = f.unsqueeze(0)
        if f.shape[0] > 1:
            f = f[:1]
        feats[i] = f
    max_T = max(f.shape[-1] for f in feats)
    n_mels = feats[0].shape[1]
    B = len(feats)
    feat_batch = torch.zeros(B, 1, n_mels, max_T, dtype=feats[0].dtype)
    input_lengths = torch.zeros(B, dtype=torch.long)
    for i, f in enumerate(feats):
        T = f.shape[-1]
        feat_batch[i, :, :, :T] = f
        input_lengths[i] = T

    targets = [b["target"] for b in batch]
    target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets, dim=0)

    texts = [b["text"] for b in batch]

    return {
        "features": feat_batch,
        "input_lengths": input_lengths,
        "targets": targets_concat,
        "target_lengths": target_lengths,
        "texts": texts,
    }


def create_dataloader(csv_path: str, batch_size: int, shuffle: bool, augment: bool = False) -> Tuple[DataLoader, dict, dict]:
    """
    Create a DataLoader for speech recognition.
    
    Args:
        csv_path: Path to CSV manifest
        batch_size: Batch size
        shuffle: Whether to shuffle
        augment: Whether to apply data augmentation (use True for training)
        
    Returns:
        DataLoader, char2idx dict, idx2char dict
    """
    dataset = SpeechDataset(csv_path, augment=augment)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return loader, dataset.char2idx, dataset.idx2char


