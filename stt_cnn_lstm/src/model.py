from typing import Tuple

import torch
import torch.nn as nn


class CNNSpeechEncoder(nn.Module):
    def __init__(self, n_mels: int, cnn_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(cnn_channels)
        self.conv2 = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_channels)
        self.relu = nn.ReLU(inplace=True)
        self.n_mels = n_mels
        self.cnn_channels = cnn_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        B, C, M, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T, C * M)
        x = x.permute(1, 0, 2).contiguous()
        return x


class CNNLSTMCTC(nn.Module):
    def __init__(self, vocab_size: int, n_mels: int = 80, hidden_size: int = 256):
        super().__init__()
        self.encoder = CNNSpeechEncoder(n_mels=n_mels, cnn_channels=64)
        input_size = 64 * n_mels
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(features)
        out, _ = self.lstm(enc)
        logits = self.fc(out)
        log_probs = self.log_softmax(logits)
        return log_probs


def create_model(vocab_size: int, n_mels: int = 80) -> CNNLSTMCTC:
    return CNNLSTMCTC(vocab_size=vocab_size, n_mels=n_mels)


