import argparse
import os
from typing import List

import torch
import torchaudio

from .decode import ctc_decode
from .features import LogMelFeatureExtractor, SAMPLE_RATE
from .model import create_model
from .utils import indices_to_text


def load_model(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    vocab_size = len(ckpt["char2idx"])
    model = create_model(vocab_size=vocab_size)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["char2idx"], ckpt["idx2char"]


def transcribe_files(model_path: str, audio_paths: List[str]) -> None:
    model, char2idx, idx2char = load_model(model_path)
    device = torch.device("cpu")
    model.to(device)
    extractor = LogMelFeatureExtractor()

    for p in audio_paths:
        full_path = p
        if not os.path.isabs(full_path):
            full_path = os.path.join(os.getcwd(), full_path)
        try:
            wav, sr = torchaudio.load(full_path)
        except Exception as e:
            print(f"Failed to load {p}: {e}")
            continue

        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)

        if wav.dim() > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav_mono = wav.squeeze(0) if wav.dim() > 1 else wav
        feats = extractor(wav_mono)
        feats = feats.unsqueeze(0)  # (B=1, 1, n_mels, T)
        with torch.no_grad():
            log_probs = model(feats.to(device))
            decoded = ctc_decode(log_probs, beam_width=5)[0]  # Use beam search for better accuracy
            text = indices_to_text(decoded, idx2char)
        print(f"{p} -> {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--audio_path",
        type=str,
        nargs="+",
        required=True,
        help="One or more WAV files",
    )
    args = parser.parse_args()
    transcribe_files(args.checkpoint, args.audio_path)


if __name__ == "__main__":
    main()


