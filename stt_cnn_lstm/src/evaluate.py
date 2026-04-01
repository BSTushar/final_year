import argparse
import os

import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

from .decode import ctc_decode
from .features import LogMelFeatureExtractor, SAMPLE_RATE
from .model import create_model
from .utils import cer, wer, indices_to_text


def load_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    model = create_model(vocab_size=len(ckpt["char2idx"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["char2idx"], ckpt["idx2char"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.csv)
    if "path" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain 'path' and 'text'")

    model, char2idx, idx2char = load_checkpoint(args.checkpoint)
    device = torch.device("cpu")
    model.to(device)
    extractor = LogMelFeatureExtractor()

    total_wer = 0.0
    total_cer = 0.0
    n = 0

    print(f"\nEvaluating on {len(df)} samples...")
    print(f"{'='*80}\n")

    # Create progress bar for evaluation
    pbar = tqdm(
        df.iterrows(),
        total=len(df),
        desc="Evaluating",
        unit="sample",
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    with torch.no_grad():
        for _, row in pbar:
            path = str(row["path"])
            ref = str(row["text"])
            full_path = path
            if not os.path.isabs(full_path):
                full_path = os.path.join(os.getcwd(), full_path)
            try:
                wav, sr = torchaudio.load(full_path)
            except Exception:
                continue

            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)

            if wav.dim() > 1 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav_mono = wav.squeeze(0) if wav.dim() > 1 else wav
            feats = extractor(wav_mono)
            feats = feats.unsqueeze(0)  # (B=1, 1, n_mels, T)
            log_probs = model(feats.to(device))  # (T, B, C)
            decoded = ctc_decode(log_probs, beam_width=5)[0]  # Use beam search for better accuracy
            hyp = indices_to_text(decoded, idx2char)

            w = wer(ref, hyp)
            c = cer(ref, hyp)
            total_wer += w
            total_cer += c
            n += 1

            # Update progress bar with current metrics
            current_wer = total_wer / n
            current_cer = total_cer / n
            pbar.set_postfix({
                'WER': f'{current_wer:.4f}',
                'CER': f'{current_cer:.4f}',
                'samples': n
            })

    if n == 0:
        print("No valid samples for evaluation.")
        return

    print(f"\n{'='*80}")
    print(f"Evaluation Results:")
    print(f"  Total samples: {n}")
    print(f"  WER: {total_wer / n:.4f}")
    print(f"  CER: {total_cer / n:.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


