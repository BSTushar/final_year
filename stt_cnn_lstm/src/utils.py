import json
import os
from typing import Dict, List, Tuple


VOCAB_CHARS = list("abcdefghijklmnopqrstuvwxyz '.,?-")


def build_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    char2idx = {"<blank>": 0}
    idx2char = {0: "<blank>"}
    idx = 1
    for ch in VOCAB_CHARS:
        char2idx[ch] = idx
        idx2char[idx] = ch
        idx += 1
    return char2idx, idx2char


def text_to_indices(text: str, char2idx: Dict[str, int]) -> List[int]:
    indices: List[int] = []
    for ch in text.lower():
        if ch in char2idx:
            indices.append(char2idx[ch])
    return indices


def indices_to_text(indices: List[int], idx2char: Dict[int, str]) -> str:
    chars: List[str] = []
    for i in indices:
        if i in idx2char and i != 0:
            chars.append(idx2char[i])
    return "".join(chars)


def _levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def cer(ref: str, hyp: str) -> float:
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    if len(ref_chars) == 0:
        return 0.0
    dist = _levenshtein(ref_chars, hyp_chars)
    return dist / max(1, len(ref_chars))


def wer(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return 0.0
    dist = _levenshtein(ref_words, hyp_words)
    return dist / max(1, len(ref_words))


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, obj) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        safe_mkdir(dir_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def update_history(
    history_path: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    train_wer: float = None,
    val_wer: float = None,
    val_cer: float = None,
    error_stats: Dict[str, float] = None,
) -> None:
    history = load_json(history_path, default={"epochs": []})
    epoch_data = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "train_acc": float(train_acc),
        "val_wer": float(val_wer) if val_wer is not None else 1.0,
        "val_cer": float(val_cer) if val_cer is not None else 1.0,
        "error_stats": error_stats or {},
    }
    if train_wer is not None:
        epoch_data["train_wer"] = float(train_wer)
    history["epochs"].append(epoch_data)
    save_json(history_path, history)


