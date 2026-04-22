import io
import json
import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment

from .preprocessing import preprocess_audio
from .types import STTResult


class LocalOfflineEngine:
    """
    Primary offline STT engine.

    Uses local Vosk model with audio preprocessing to provide robust on-device
    transcription and confidence/noise signals for routing decisions.
    """

    def __init__(self, model_dir: str, sample_rate: int = 16000) -> None:
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self._vosk_model = None
        self._vosk_model_cls = None
        self._recognizer_cls = None

    def _ensure_model_loaded(self) -> None:
        if self._vosk_model is not None:
            return
        try:
            from vosk import KaldiRecognizer, Model as VoskModel
        except Exception as exc:
            raise RuntimeError("Vosk package is unavailable for offline mode.") from exc

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Offline model directory not found at '{self.model_dir}'."
            )
        self._vosk_model_cls = VoskModel
        self._recognizer_cls = KaldiRecognizer
        self._vosk_model = self._vosk_model_cls(self.model_dir)

    def _decode_audio_bytes(self, raw_bytes: bytes) -> Tuple[torch.Tensor, int]:
        try:
            wav, sr = torchaudio.load(io.BytesIO(raw_bytes), format="wav")
            return wav, sr
        except Exception:
            pass

        tmp_webm_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_webm:
                tmp_webm.write(raw_bytes)
                tmp_webm_path = tmp_webm.name
            audio = AudioSegment.from_file(tmp_webm_path, format="webm")
            audio = audio.set_frame_rate(self.sample_rate).set_channels(1)
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            wav, sr = torchaudio.load(wav_buffer, format="wav")
            return wav, sr
        finally:
            if tmp_webm_path and os.path.exists(tmp_webm_path):
                try:
                    os.unlink(tmp_webm_path)
                except Exception:
                    pass

    def transcribe(self, raw_bytes: bytes) -> STTResult:
        self._ensure_model_loaded()
        wav, sr = self._decode_audio_bytes(raw_bytes)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
            sr = self.sample_rate

        if wav.dim() > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav_mono = wav.squeeze(0) if wav.dim() > 1 else wav

        if wav_mono.numel() < 400:
            return STTResult(error="Audio too short. Please record at least 0.5 seconds.")

        processed, metrics = preprocess_audio(wav_mono)
        duration_sec = float(processed.shape[0] / float(sr))

        wav_np = processed.detach().cpu().numpy()
        wav_np = np.clip(wav_np, -1.0, 1.0).astype("float32")
        pcm16 = (wav_np * 32767).astype("int16").tobytes()

        recognizer = self._recognizer_cls(self._vosk_model, self.sample_rate)
        recognizer.AcceptWaveform(pcm16)
        raw_result = recognizer.FinalResult()

        try:
            result_json = json.loads(raw_result)
        except Exception:
            result_json = {}

        text = (result_json.get("text") or "").strip()
        words = result_json.get("result", []) or []
        if words:
            confs = [w.get("conf", 0.0) for w in words if "conf" in w]
            avg_conf = float(sum(confs) / max(len(confs), 1))
        else:
            avg_conf = 0.0

        if not text:
            return STTResult(
                transcription="",
                warning="Offline recognizer returned empty text.",
                engine="local_offline",
                confidence=avg_conf,
                noise_db=metrics["noise_db"],
                duration_sec=duration_sec,
                metadata={"vad_ratio": metrics["vad_ratio"]},
            )

        return STTResult(
            transcription=text,
            engine="local_offline",
            confidence=avg_conf,
            noise_db=metrics["noise_db"],
            duration_sec=duration_sec,
            metadata={"vad_ratio": metrics["vad_ratio"]},
        )

