import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

from .fallbackProvider import FallbackProvider, GeminiFallbackProvider, NoopFallbackProvider
from .localEngine import LocalOfflineEngine
from .types import STTResult


@dataclass
class HybridRouterConfig:
    min_confidence: float = 0.70
    max_noise_level_db: float = 30.0
    min_vad_ratio: float = 0.10
    failed_recognition_threshold: int = 2
    fallback_timeout_s: float = 10.0

    @staticmethod
    def from_env() -> "HybridRouterConfig":
        return HybridRouterConfig(
            min_confidence=float(os.getenv("STT_MIN_CONFIDENCE", "0.70")),
            max_noise_level_db=float(os.getenv("STT_MAX_NOISE_LEVEL_DB", "30.0")),
            min_vad_ratio=float(os.getenv("STT_MIN_VAD_RATIO", "0.10")),
            failed_recognition_threshold=int(os.getenv("STT_FAIL_THRESHOLD", "2")),
            fallback_timeout_s=float(os.getenv("STT_FALLBACK_TIMEOUT_S", "10")),
        )


class HybridSTTRouter:
    """
    Local-first router with policy-based failover.

    Routing decision priorities:
    1) Local inference remains primary.
    2) Fallback only for low confidence / high noise / repeated failure.
    """

    def __init__(self, local_engine: LocalOfflineEngine, fallback_provider: FallbackProvider, config: HybridRouterConfig):
        self.local_engine = local_engine
        self.fallback_provider = fallback_provider
        self.config = config
        self._consecutive_failures = 0

    @staticmethod
    def build_default() -> "HybridSTTRouter":
        model_dir = os.getenv("VOSK_MODEL_DIR", os.path.join("pretrained_models", "vosk-en"))
        local_engine = LocalOfflineEngine(model_dir=model_dir)
        fallback_name = os.getenv("STT_FALLBACK_PROVIDER", "none").strip().lower()

        if fallback_name == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if api_key:
                provider: FallbackProvider = GeminiFallbackProvider(
                    api_key=api_key,
                    model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                    retries=int(os.getenv("STT_FALLBACK_RETRIES", "2")),
                )
            else:
                provider = NoopFallbackProvider()
        else:
            provider = NoopFallbackProvider()

        return HybridSTTRouter(
            local_engine=local_engine,
            fallback_provider=provider,
            config=HybridRouterConfig.from_env(),
        )

    def _log_event(self, event: str, payload: Dict[str, object]) -> None:
        log_obj = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        logger.info("%s", json.dumps(log_obj, ensure_ascii=True))

    def _should_fallback(self, local: STTResult) -> Tuple[bool, str]:
        confidence = local.confidence if local.confidence is not None else 0.0
        vad_ratio = float(local.metadata.get("vad_ratio", 0.0))
        noise_level_db = abs(local.noise_db) if local.noise_db is not None else 0.0
        weak_text = not local.transcription or len(local.transcription.strip()) < 2

        if self._consecutive_failures >= self.config.failed_recognition_threshold:
            return True, "repeated_failed_recognition"
        if weak_text:
            return True, "empty_or_weak_transcription"
        if confidence < self.config.min_confidence:
            return True, "low_confidence"
        if noise_level_db > self.config.max_noise_level_db:
            return True, "excessive_background_noise"
        if vad_ratio < self.config.min_vad_ratio:
            return True, "low_voice_activity"
        return False, ""

    def transcribe(self, raw_bytes: bytes, mode: str = "enhanced") -> STTResult:
        run_mode = (mode or "enhanced").strip().lower()
        local = self.local_engine.transcribe(raw_bytes)
        local.metadata["mode"] = run_mode

        if local.error:
            self._consecutive_failures += 1
            self._log_event("local_error", {"error": local.error, "failures": self._consecutive_failures})
            if run_mode == "offline":
                return local
            fallback = self.fallback_provider.transcribe(raw_bytes, timeout_s=self.config.fallback_timeout_s)
            fallback.metadata["fallback_reason"] = "local_engine_error"
            return fallback

        if local.transcription:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        if run_mode == "offline":
            self._log_event("offline_only", {"engine": local.engine})
            return local

        should_fallback, reason = self._should_fallback(local)
        self._log_event(
            "router_decision",
            {
                "mode": run_mode,
                "should_fallback": should_fallback,
                "reason": reason,
                "confidence": local.confidence,
                "noise_db": local.noise_db,
                "failures": self._consecutive_failures,
            },
        )

        if not should_fallback:
            return local

        fallback = self.fallback_provider.transcribe(raw_bytes, timeout_s=self.config.fallback_timeout_s)
        fallback.metadata["fallback_reason"] = reason

        def _with_local_audio_metrics(result: STTResult) -> STTResult:
            """Fallback APIs do not report level/duration; reuse values from local preprocessing."""
            return replace(
                result,
                noise_db=result.noise_db if result.noise_db is not None else local.noise_db,
                duration_sec=result.duration_sec if result.duration_sec is not None else local.duration_sec,
            )

        if fallback.error and local.transcription:
            self._log_event(
                "fallback_failed",
                {
                    "reason": reason,
                    "fallback_engine": fallback.engine,
                    "fallback_error": fallback.error,
                },
            )
            # Fail-open to local result if fallback provider fails.
            local.metadata["fallback_reason"] = reason
            local.metadata["fallback_error"] = fallback.error
            local.warning = (
                local.warning + " | " if local.warning else ""
            ) + f"Fallback unavailable ({reason}): {fallback.error}. Returned offline result."
            return local
        return _with_local_audio_metrics(fallback)

