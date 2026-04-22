from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class STTResult:
    transcription: str = ""
    engine: str = "unknown"
    confidence: Optional[float] = None
    noise_db: Optional[float] = None
    duration_sec: Optional[float] = None
    warning: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    used_fallback: bool = False

    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "transcription": self.transcription,
            "engine": self.engine,
            "used_fallback": self.used_fallback,
            "metadata": self.metadata,
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.noise_db is not None:
            payload["noise_db"] = self.noise_db
        if self.duration_sec is not None:
            payload["duration_sec"] = self.duration_sec
        if self.warning:
            payload["warning"] = self.warning
        if self.error:
            payload["error"] = self.error
        return payload

