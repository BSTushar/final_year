import base64
import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import List, Optional

from .types import STTResult


class FallbackProvider(ABC):
    @abstractmethod
    def transcribe(self, raw_bytes: bytes, timeout_s: float = 10.0) -> STTResult:
        raise NotImplementedError


class NoopFallbackProvider(FallbackProvider):
    def transcribe(self, raw_bytes: bytes, timeout_s: float = 10.0) -> STTResult:
        return STTResult(
            transcription="",
            engine="fallback_disabled",
            warning="Fallback provider is disabled.",
            error="Fallback is not configured.",
            used_fallback=True,
        )


class GeminiFallbackProvider(FallbackProvider):
    """
    External provider wrapper with retry/timeout control.

    Uses Gemini generateContent endpoint via API key from environment.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", retries: int = 2) -> None:
        # Normalize common .env mistakes (smart quotes, accidental wrapping quotes).
        key = (api_key or "").strip().strip("\ufeff")
        if len(key) >= 2 and key[0] == key[-1] and key[0] in "'\"":
            key = key[1:-1].strip()
        self.api_key = key
        self.model_name = (model_name or "").strip()
        self.retries = max(0, retries)

    def _candidate_models(self) -> List[str]:
        # Try requested model first, then fallbacks. gemini-2.0-flash* is not offered to new
        # API projects (404); prefer gemini-2.5-* per Google deprecations.
        extra = os.getenv("GEMINI_EXTRA_MODELS", "").strip()
        extras = [m.strip() for m in extra.split(",") if m.strip()]

        candidates: List[str] = [self.model_name]
        if self.model_name and not self.model_name.endswith("-latest"):
            candidates.append(f"{self.model_name}-latest")
        candidates.extend(
            [
                "gemini-2.5-flash",
                "gemini-2.5-flash-latest",
                "gemini-2.5-pro",
                "gemini-1.5-flash-latest",
                "gemini-1.5-pro",
            ]
        )
        candidates.extend(extras)

        seen = set()
        unique = []
        for model in candidates:
            if model and model not in seen:
                seen.add(model)
                unique.append(model)
        return unique

    @staticmethod
    def _quota_exhausted_zero(body: str) -> bool:
        """True when Google reports no free-tier allowance (billing / plan issue)."""
        if not body:
            return False
        if '"limit": 0' in body or '"limit":0' in body:
            return True
        if "free_tier" in body.lower() and "quota exceeded" in body.lower():
            return True
        return False

    def _build_url(self, api_version: str, model_name: str) -> str:
        return (
            f"https://generativelanguage.googleapis.com/{api_version}/models/"
            f"{model_name}:generateContent"
        )

    def _transcribe_once(self, raw_bytes: bytes, timeout_s: float, api_version: str, model_name: str) -> STTResult:
        b64_audio = base64.b64encode(raw_bytes).decode("utf-8")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": "Transcribe this speech audio to plain text only."},
                        {
                            "inline_data": {
                                "mime_type": "audio/webm",
                                "data": b64_audio,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0},
        }
        url = self._build_url(api_version=api_version, model_name=model_name)
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        response_json = json.loads(raw)

        text = ""
        try:
            text = (
                response_json["candidates"][0]["content"]["parts"][0].get("text", "").strip()
            )
        except Exception:
            text = ""

        if not text:
            return STTResult(
                transcription="",
                engine="gemini_fallback",
                warning="Fallback provider returned empty transcription.",
                used_fallback=True,
            )
        return STTResult(
            transcription=text,
            engine="gemini_fallback",
            confidence=None,
            used_fallback=True,
            metadata={"provider_model": model_name, "provider_api_version": api_version},
        )

    def transcribe(self, raw_bytes: bytes, timeout_s: float = 10.0) -> STTResult:
        attempts = self.retries + 1
        last_error: Optional[str] = None
        api_versions = ["v1beta", "v1"]
        model_candidates = self._candidate_models()

        for attempt in range(1, attempts + 1):
            for api_version in api_versions:
                for model_name in model_candidates:
                    try:
                        return self._transcribe_once(
                            raw_bytes,
                            timeout_s=timeout_s,
                            api_version=api_version,
                            model_name=model_name,
                        )
                    except urllib.error.HTTPError as exc:
                        body = ""
                        try:
                            body = exc.read().decode("utf-8", errors="ignore")
                        except Exception:
                            body = ""
                        last_error = f"HTTP {exc.code}: {exc.reason}. {body}".strip()
                        if exc.code == 400 and (
                            "API_KEY_INVALID" in body or "API Key not found" in body
                        ):
                            return STTResult(
                                transcription="",
                                engine="gemini_fallback",
                                error=(
                                    "Invalid Gemini API key (revoked, typo, or wrong key type). "
                                    "Create a new key in Google AI Studio, set GEMINI_API_KEY in .env "
                                    "with no quotes/spaces, restart Flask. "
                                    f"Details: {last_error}"
                                ),
                                used_fallback=True,
                            )
                        if exc.code == 429 and self._quota_exhausted_zero(body):
                            return STTResult(
                                transcription="",
                                engine="gemini_fallback",
                                error=(
                                    "Gemini API quota is 0 or exhausted for this key/project "
                                    "(link billing in Google Cloud / AI Studio, or use a new project). "
                                    f"Last response: {last_error}"
                                ),
                                used_fallback=True,
                            )
                        continue
                    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                        last_error = str(exc)
                        continue
            if attempt < attempts:
                time.sleep(0.6 * attempt)

        return STTResult(
            transcription="",
            engine="gemini_fallback",
            error=f"Fallback failed after retries: {last_error or 'unknown error'}",
            used_fallback=True,
        )

