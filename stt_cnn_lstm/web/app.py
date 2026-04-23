import logging
import os
import re
from typing import Dict

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from services.stt.router import HybridSTTRouter


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# Load environment from project root explicitly so config works
# regardless of the current working directory used to start Flask.
# override=True: a stale GEMINI_API_KEY in Windows User/shell env would otherwise
# win over .env (python-dotenv default), which looks like an "invalid" key in-app.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"), override=True)
_configure_logging()
router = HybridSTTRouter.build_default()
DEFAULT_MODE = os.getenv("STT_DEFAULT_MODE", "enhanced").strip().lower()
NON_ENGLISH_WARNING = "Please try in English only. Other languages are future scope."
ENGLISH_TRANSCRIPT_RE = re.compile(r"^[A-Za-z0-9\s.,!?;:'\"()\-\[\]{}_/&%$@#*+=<>`~]*$")
WORD_RE = re.compile(r"[a-z]+")


ACCENT_RULES = [
    ("indian_english", {"yaar", "na", "only", "itself", "prepone", "outstation", "timepass", "batchmate"}),
    ("british_english", {"mate", "cheers", "bloody", "flat", "lorry", "biscuit", "holiday", "petrol"}),
    ("american_english", {"awesome", "gonna", "wanna", "guys", "apartment", "truck", "cookie", "gas"}),
]
ACCENT_COUNTRY_MAP = {
    "indian_english": "India",
    "british_english": "United Kingdom",
    "american_english": "United States",
    "neutral_english": "Unknown",
    "unknown": "Unknown",
}


def _is_english_text(text: str) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return True
    return bool(ENGLISH_TRANSCRIPT_RE.fullmatch(normalized))


def _apply_english_guard(payload: dict) -> dict:
    transcription = (payload.get("transcription") or "").strip()
    if not transcription or _is_english_text(transcription):
        return payload

    existing_warning = payload.get("warning", "").strip()
    payload["warning"] = (
        f"{existing_warning} | {NON_ENGLISH_WARNING}" if existing_warning else NON_ENGLISH_WARNING
    )
    payload["transcription"] = ""
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metadata["language_guard"] = "blocked_non_english_transcription"
    payload["metadata"] = metadata
    return payload


def _detect_experimental_accent(transcription: str) -> Dict[str, object]:
    text = (transcription or "").strip().lower()
    if not text:
        return {
            "label": "unknown",
            "country": "Unknown",
            "confidence": 0.0,
            "reason": "empty_transcription",
        }

    words = WORD_RE.findall(text)
    if not words:
        return {
            "label": "unknown",
            "country": "Unknown",
            "confidence": 0.0,
            "reason": "no_alpha_tokens",
        }

    scores: Dict[str, int] = {label: 0 for label, _ in ACCENT_RULES}
    for word in words:
        for label, vocab in ACCENT_RULES:
            if word in vocab:
                scores[label] += 1

    best_label = "neutral_english"
    best_hits = 0
    second_hits = 0
    for label, _ in ACCENT_RULES:
        hits = scores[label]
        if hits > best_hits:
            second_hits = best_hits
            best_hits = hits
            best_label = label
        elif hits > second_hits:
            second_hits = hits

    if best_hits == 0:
        return {
            "label": "neutral_english",
            "country": ACCENT_COUNTRY_MAP["neutral_english"],
            "confidence": 0.2,
            "reason": "no_marker_words",
        }

    # Confidence is intentionally conservative because this is a lexical heuristic.
    total_hits = sum(scores.values())
    margin = best_hits - second_hits
    confidence = min(0.85, 0.35 + (best_hits * 0.12) + (margin * 0.08) + (0.03 if total_hits >= 2 else 0))
    confidence = round(max(0.0, confidence), 2)
    return {
        "label": best_label,
        "country": ACCENT_COUNTRY_MAP.get(best_label, "Unknown"),
        "confidence": confidence,
        "reason": "marker_word_heuristic",
    }


def _response_from_result(result):
    payload = _apply_english_guard(result.to_json())
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    accent_input = payload.get("transcription", "")
    metadata["accent_experimental"] = _detect_experimental_accent(accent_input)
    payload["metadata"] = metadata
    if result.error:
        return jsonify(payload), 500
    return jsonify(payload)

def _extract_audio_bytes():
    if "audio" not in request.files:
        return None, ("No audio file uploaded", 400)
    data = request.files["audio"].read()
    if not data:
        return None, ("Empty audio file", 400)
    return data, None


@app.route("/")
def index():
    return render_template("index.html", default_mode=DEFAULT_MODE)


@app.route("/infer_hybrid", methods=["POST"])
def infer_hybrid_route():
    raw_bytes, error = _extract_audio_bytes()
    if error:
        return jsonify({"error": error[0]}), error[1]

    requested_mode = request.form.get("mode", DEFAULT_MODE).strip().lower()
    if requested_mode not in {"offline", "enhanced"}:
        requested_mode = DEFAULT_MODE

    result = router.transcribe(raw_bytes, mode=requested_mode)
    return _response_from_result(result)


@app.route("/infer_vosk", methods=["POST"])
def infer_vosk_compat_route():
    # Backward-compatible endpoint mapped to offline mode.
    raw_bytes, error = _extract_audio_bytes()
    if error:
        return jsonify({"error": error[0]}), error[1]
    result = router.transcribe(raw_bytes, mode="offline")
    return _response_from_result(result)


@app.route("/infer", methods=["POST"])
def infer_compat_route():
    # Legacy endpoint now routed through hybrid mode for better robustness.
    raw_bytes, error = _extract_audio_bytes()
    if error:
        return jsonify({"error": error[0]}), error[1]
    result = router.transcribe(raw_bytes, mode=DEFAULT_MODE)
    return _response_from_result(result)


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(debug=False, host=host, port=port)


