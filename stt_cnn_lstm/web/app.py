import logging
import os

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
    payload = result.to_json()
    if result.error:
        return jsonify(payload), 500
    return jsonify(payload)


@app.route("/infer_vosk", methods=["POST"])
def infer_vosk_compat_route():
    # Backward-compatible endpoint mapped to offline mode.
    raw_bytes, error = _extract_audio_bytes()
    if error:
        return jsonify({"error": error[0]}), error[1]
    result = router.transcribe(raw_bytes, mode="offline")
    payload = result.to_json()
    if result.error:
        return jsonify(payload), 500
    return jsonify(payload)


@app.route("/infer", methods=["POST"])
def infer_compat_route():
    # Legacy endpoint now routed through hybrid mode for better robustness.
    raw_bytes, error = _extract_audio_bytes()
    if error:
        return jsonify({"error": error[0]}), error[1]
    result = router.transcribe(raw_bytes, mode=DEFAULT_MODE)
    payload = result.to_json()
    if result.error:
        return jsonify(payload), 500
    return jsonify(payload)


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(debug=False, host=host, port=port)


