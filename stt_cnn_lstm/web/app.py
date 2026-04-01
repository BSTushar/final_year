import io
import json
import os
import tempfile

from flask import Flask, jsonify, render_template, request
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment

from src.features import LogMelFeatureExtractor, SAMPLE_RATE
from src.infer import load_model
from src.decode import ctc_decode
from src.utils import indices_to_text

try:
    from vosk import Model as VoskModel, KaldiRecognizer

    VOSK_AVAILABLE = True
except Exception:
    VOSK_AVAILABLE = False
    VoskModel = None
    KaldiRecognizer = None


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)


MODEL_PATH = os.path.join("checkpoints", "best_by_wer.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join("checkpoints", "last_epoch.pt")

model = None
char2idx = None
idx2char = None
extractor = LogMelFeatureExtractor()

VOSK_MODEL_DIR = os.path.join("pretrained_models", "vosk-en")
vosk_model = None
VOSK_SAMPLE_RATE = 16000


def load_global_model():
    global model, char2idx, idx2char
    if model is None:
        if os.path.exists(MODEL_PATH):
            m, c2i, i2c = load_model(MODEL_PATH)
            model = m
            char2idx = c2i
            idx2char = i2c
            model.to(torch.device("cpu"))
            model.eval()
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}. Please train the model first.")


def load_vosk_model():
    global vosk_model
    if not VOSK_AVAILABLE:
        raise RuntimeError(
            "Vosk is not installed. Install it with 'pip install vosk' to use the pretrained recognizer."
        )

    if vosk_model is None:
        if not os.path.isdir(VOSK_MODEL_DIR):
            raise FileNotFoundError(
                f"Vosk model directory not found at '{VOSK_MODEL_DIR}'. "
                "Download a small English model from the Vosk website and extract it there."
            )
        vosk_model = VoskModel(VOSK_MODEL_DIR)

    return vosk_model


@app.route("/")
def index():
    plots_dir = os.path.join(app.static_folder, "plots")
    diagrams_dir = os.path.join(app.static_folder, "diagrams")
    plot_files = []
    diagram_files = []
    if os.path.exists(plots_dir):
        for f in os.listdir(plots_dir):
            if f.lower().endswith(".png"):
                plot_files.append(f"plots/{f}")
    if os.path.exists(diagrams_dir):
        for f in os.listdir(diagrams_dir):
            if f.lower().endswith(".png"):
                diagram_files.append(f"diagrams/{f}")
    return render_template(
        "index.html",
        plots=sorted(plot_files),
        diagrams=sorted(diagram_files),
        vosk_available=VOSK_AVAILABLE,
    )


@app.route("/infer", methods=["POST"])
def infer_route():
    try:
        load_global_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    
    if model is None:
        return jsonify({"error": "Model checkpoint not found. Train the model first."}), 500

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files["audio"]
    data = file.read()
    if not data:
        return jsonify({"error": "Empty audio file"}), 400

    try:
        wav, sr = torchaudio.load(io.BytesIO(data), format="wav")
    except Exception:
        try:
            tmp_webm_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_webm:
                    tmp_webm.write(data)
                    tmp_webm_path = tmp_webm.name
                
                audio = AudioSegment.from_file(tmp_webm_path, format="webm")
                audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
                
                wav_buffer = io.BytesIO()
                audio.export(wav_buffer, format="wav")
                wav_buffer.seek(0)
                
                wav, sr = torchaudio.load(wav_buffer, format="wav")
                
                if tmp_webm_path and os.path.exists(tmp_webm_path):
                    try:
                        os.unlink(tmp_webm_path)
                    except Exception:
                        pass
            except Exception as conv_err:
                if tmp_webm_path and os.path.exists(tmp_webm_path):
                    try:
                        os.unlink(tmp_webm_path)
                    except Exception:
                        pass
                raise conv_err
        except Exception as e:
            return jsonify({"error": f"Failed to convert audio format. Error: {str(e)}. Make sure ffmpeg is installed or try a different browser."}), 400

    if sr != SAMPLE_RATE:
        try:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
        except Exception as e:
            return jsonify({"error": f"Failed to resample audio: {e}"}), 400

    if wav.dim() > 1 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav_mono = wav.squeeze(0) if wav.dim() > 1 else wav
    
    if wav_mono.numel() < 400:
        return jsonify({"error": "Audio too short. Please record at least 0.5 seconds."}), 400
    
    feats = extractor(wav_mono)
    if feats.shape[-1] < 10:
        return jsonify({"error": "Audio too short after processing."}), 400
    
    feats = feats.unsqueeze(0)
    with torch.no_grad():
        log_probs = model(feats)
        decoded = ctc_decode(log_probs, beam_width=5)[0]  # Use beam search for better accuracy
        text = indices_to_text(decoded, idx2char)
        
        if not text or text.strip() == "":
            max_prob_idx = log_probs.argmax(dim=-1).cpu().numpy()
            non_blank_count = (max_prob_idx != 0).sum()
            return jsonify({
                "transcription": "",
                "warning": f"Empty transcription. Model output: {len(decoded)} non-blank tokens. This usually means the model needs more training. Try training for more epochs or check if the checkpoint is from a trained model.",
                "debug_info": {
                    "decoded_length": len(decoded),
                    "non_blank_frames": int(non_blank_count),
                    "total_frames": log_probs.shape[0]
                }
            })

    return jsonify({"transcription": text, "engine": "cnn_lstm"})


@app.route("/infer_vosk", methods=["POST"])
def infer_vosk_route():
    try:
        model_dir = load_vosk_model()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files["audio"]
    data = file.read()
    if not data:
        return jsonify({"error": "Empty audio file"}), 400

    try:
        wav, sr = torchaudio.load(io.BytesIO(data), format="wav")
    except Exception:
        try:
            tmp_webm_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_webm:
                    tmp_webm.write(data)
                    tmp_webm_path = tmp_webm.name

                audio = AudioSegment.from_file(tmp_webm_path, format="webm")
                audio = audio.set_frame_rate(VOSK_SAMPLE_RATE).set_channels(1)

                wav_buffer = io.BytesIO()
                audio.export(wav_buffer, format="wav")
                wav_buffer.seek(0)

                wav, sr = torchaudio.load(wav_buffer, format="wav")

                if tmp_webm_path and os.path.exists(tmp_webm_path):
                    try:
                        os.unlink(tmp_webm_path)
                    except Exception:
                        pass
            except Exception as conv_err:
                if tmp_webm_path and os.path.exists(tmp_webm_path):
                    try:
                        os.unlink(tmp_webm_path)
                    except Exception:
                        pass
                raise conv_err
        except Exception as e:
            return jsonify(
                {
                    "error": f"Failed to convert audio format. Error: {str(e)}. "
                    "Make sure ffmpeg is installed or try a different browser."
                }
            ), 400

    if sr != VOSK_SAMPLE_RATE:
        try:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=VOSK_SAMPLE_RATE)
        except Exception as e:
            return jsonify({"error": f"Failed to resample audio: {e}"}), 400

    if wav.dim() > 1 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav_mono = wav.squeeze(0) if wav.dim() > 1 else wav

    if wav_mono.numel() < 400:
        return jsonify({"error": "Audio too short. Please record at least 0.5 seconds."}), 400

    wav_np = wav_mono.detach().cpu().numpy()
    wav_np = np.clip(wav_np, -1.0, 1.0).astype("float32")
    pcm16 = (wav_np * 32767).astype("int16").tobytes()

    rms = float(np.sqrt(np.mean(wav_np**2)) + 1e-8)
    noise_db = 20.0 * float(np.log10(rms))
    duration_sec = float(wav_np.shape[0] / float(VOSK_SAMPLE_RATE))

    recognizer = KaldiRecognizer(model_dir, VOSK_SAMPLE_RATE)
    if not recognizer.AcceptWaveform(pcm16):
        result = recognizer.FinalResult()
    else:
        result = recognizer.FinalResult()

    try:
        result_json = json.loads(result)
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
        return jsonify(
            {
                "transcription": "",
                "warning": "Pretrained recognizer returned empty text. Try speaking a longer or clearer sentence.",
                "engine": "vosk",
                "noise_db": noise_db,
                "duration_sec": duration_sec,
                "confidence": avg_conf,
            }
        )

    return jsonify(
        {
            "transcription": text,
            "engine": "vosk",
            "noise_db": noise_db,
            "duration_sec": duration_sec,
            "confidence": avg_conf,
        }
    )


if __name__ == "__main__":
    os.makedirs(os.path.join("web", "static", "plots"), exist_ok=True)
    os.makedirs(os.path.join("web", "static", "diagrams"), exist_ok=True)
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(debug=False, host=host, port=port)


