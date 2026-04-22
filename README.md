# Robust speech recognition in noisy environments (CNN–LSTM)

VTU final-year project: **speech-to-text** with a **CNN–LSTM** acoustic model (CTC, PyTorch, CPU-friendly), a **Flask** web demo, **Vosk** for live offline decoding, and an optional **Gemini** fallback (configurable via `.env`).

## Layout

| Path | Contents |
|------|----------|
| **`stt_cnn_lstm/`** | Application: `src/`, `web/`, `services/stt/`, data, checkpoints, Docker |
| **`stt_cnn_lstm/README.md`** | Training, evaluation, demo, hybrid STT configuration |
| **`stt_cnn_lstm/PPT_CONTENT.md`** | Slide-oriented notes |

## Clone (Git LFS for checkpoints)

Weights (`.pt`) may use **Git LFS**. Install [Git LFS](https://git-lfs.com/), then:

```powershell
git clone https://github.com/BSTushar/final_year.git
cd final_year
git lfs pull
```

## Run the web demo (Windows)

From **`stt_cnn_lstm/`**:

```powershell
cd stt_cnn_lstm
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env if using Gemini fallback (optional)
python -m web.app
```

Open **http://127.0.0.1:5000/** and allow the microphone. If WebM conversion fails, install **ffmpeg**.

## Docker

From **`stt_cnn_lstm/`**:

```powershell
docker build -t stt-demo .
docker run --rm -p 5000:5000 stt-demo
```

## Team

- Abhimanyu Tiwari (1CR22IS004)  
- BS Tushar (1CR22IS035)  
- Atiksh V Jain (1CR22IS026)  

**Guide:** Shilpa Mangesh Pande, Asst. Professor, Dept. of ISE, CMRIT  
**Affiliation:** Visvesvaraya Technological University (VTU)

Academic use per your institution’s rules.
