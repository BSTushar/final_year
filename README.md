# Robust Speech Recognition in Noisy Environments (CNN–LSTM)

VTU final-year project: end-to-end **speech-to-text** using a **CNN–LSTM** acoustic model trained with **CTC loss** (PyTorch, CPU-friendly), plus a **Flask** web demo and optional **Vosk** path for live recognition.

## Repository layout

| Path | Contents |
|------|----------|
| **`stt_cnn_lstm/`** | Main application: source code, data, checkpoints, web UI, Docker |
| **`PROJECT_WRITEUP.md`** | Short write-up (title, abstract, methodology, results, conclusion) |
| **`remember.txt`** | Minimal commands for demo day |
| **`VTU_Report_Sections/`** | Report PDFs and related materials |
| **`PPT.pdf`**, **`PPT.pptx`** | Presentation slides |

Full technical documentation, training commands, and viva notes: **`stt_cnn_lstm/README.md`**.

## Clone (Git LFS required for checkpoints)

Model weights (`.pt`) are stored with **Git LFS**. Install [Git LFS](https://git-lfs.com/) first, then:

```powershell
git clone https://github.com/BSTushar/final_year.git
cd final_year
git lfs pull
```

## Run the web demo (Windows)

From `stt_cnn_lstm/`:

```powershell
cd stt_cnn_lstm
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m web.app
```

Open **http://127.0.0.1:5000/** — allow the microphone when prompted.

**Note:** The UI records audio in the browser and posts it to the server. **ffmpeg** helps convert WebM; install it if conversion errors appear. See **`remember.txt`** for a compact checklist.

## Docker

From `stt_cnn_lstm/`:

```powershell
docker build -t stt-demo .
docker run --rm -p 5000:5000 stt-demo
```

Then open **http://localhost:5000/**.

## Train / evaluate / plots

See **`stt_cnn_lstm/README.md`** for:

- `python -m src.train` (manifests under `data/manifests/`)
- `python -m src.evaluate`
- `python -m src.plots`

## Team

- Abhimanyu Tiwari (1CR22IS004)  
- BS Tushar (1CR22IS035)  
- Atiksh V Jain (1CR22IS026)  

**Guide:** Shilpa Mangesh Pande, Asst. Professor, Dept. of ISE, CMRIT  
**Affiliation:** Visvesvaraya Technological University (VTU)

## License / use

Academic project; use and attribution per your institution’s rules.
