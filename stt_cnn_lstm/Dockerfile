# Speech-to-text demo (Flask). Build from this folder (stt_cnn_lstm).
# REQUIRES: checkpoints/*.pt and pretrained_models/vosk-en on disk before build
# (or mount them at run time — see README / friend instructions you send).

FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY web ./web
COPY data ./data
COPY checkpoints ./checkpoints
COPY pretrained_models ./pretrained_models
COPY training_history.json ./training_history.json

ENV PYTHONUNBUFFERED=1
# Listen on all interfaces so `docker run -p 5000:5000` works
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000

EXPOSE 5000

CMD ["python", "-m", "web.app"]
