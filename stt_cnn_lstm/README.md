## VTU_FINAL_YEAR_FIN – Speech to Text using CNN–LSTM in Noisy Environments

This repository implements a complete, CPU-only Speech-to-Text (ASR) system using a CNN–LSTM acoustic model trained from scratch with CTC loss, plus a Flask web interface for microphone-based demo and automatic generation of plots and diagrams. The focus is on converting speech to text in noisy environments, understanding how WER, CER and blank ratio evolve during training, and providing all artifacts needed for VTU internal/external viva, exhibition, and technical questioning.

The system is:

- **End-to-end and auditable**: full pipeline from raw audio and manifests to training, evaluation, plots, and live demo.
- **Trained from scratch**: the core CNN–LSTM model is not pretrained; it learns directly from the provided dataset.
- **VTU-friendly**: generates figures and diagrams useful for the project report and viva.
- **Windows + CPU ready**: no GPU, no transformers, no wav2vec2, no HuggingFace.

---

## 1. Quick Project Overview (What We Solved)

The project addresses the problem of converting spoken English audio into text when recordings contain background noise (fan noise, chatter, environment sounds). Speech-to-Text in noisy conditions is challenging because noise overlaps with speech frequencies, distorts phoneme boundaries, and makes alignment between variable-length audio and text difficult. We use a CNN–LSTM model on log-Mel spectrogram features, trained with Connectionist Temporal Classification (CTC) loss so that the model can learn frame-to-character mappings without explicit alignments. Performance is tracked through Word Error Rate (WER), Character Error Rate (CER), and blank ratio, and the system is exposed via a browser-based demo.

---

## 2. Who Explains What (Team Roles & Coverage Map)

This section divides the explanations for viva/demo among the three members, and shows which README sections each member should focus on.

### Tushar – Lead / System & ML Owner

Tushar focuses on:

- **High-level system and model**:
  - Section 1: Quick Project Overview
  - Section 3: System & Code Overview
  - Section 6: Model Architecture & Data Flow
- **Training and modeling decisions**:
  - Section 7: Training Strategy & Checkpointing
  - Section 8: Evaluation, Metrics & Observed Results (technical interpretation)
- **Why CNN–LSTM + CTC**:
  - Section 4: Core Mathematical Concepts (CNN, LSTM, CTC, blank)
- **Technical comparison and future work**:
  - Section 12: Common Examiner Questions & Suggested Answers (architecture-focused questions)
  - Section 14: Future Enhancements

Tushar explains the overall architecture, why CNN + LSTM was chosen instead of Transformers, how CTC works conceptually, how WER/CER/blank behave across epochs, and why training from scratch is academically significant.

### Abhimanyu – Data & Pipeline Owner

Abhimanyu focuses on:

- **Data and manifests**:
  - Section 5: Dataset & Manifests
- **Code pipeline and structure**:
  - Section 3: System & Code Overview (with emphasis on `data/`, `src/dataset.py`, `src/features.py`)
  - Section 6: Model Architecture & Data Flow (data flow part: audio → features → model input)
- **Preprocessing and robustness**:
  - Section 7: Training Strategy & Checkpointing (data loading, skipping bad samples)
  - Section 9: Plot & Diagram Generation (where dataset statistics come from)

Abhimanyu explains how the dataset is organized, how `train.csv` and `val.csv` link audio to text, how noisy data is handled, what preprocessing is applied (log-Mel, normalization), and why dataset size and quality affect WER and CER.

### Atiksh – Evaluation & Demo Owner

Atiksh focuses on:

- **Metrics and interpretation**:
  - Section 4: Core Mathematical Concepts (WER, CER, blank conceptually)
  - Section 8: Evaluation, Metrics & Observed Results
  - Section 11: Demo Script (What To Say Live)
- **Live demo and examiner interaction**:
  - Section 10: Web Demo (How to Run & What to Show)
  - Section 12: Common Examiner Questions & Suggested Answers (demo and metric-focused questions)
  - Section 13: Limitations

Atiksh explains how WER and CER are computed, what blank% means in CTC, how these metrics evolved during training (e.g., WER from ~1.0 to ~0.53, blank from ~95% to ~82%), how to interpret imperfect outputs in the live demo, and what limitations and future improvements are realistic.

Each member is prepared to answer cross-questions if required.

---

## 3. System & Code Overview (Project Structure)

The project is organized as:

```text
stt_cnn_lstm/
├── data/
│   ├── raw/                 # WAV files (16 kHz, mono)
│   └── manifests/
│       ├── train.csv        # training manifest
│       └── val.csv          # validation manifest
├── src/
│   ├── features.py          # Log-Mel extraction (single source of truth)
│   ├── dataset.py           # Dataset + DataLoader using manifests
│   ├── model.py             # CNN–LSTM–CTC acoustic model
│   ├── decode.py            # Greedy CTC decoding
│   ├── utils.py             # text processing, metrics (WER/CER), helpers
│   ├── train.py             # training loop, early stopping, checkpointing
│   ├── evaluate.py          # standalone evaluation on CSV (WER, CER)
│   ├── infer.py             # offline inference on WAV files
│   ├── plots.py             # automatic plot + diagram generation
├── web/
│   ├── app.py               # Flask backend + Vosk integration for robust demo
│   ├── templates/
│   │   └── index.html       # modern single-page UI with mic, plots, history
│   └── static/
│       ├── plots/           # auto-generated training/result figures (PNG)
│       └── diagrams/        # auto-generated system diagrams (PNG)
├── checkpoints/
│   ├── last_epoch.pt        # latest training checkpoint
│   └── best_by_wer.pt       # best checkpoint by validation WER
├── requirements.txt         # Python dependencies (includes vosk, Flask, torch)
└── README.md                # this file (viva + usage)
```

Key points:

- There is **one feature pipeline** (`src/features.py`) used consistently by training, evaluation, offline inference, and the web backend.
- Training and evaluation use **CSV manifests** to define the dataset, not hard-coded file lists.
- The **Flask web app** provides a clean demo interface that reuses the same feature extraction and decoding logic; it also integrates a pretrained Vosk recognizer purely for robust demo, without changing the fact that the CNN–LSTM model is trained from scratch.

---

## 4. Core Mathematical Concepts (Viva-Oriented, No Equations)

This section is meant for oral explanation, not derivations.

- **CNN (Convolutional Neural Network)**  
  The input is a log-Mel spectrogram (time × frequency). A CNN applies small learnable filters that slide over this 2D input and compute weighted sums, followed by nonlinear activations. Each filter responds to specific local patterns (for example, formant-like structures or onsets), and stacking layers builds more abstract features. In our ASR pipeline, the CNN reduces noise and extracts robust local time–frequency features before passing them to the LSTM.

- **LSTM (Long Short-Term Memory)**  
  The CNN output is reshaped into a sequence over time and fed into a bidirectional LSTM. The LSTM processes one time step at a time while maintaining an internal memory. Its gates decide what information to keep, forget, or output. This allows the model to capture long-term dependencies in speech (e.g., how earlier phonemes influence later ones), which simple feed-forward networks cannot do effectively.

- **CTC Loss (Connectionist Temporal Classification)**  
  In speech, we do not have a precise label for every individual frame. CTC solves this by allowing the model to emit a sequence of characters plus a special blank token. Many different frame-level sequences can correspond to the same text; CTC sums their probabilities and encourages the model to choose any valid alignment that matches the target transcription. This avoids the need for manual alignment between audio frames and characters.

- **Blank Token and Blank Ratio**  
  The blank token means “no character emitted at this frame.” At the start of training, the model is unsure and tends to output blank for most frames, leading to a very high blank ratio (~95%). As the model learns, it emits more meaningful characters and fewer blanks; we observed blank ratio reducing to around ~82%, which is a positive sign.

- **Why Alignment is Difficult in Speech**  
  The duration of phonemes varies by speaker and context, and co-articulation blends sounds together. Background noise further masks boundaries. Because of this, it is impractical to define exactly which frame corresponds to which character. CTC handles this automatically, which is why it is widely used in end-to-end ASR.

---

## 5. Dataset & Manifests

Place audio and manifests as follows:

1. Put **16 kHz mono WAV** files into `data/raw/`.
2. Create CSV manifest files in `data/manifests/`:
   - `train.csv`
   - `val.csv`

Each CSV must have two columns with headers:

```text
path,text
data/raw/example1.wav,hello how are you
data/raw/example2.wav,i am fine thank you
```

Notes:

- `path` is the relative path from the project root to the WAV file.
- `text` is the reference transcription, typically lowercase with spaces and basic punctuation.
- During loading:
  - Corrupt or too-short audio files are **skipped safely** (no training crash).
  - Text is converted to character sequences; index `0` is reserved for the CTC blank token.
- The **size and diversity** of `train.csv` strongly influence WER and CER. With limited audio hours and speaker variety, the model cannot match large commercial systems, but still shows clear learning behavior.

Abhimanyu should be able to walk through the structure of the manifests, how they are loaded by `src/dataset.py`, and why manifests are more flexible than hard-coded filenames.

---

## 6. Model Architecture & Data Flow (Step-by-Step)

The full ASR pipeline for the CNN–LSTM model is:

1. **Audio Input (.wav)**  
   - Input: mono, 16 kHz WAV file.  
   - Source: `data/raw/*.wav` for training/evaluation; microphone recordings for the web demo.

2. **Feature Extraction (Log-Mel Spectrogram)**  
   - Implemented in `src/features.py`.  
   - Steps:
     - Read waveform and normalize.
     - Compute STFT with parameters like `n_fft=400`, `hop_length=160`.
     - Apply Mel filter bank to obtain Mel-spectrogram.
     - Take logarithm to get log-Mel spectrogram.
     - Optionally apply per-utterance normalization.

3. **CNN Feature Learning**  
   - Implemented in `src/model.py`.  
   - 2D convolutional layers with BatchNorm and ReLU operate over time–frequency patches.
   - Output is a compressed feature map that emphasizes robust patterns while reducing variability.

4. **LSTM Temporal Modeling**  
   - The CNN output is reshaped to a sequence over time (one feature vector per time step).
   - A 2-layer bidirectional LSTM with hidden size 256 and dropout processes the sequence.
   - The LSTM learns long-term temporal dependencies and context.

5. **CTC Prediction Layer**  
   - A linear layer projects LSTM outputs to the character vocabulary plus blank.
   - Log-softmax converts logits into log-probabilities at each time step.

6. **CTC Loss During Training**  
   - The training loop in `src/train.py` computes CTCLoss with `blank=0` and `zero_infinity=True`.
   - This loss aggregates over all valid alignments between the frame-wise outputs and target transcription.

7. **Decoding During Inference**  
   - Implemented in `src/decode.py` and reused across evaluation, offline inference, and web backend.
   - Greedy decoding picks the most probable label at each time step, collapses repeated labels, and removes blanks.
   - The resulting character sequence is mapped back to a string.

8. **Final Text Output & Web Display**  
   - The predicted text is printed in the console for CLI scripts (`evaluate.py`, `infer.py`).
   - The Flask app (`web/app.py`) returns JSON to the frontend, which displays the recognized text and logs it in a recent history table.

Tushar should connect each step here to the corresponding modules and explain why this pipeline is appropriate for noisy speech recognition.

---

## 7. Training Strategy & Checkpointing

Run training from the `stt_cnn_lstm` directory:

```bash
python -m src.train \
  --train_csv data/manifests/train.csv \
  --val_csv data/manifests/val.csv \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-3
```

Key aspects:

- **Optimizer and hyperparameters**:
  - Optimizer: Adam (`lr = 1e-3`).
      - Batch size: 8 (CPU-friendly).
- **Behavior at early epochs**:
  - WER close to ~1.0 (almost all words are incorrect).
  - Blank ratio ~95% (model mostly outputs blank).
- **Improvement over time**:
  - As epochs increase, the model learns alignments:
    - WER reduces (down to ~0.53 in later checkpoints).
    - CER also decreases.
    - Blank ratio drops from ~95% toward ~82%.
- **Checkpointing**:
  - After each epoch:
    - `checkpoints/last_epoch.pt` saves the latest state (model, optimizer, epoch, metrics).
    - `checkpoints/best_by_wer.pt` keeps the best model so far based on lowest validation WER.
  - Training history (`training_history.json`) logs:
    - training loss, training accuracy
    - validation loss
    - WER, CER, blank ratio and error distributions per epoch
- **Resuming training**:
  - The code supports resuming from `last_epoch.pt` and also using the weights from `best_by_wer.pt`.
  - Training was resumed from later epochs (for example around epoch 31 and 50), ensuring that learned weights and optimizer state are reused rather than starting from scratch.

Tushar should emphasize that this training is **from scratch**, which is more demanding but transparent and exam-friendly.

---

## 8. Evaluation, Metrics & Observed Results

To evaluate a trained model on the validation set:

```bash
python -m src.evaluate \
  --csv data/manifests/val.csv \
  --checkpoint checkpoints/best_by_wer.pt
```

The evaluation script:

- Loads the model and checkpoint.
- Computes:
  - **Word Error Rate (WER)**.
  - **Character Error Rate (CER)**.
  - Error distribution: substitutions, insertions, deletions.
- Uses the same feature extraction and greedy CTC decoding as during training.

Observed behavior during project training:

- **Initial stages**:
  - WER ≈ 1.0 (very poor recognition).
  - CER also very high.
  - Blank ratio ≈ 95%.
- **Later stages (after continued training and checkpointing)**:
  - WER improved to roughly **~0.53**.
  - Blank ratio decreased to around **~82%**.
  - CER correspondingly dropped, indicating better character-level predictions.

Interpretation:

- This trend confirms that the CNN–LSTM–CTC pipeline is learning meaningful mappings from noisy speech to text.
- Absolute performance is limited by dataset size and the fact that the model is trained from scratch on CPU, but the **direction of change** in WER, CER, and blank ratio is correct and explainable.

Atiksh should be ready to explain what each metric means and how these concrete values were used to judge progress.

---

## 9. Plot & Diagram Generation

All training history and dataset statistics are turned into figures suitable for the report and viva via:

```bash
python -m src.plots
```

This script:

- Reads `training_history.json` and dataset metadata.
- Generates **plots** into `web/static/plots/`, including (names may vary but conceptually cover):
  - Dataset distribution and basic statistics.
  - Training accuracy vs epochs.
  - Training vs validation loss curves.
  - WER and CER trends across epochs.
  - Performance comparisons and error-type distributions.
- Generates **diagrams** into `web/static/diagrams/`, such as:
  - Overall system architecture.
  - CNN–LSTM block diagram.
  - Data flow diagram and workflow chart.
  - Project folder hierarchy.

These images are:

- Saved as PNG files (easy to insert into the VTU report).
- Automatically displayed in the web UI in a **sliding carousel** with captions, for easy explanation during the demo.

Abhimanyu can connect these plots back to dataset and training behavior, while Tushar and Atiksh use them to explain architecture and results.

---

## 10. Web Demo (How to Run & What to Show)

1. **Prerequisites**
   - At least one trained checkpoint, preferably `checkpoints/best_by_wer.pt`.
   - Python dependencies installed: `pip install -r requirements.txt`.

2. **Start the Flask server**

```bash
cd stt_cnn_lstm
python -m web.app
```

3. **Open the web UI**

- In a browser, go to:

```text
http://127.0.0.1:5000/
```

4. **Using the interface**

- The page shows, in one vertical flow:
  - Microphone section with animated mic and controls.
  - Sliding carousels for **plots** and **diagrams**.
  - A **recent recognitions table** listing previous utterances and outputs.
- Steps:
  - Click “Start Recording”.
  - Speak a short English sentence.
  - Click “Stop Recording”.
  - The recorded audio (WebM/PCM) is converted to WAV, passed to the backend model/engine, and the transcription is shown.

5. **Engines**

- The **academic focus** is the CNN–LSTM model trained from scratch.
- For a more robust general-purpose live demo, the app also integrates a **pretrained Vosk ASR engine**, which is clearly separated in the code.

Atiksh should handle the live demonstration, explaining what happens behind the scenes using the data flow from Section 6.

---

## 11. Demo Script (What To Say Live)

A suggested script that matches this repository:

1. **Introduction**
   - “We have implemented a Speech-to-Text system using a CNN–LSTM model trained from scratch with CTC loss. The goal is to handle speech in noisy environments and analyze how training behavior (WER, CER, blank ratio) changes over epochs.”

2. **Demo sentence 1**
   - Speak: “Hello, welcome to today’s project demo.”
   - Explain:
     - “The browser records my voice and sends the audio to the Flask backend.”
     - “The backend converts it to a log-Mel spectrogram, passes it through the CNN–LSTM model, and uses CTC decoding to generate text.”
     - “Because our model is trained from scratch on limited data, some words may be imperfect, but it shows clear learning behavior compared to early epochs.”

3. **Demo sentence 2**
   - Speak: “This system converts noisy speech into text.”
   - Explain:
     - “We are now deliberately speaking in a normal environment with some background noise.”
     - “Our training data includes noisy examples, but strong or unseen noise can still confuse the model.”
     - “That is why we see WER around 0.5 rather than near zero, which is acceptable for a student project trained on CPU only.”

4. **Demo sentence 3**
   - Speak: “We trained this CNN–LSTM model from scratch for our VTU project.”
   - Explain:
     - “There is no pretrained acoustic model here; the CNN and LSTM weights are learned entirely from our dataset and manifests.”
     - “We also integrated a separate pretrained Vosk engine for comparison and a smoother demo, but our analysis focuses on the model we trained ourselves.”

5. **Closing**
   - “The plots and diagrams you see below summarize our dataset, training evolution, architecture, and error analysis, which we will now explain in more detail.”

---

## 12. Common Examiner Questions & Suggested Answers

- **Why did you choose CNN–LSTM instead of a simple fully connected network?**  
  CNNs are good at extracting local patterns from spectrograms (time–frequency patches), which makes them robust to small shifts and noise. LSTMs are designed to capture temporal dependencies across many frames, which is critical in speech. A fully connected network would ignore locality and temporal order, making it less suitable for ASR.

- **Why not use Transformer-based models?**  
  Transformer-based ASR models generally require large datasets and significant GPU resources to train effectively. For a final-year project with limited data and CPU-only training, CNN–LSTM with CTC is more practical and still allows us to implement and understand all key ASR components. We list Transformers as a future enhancement once more data and compute are available.

- **What is WER and how do you compute it?**  
  Word Error Rate (WER) measures the difference between the predicted and reference text at the word level. We compute the minimum number of substitutions, insertions, and deletions needed to transform the prediction into the reference, and divide by the number of words in the reference. Lower WER is better.

- **What is CER and when is it useful?**  
  Character Error Rate (CER) is the same idea applied at the character level. It is useful when word boundaries are inconsistent or when we want a more fine-grained view of performance, especially for shorter words or subword patterns.

- **Why was blank% very high initially and why is that acceptable?**  
  At the beginning of training, the model does not know when to emit characters, so it outputs the CTC blank token for most frames, giving a blank ratio near 95%. This is expected. As training progresses, the blank ratio decreases (towards ~82%), showing that the model is gaining confidence in emitting real characters.

- **Did you use any pretrained models in this project?**  
  The core CNN–LSTM acoustic model that we analyze in training and evaluation is **trained from scratch** on our dataset using CTC. No pretrained weights are used for this model. For the live demo, we additionally integrated a separate **pretrained Vosk engine** for robustness on arbitrary speech, but it is clearly separated from the experimental CNN–LSTM training pipeline.

- **What happens when the audio is very noisy?**  
  Strong or unseen noise patterns distort the spectrogram, making it harder for the model to recognize phonetic cues. The model may output more blanks or wrong characters, leading to higher WER and CER. This is a limitation of training on limited data and without advanced noise-robust techniques.

- **How is your system different from Google Speech-to-Text or other cloud APIs?**  
  Google STT is trained on thousands of hours of speech with complex architectures, large language models, and heavy compute. Our system is a transparent academic prototype that demonstrates the full ASR pipeline—from raw audio, through CNN–LSTM + CTC training, to evaluation, plots, and a web demo—using a dataset and resources realistic for a VTU project. The goal is understanding and demonstrating concepts, not competing with industrial accuracy.

---

## 13. Limitations (Honest Discussion)

- **Dataset size and diversity**  
  The amount of labeled audio is limited, with fewer speakers, accents, and noise types than large industrial datasets. This limits generalization and keeps WER higher than commercial systems.

- **Training from scratch on CPU**  
  Training from scratch is academically valuable but computationally expensive and data-hungry. On CPU and limited data, we cannot match state-of-the-art performance, but we gain full control and understanding of the model.

- **Accent and domain generalization**  
  Performance may degrade for accents, speaking styles, or vocabularies very different from the training data. This is typical for small-scale ASR projects.

- **Latency and real-time constraints**  
  On CPU, processing longer utterances can introduce noticeable delay in inference, which would need optimization or GPU deployment for strict real-time use.

---

## 14. Future Enhancements

- **Beam search decoding**  
  Replace greedy decoding with CTC beam search to consider multiple candidate sequences and integrate scoring that better resolves ambiguous frames.

- **More and better data**  
  Collect additional hours of labeled speech with more speakers, accents, and real-world noise conditions to significantly reduce WER and CER.

- **Advanced data augmentation**  
  Apply techniques such as SpecAugment (time and frequency masking on spectrograms) and more systematic noise mixing to improve robustness.

- **Transformer/Conformer-based ASR**  
  When sufficient data and compute are available, experiment with Transformer or Conformer architectures, which are closer to current state-of-the-art ASR.

- **External language model integration**  
  Add a word-level or character-level language model to the decoding stage to enforce more plausible word sequences and reduce linguistically unlikely errors.

This README is aligned with the actual implementation in the repository and is designed to be read directly during VTU internal/external viva, the project exhibition, and technical discussions with faculty and examiners.


