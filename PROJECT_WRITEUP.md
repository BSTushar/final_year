# PROJECT WRITE-UP

## TITLE

**Robust Speech Recognition in Noisy Environments Using Hybrid CNN-LSTM Architecture**

---

## ABSTRACT

This project implements a deep learning-based speech-to-text system designed to handle noisy environments. The system uses Log-Mel Spectrogram features (80 mel bins) to represent speech signals, effectively capturing perceptually significant spectral characteristics. 

Convolutional Neural Network (CNN) layers extract robust local patterns from spectrograms, while bidirectional Long Short-Term Memory (LSTM) layers learn temporal relationships across speech sequences. The model is trained on a custom dataset of approximately 1000 voice samples with data augmentation through noise injection and SpecAugment techniques. Training uses Connectionist Temporal Classification (CTC) loss, enabling end-to-end learning without explicit alignment between audio frames and text labels.

Experimental evaluation confirms that the CNN-LSTM framework delivers enhanced transcription performance with improved tolerance to noise compared to conventional baseline models. System effectiveness is assessed using Word Error Rate (WER), Character Error Rate (CER), and recognition accuracy across different noise levels. The results validate that the proposed method achieves a favorable balance between recognition accuracy, noise robustness, and computational efficiency, demonstrating potential for deployment in real-world speech recognition applications.

---

## METHODOLOGY

### System Overview

**Dataset and Data Pipeline:**
- Custom dataset comprising approximately 1000 voice samples stored in `data/raw/` directory
- CSV manifest files in `data/manifests/` containing audio paths and corresponding text transcripts
- Audio preprocessing: resampling to 16 kHz, stereo to mono conversion, silence removal, amplitude normalization
- Log-Mel Spectrogram feature extraction using 80 mel bins with window size of 400 samples and frame shift of 160 samples

**Model Architecture:**
- **CNN Encoder:** Two convolutional layers with 64 filters each (3×3 kernel, padding=1)
- **Bidirectional LSTM:** 2 layers, 256 hidden units per direction, dropout 0.3
- **Output Layer:** Linear projection from 512 dimensions (256×2) to 29 dimensions (vocabulary size)
- **CTC Decoding:** Greedy decoding during training, beam search during validation

**Training Strategy:**
- Optimizer: Adam with learning rate 0.001
- Loss Function: CTC Loss with blank token at index 0
- Regularization: Dropout 0.3, gradient clipping (max_norm=5.0)
- Early stopping with patience of 7 epochs
- Batch size: 8 (CPU-friendly)
- Training epochs: 20

**Data Flow:**
1. Audio Input (16 kHz mono WAV) → Preprocessing
2. Log-Mel Spectrogram (80 bins) → Feature Extraction
3. CNN Encoder → Local Pattern Extraction
4. Reshape to Sequence → Temporal Format
5. Bidirectional LSTM → Temporal Modeling
6. Linear Layer → Character Probabilities
7. CTC Decoding → Text Output

---

## RESULTS

### Performance Metrics

**Training Evolution:**
- **Initial Stages:** WER ≈ 1.0, CER very high, blank ratio ≈ 95%
- **Later Stages:** WER improved to ~0.53, blank ratio decreased to ~82%
- **Training Accuracy:** Improved from near 0% to measurable recognition rates

**Noise Robustness:**
- **Clean Speech:** System achieves low WER and CER, matching reference transcripts well
- **Moderate Noise (20 dB SNR):** Performance very close to clean conditions
- **Higher Noise (10 dB SNR):** Some degradation but outperforms baseline DNN and CNN models
- **Very High Noise (5 dB SNR or below):** Performance degrades further (area for future improvement)

**Comparative Analysis:**
- Hybrid CNN-LSTM architecture outperforms traditional DNN-based models
- Outperforms pure CNN models across different noise levels
- Baseline models show significant performance degradation in noisy conditions
- Our system maintains relatively stable performance

**Key Achievements:**
- End-to-end training from scratch (no pretrained models)
- CPU-only training and deployment capability
- Real-time web interface demonstration
- Validated with WER/CER metrics at different noise levels

---

## CONCLUSION

This project successfully implements a complete deep learning-based speech-to-text system designed, implemented, and evaluated in a VTU academic setting. The system combines Log-Mel Spectrogram features (80 mel bins), CNN encoder with two convolutional layers (64 channels each), bidirectional LSTM with 2 layers and 256 hidden units, and CTC loss for end-to-end training.

The approach utilizes SpecAugment and speed perturbation for data augmentation to improve robustness. The system demonstrates improved performance in various acoustic conditions and can be trained on normal CPUs, making it suitable for practical deployment scenarios.

The hybrid CNN-LSTM architecture achieves a favorable balance between recognition accuracy, improved noise tolerance, and computational efficiency. The approach is validated both quantitatively through WER and CER metrics at different noise levels and qualitatively through a live web-based microphone demo.

The results show promise for deployment in real-world speech recognition applications, particularly in resource-constrained environments where GPU acceleration is not available. The system demonstrates that effective speech recognition can be achieved with careful architecture design, appropriate feature extraction, and robust training strategies, even with limited computational resources.

---

**Team Members:**
- Abhimanyu Tiwari (1CR22IS004)
- BS Tushar (1CR22IS035)
- Atiksh V Jain (1CR22IS026)

**Guide:** Shilpa Mangesh Pande, Asst. Professor, Department of ISE, CMRIT  
**Institution:** CMR Institute of Technology, Bengaluru  
**Affiliation:** Visvesvaraya Technological University

