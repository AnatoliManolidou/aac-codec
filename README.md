<div align="center">

# AAC Codec

**Simplified AAC Audio Encoder / Decoder in Python**

Multimedia Systems — ECE AUTH, 2025–26

---

</div>

## Overview

A from-scratch implementation of a simplified [AAC](https://en.wikipedia.org/wiki/Advanced_Audio_Coding)-like audio codec, built incrementally across three levels of increasing complexity:

| Level | Description | Lossy? |
|:-----:|-------------|:------:|
| **1** | MDCT filter bank with adaptive windowing | No |
| **2** | + Temporal Noise Shaping (TNS) | No |
| **3** | + Psychoacoustic model, quantization & Huffman coding | **Yes** |

---

## Project Structure

```
aac-codec/
├── level_1/                 # Basic MDCT codec
│   ├── demo_level_1.py
│   ├── Encoder_Decoder.py
│   ├── Filter_Bank.py
│   └── SSC.py
├── level_2/                 # + TNS
│   ├── demo_level_2.py
│   ├── Encoder_Decoder.py
│   ├── Filter_Bank.py
│   ├── SSC.py
│   └── TNS.py
├── level_3/                 # Full AAC-like codec
│   ├── demo_level_3.py
│   ├── Encoder_Decoder.py
│   ├── Filter_Bank.py
│   ├── Psychoacoustic_Model.py
│   ├── Quantization.py
│   ├── SSC.py
│   └── TNS.py
├── Material/                # Shared data & utilities
│   ├── huff_utils.py
│   ├── huffCodebooks.mat
│   ├── TableB219.mat
│   └── LicorDeCalandraca.wav
├── Output/                  # Reconstructed audio & encoded data
│   ├── encoded_seq.mat
│   ├── output_level1.wav
│   ├── output_level2.wav
│   └── output_level3.wav
├── Manolidou_Anatoli_Report.pdf
├── mm-2025-hw-v0.1.pdf
└── README.md
```

---

## Modules

### Filter Bank *(Level 1)*
Implements the **Modified Discrete Cosine Transform (MDCT)** and its inverse (IMDCT) with KBD / sinusoidal analysis–synthesis windows. Supports four frame types: OLS, LSS, ESH, LPS — including the 8×256 short-block transform for transients.

### SSC — Sequence Segmentation Control *(Level 1)*
Classifies each frame as *Only Long Sequence*, *Long Start*, *Eight Short Sequence*, or *Long Stop* based on a high-pass filtered energy-attack detection on the next frame. Combines per-channel decisions via a lookup table.

### TNS — Temporal Noise Shaping *(Level 2)*
Applies a 4th-order linear-prediction FIR filter to MDCT coefficients to shape quantization noise in the time domain so it is masked by transients. The decoder applies the corresponding IIR inverse filter.

### Psychoacoustic Model *(Level 3)*
Computes **Signal-to-Mask Ratios (SMR)** per critical band using:
- Spectral unpredictability from two-frame phase/magnitude prediction
- Spreading function in the Bark domain
- Tonality-adaptive masking (TMN = 18 dB, NMT = 6 dB)

### Quantization *(Level 3)*
Non-uniform AAC quantization with iterative per-band scale-factor optimization:

$$S = \text{sign}(X)\left\lfloor\left(|X| \cdot 2^{-a/4}\right)^{3/4} + 0.4054\right\rfloor$$

Ensures distortion per band stays below the psychoacoustic threshold.

### Huffman Coding *(Level 3)*
Auto-selects the optimal codebook (1–11) per spectral section, with ESC coding for large coefficients. Uses lookup tables from `huffCodebooks.mat`.

---

## Pipeline

```
Level 1:  Audio → SSC → MDCT ─────────────────────────────────── → IMDCT → Audio
Level 2:  Audio → SSC → MDCT → TNS ─────────────────────────── → iTNS → IMDCT → Audio
Level 3:  Audio → SSC → MDCT → TNS → Psycho → Quant → Huffman → decode → Audio
```

---

## Getting Started

### Prerequisites

| Package | Purpose |
|---------|---------|
| `numpy` | Array ops, FFT |
| `scipy` | Signal processing, linear algebra, `.mat` I/O |
| `soundfile` | WAV read / write |

```bash
pip install numpy scipy soundfile
```

### Running

Ensure a **stereo 48 kHz WAV** file named `LicorDeCalandraca.wav` exists in `Material/`, then run from the repo root:

```bash
# Level 1 — lossless MDCT round-trip
python level_1/demo_level_1.py

# Level 2 — lossless MDCT + TNS round-trip
python level_2/demo_level_2.py

# Level 3 — full lossy AAC encode/decode
python level_3/demo_level_3.py
```
Each demo writes the reconstructed audio to `Output/` and prints the **SNR (dB)**. Level 3 additionally reports **bitrate** and **compression ratio**.

---

## Results

Test signal: **LicorDeCalandraca.wav** — stereo, 48 kHz, 16-bit (original bitrate 1 536 kbps)

| Level | SNR (dB) | Noise Power | Bitrate (kbps) | Compression |
|:-----:|:--------:|:-----------:|:--------------:|:-----------:|
| 1 | 253.99 | ≈ 8.1 × 10⁻²² | 1 536 | 1 : 1 |
| 2 | 253.99 | ≈ 8.1 × 10⁻²² | 1 536 | 1 : 1 |
| 3 | **10.50** | — | **264.6** | **5.80 : 1** |

### Observations

- **Levels 1 & 2** are lossless round-trips (the only "noise" is floating-point precision). Adding TNS does not introduce any loss because the FIR analysis and IIR synthesis filters use the same quantized coefficients, making the process perfectly invertible.
- **Level 3** introduces lossy quantization guided by the psychoacoustic model. Despite the significant SNR drop, the reconstructed audio exhibits minimal perceptible degradation — barely noticeable on first listen — since the quantization noise is shaped to stay below the masking threshold of the human ear.
- The achieved bitrate of **~265 kbps** for stereo comfortably exceeds the ITU-R broadcast quality threshold, which AAC typically reaches at 128 kbps for stereo signals.

---

<div align="center">
<sub>Aristotle University of Thessaloniki — School of Electrical & Computer Engineering</sub>
</div>
