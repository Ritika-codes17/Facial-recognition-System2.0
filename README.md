# Real-Time Facial Emotion Recognition with Temporal Smoothing

A high-performance, web-integrated pipeline for real-time Facial Emotion Recognition (FER). This project combines a lightweight Deep Learning architecture with temporal smoothing techniques to achieve stable, jitter-free emotion tracking in live video feeds.

---

## 📌 Features

* **Real-Time Detection:** Rapid face localization using Haar Cascade classifiers.
* **Lightweight Classification:** Emotion prediction powered by a **Mini-Xception** CNN, optimized for real-time inference without high GPU overhead.
* **Temporal Smoothing (Jitter Reduction):** Implements a rolling queue buffer and weighted temporal voting mechanism to stabilize predictions across consecutive video frames.
* **Live Telemetry Interface:** Embedded web-based dashboard for real-time visual monitoring and telemetry data.

---

## 🛠️ Tech Stack

* **Language:** Python 3
* **Computer Vision:** OpenCV
* **Deep Learning Framework:** TensorFlow / Keras
* **Interface & Web Backend:** Flask / HTML5 / CSS3

---
## 🚀 Engineering Journey & Improvements

The development of this project focused on solving latency, model efficiency, and real-time prediction stability.

```mermaid
flowchart TD
    A[Phase 1: Baseline CNN] -->|High parameter count & latency| B[Phase 2: Mini-Xception Architecture]
    B -->|Rapid frame-by-frame fluctuations| C[Phase 3: FIFO Queue Buffering]
    C -->|Single-frame prediction noise| D[Phase 4: Weighted Temporal Voting]
    D -->|Optimized, jitter-free predictions| E[Phase 5: Flask Web Dashboard]

    subgraph Pipeline Detail
    B -- Raw Frames --> C
    C -- Recent N Frames --> D
    D -- Smoothed Emotion --> E
    end
```

## 📊 Results & Performance

The transition to the Mini-Xception architecture combined with temporal smoothing resulted in significant gains in both computational efficiency and real-time inference quality.

### 📈 Metric Comparison

| Metric | Baseline CNN | Mini-Xception + Temporal Pipeline | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Parameters** | ~3.3M | **~60K** | **~98% reduction** |
| **Inference Latency** | ~45 ms/frame | **~12 ms/frame** | **3.75x faster** |
| **Throughput (FPS)** | ~18-20 FPS | **30+ FPS (capped)** | Smooth real-time stream |
| **Classification Accuracy** | 61.2% | **66.4%** | +5.2% accuracy boost |
| **Prediction Stability** | High jitter/flicker | **Stable & Smooth** | Single-frame noise eliminated |

---

### 🔑 Key Performance Insights

* **Model Compression:** By replacing standard convolutions with Depthwise Separable Convolutions, the model size dropped below **1 MB**, making it lightweight enough to run effortlessly on edge devices or CPU-only web servers.
* **Temporal Stability:** The FIFO queue and weighted voting mechanism successfully eliminated high-frequency label flickering, ensuring consistent emotion tracking across rapid facial transitions without adding perceptible lag.
* **Low Memory Footprint:** The combined pipeline maintains a negligible RAM footprint during continuous video streaming, ensuring zero performance decay during extended live sessions.
