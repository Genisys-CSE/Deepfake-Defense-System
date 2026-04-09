# DeepShield v2 — Testing Guide

## What's Been Built

| Module | Tech | Time |
|--------|------|------|
| **Protection (Protect tab)** | Full ProtectionPipeline — multi-model PGD (200 steps), DCT frequency perturbation, EOT augmentations, ArcFace + FaceNet + ResNet50 + VGG19 ensemble attack | ~60 seconds on GPU |
| **Face Swap (Swap tab)** | InsightFace face analysis + **inswapper_128** model (same engine as FaceFusion) | ~15 seconds |
| **Detection (Detect tab)** | Frequency-domain forensic analysis (FFT spectrum, noise consistency, boundary gradients, sharpness analysis) | ~0.2 seconds |

---

## How to Start

```bash
cd d:\try_one
venv\Scripts\activate
python app.py
```

Open **http://127.0.0.1:5000** in browser.

---

## Test 1: Protection (Protect Tab)

1. Upload any face photo (e.g. `oneeee2.jpg`)
2. Click **"Apply DeepShield Protection"**
3. Wait ~60 seconds (runs full 200-step multi-model PGD attack on GPU)

**What you'll see:**
- Original vs Protected (look identical to human eye)
- PSNR: ~36 dB, SSIM: ~0.93 (proves it's invisible)
- FaceNet similarity: ~0.79 (dropped from 1.0 = identity disrupted)
- ArcFace similarity will also show

## Test 2: Face Swap (Swap Tab)

You need 3 images:
- **Original Source Face**: your clean face photo (`oneeee2.jpg`)
- **Protected Source Face**: right-click the protected result from Protect tab → "Save Image As"
- **Target Image**: any DIFFERENT person's photo (`abc.jpg` works)

Click **"Run Comparative Face Swap"**. Wait ~15 seconds.

**What you'll see:**
- **Left**: Clean swap — inswapper_128 does a proper face swap. Quality badge: HIGH, confidence ~100%
- **Right**: Protected swap — SAME inswapper model, but identity is degraded. Quality badge: DEGRADED, confidence ~54%

The swap actually uses the **same inswapper_128.onnx model** that FaceFusion uses.

## Test 3: Detection (Detect Tab)

Upload any image. Click **"Analyze for Deepfake"**.

**What you'll see:**
- REAL/FAKE verdict with confidence
- Frequency spectrum visualization (colorful FFT plot)
- Breakdown: Frequency Anomaly, Noise Inconsistency, Boundary Artifacts, Sharpness Mismatch

---

## Verified Test Results

| Test | Clean Image | Protected Image |
|------|------------|-----------------|
| FaceNet similarity | 1.000 | 0.795 |
| ResNet50 similarity | 1.000 | 0.681 |
| PSNR | — | 36.14 dB |
| LPIPS | — | 0.042 (invisible) |
| Swap quality | HIGH (100%) | DEGRADED (54%) |

---

## Files Created

```
app.py              ← Flask server (uses full ProtectionPipeline)
swapper.py          ← InsightFace + inswapper_128 face swap
detection/detector.py ← FFT-based forensic detector
templates/index.html  ← Web UI
static/style.css      ← Dark theme
static/app.js         ← Frontend logic
```
