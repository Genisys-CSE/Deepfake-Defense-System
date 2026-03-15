# DeepShield — Anti-Deepfake Protection System v2.0

Protects images against deepfake creation by embedding invisible, multi-layered
perturbations that disrupt face-swap and face-reenactment pipelines.

## How It Works

DeepShield uses a **multi-model ensemble adversarial attack** combined with
**DCT frequency perturbation** and **gram-matrix texture disruption** to break
the feature extraction and rendering stages of deepfake generators — all while
keeping the perturbations invisible to the human eye.

### Attack Vectors

| Layer | What It Disrupts | How |
|-------|-----------------|-----|
| **Identity** | FaceNet embeddings (×2 models) | Cosine similarity minimisation |
| **Features** | VGG19 multi-layer + ResNet50 | Ensemble feature disruption |
| **Texture** | Gram matrices (style transfer) | Style statistics maximisation |
| **Frequency** | DCT coefficients (JPEG-robust) | Mid-high freq band noise |

### Robustness (EOT)

Perturbations are optimised to survive real-world transformations via
Expectation Over Transformation:
- JPEG compression (quality 70–95)
- Rotation / scaling (±5° / 0.95–1.05×)
- Colour jitter, Gaussian noise, blur

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Balanced protection (recommended for most uses)
python main.py --input photo.jpg --out protected.jpg

# Maximum protection (stronger, slight artifacts possible)
python main.py --input photo.jpg --out protected.jpg --preset maximum

# Stealth mode (near-invisible, effective against basic tools)
python main.py --input photo.jpg --out protected.jpg --preset stealth
```

### Presets

| Preset | ε (pixels) | Steps | Best For |
|--------|-----------|-------|----------|
| `maximum` | 6/255 | 300 | Maximum deepfake resistance |
| `balanced` | 4/255 | 200 | Social media uploads, profile photos |
| `stealth` | 2/255 | 150 | Professional photos, minimal artifacts |

### Advanced Options

```bash
# Override specific parameters
python main.py --input photo.jpg --out protected.jpg --steps 300 --epsilon 0.02

# Disable specific methods
python main.py --input photo.jpg --out protected.jpg --no-freq
python main.py --input photo.jpg --out protected.jpg --no-texture

# Compare two images (e.g. original vs deepfake)
python main.py --eval-only --input original.jpg --compare-img deepfake.jpg
```

## Project Structure

```
├── main.py                     # CLI entry point
├── config.py                   # Configuration & presets
├── pipeline.py                 # Protection pipeline orchestrator
├── models/
│   ├── loader.py               # Lazy model loading + GPU management
│   └── feature_extractor.py    # Multi-layer VGG19 / ResNet50 features
├── methods/
│   ├── adversarial.py          # Multi-model ensemble PGD+EOT attack
│   ├── frequency.py            # DCT frequency perturbation
│   ├── texture.py              # Gram-matrix texture disruption
│   └── transforms.py           # EOT augmentations
├── evaluation/
│   └── metrics.py              # Comprehensive evaluation metrics
└── utils/
    ├── face.py                 # Face detection & soft-mask blending
    └── image.py                # Image I/O utilities
```

## Evaluation Metrics

DeepShield reports protection effectiveness across multiple dimensions:

- **Identity cosine similarity** — Lower = identity is more disrupted
- **Feature cosine similarity** — Lower = features are more disrupted (per-layer)
- **LPIPS** — Lower = less visible perturbation
- **SSIM** — Higher = more visually similar to original
- **PSNR** — Higher = less pixel-level noise

## Requirements

- Python 3.8+
- NVIDIA GPU recommended (CUDA)
- ~2 GB VRAM (works on RTX 3050 and above)
