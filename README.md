# DeepShield

DeepShield is an AI-powered deepfake defense platform with two mission modules:
- Red Team: proactive protection to reduce deepfake reuse risk.
- Blue Team: forensic deepfake detection with explainable evidence.

The project runs as a Flask web application and includes:
- A mission-style single-page UI on `/`
- An About/Vision page on `/about`
- Production-usable API routes for protection, attack simulation, and detection

## Core Capabilities
- Protect source face images with a multi-model adversarial pipeline.
- Compare attack outcomes using clean vs protected source swaps.
- Detect manipulated images using hybrid forensic signals.
- Return not only a verdict, but also confidence, per-signal scores, and readable reasons.

## Product Modules

### Red Team (Prevention)
- **Protection**: applies full media protection and returns quality + identity disruption metrics.
- **Attack Simulation**: runs comparative face swap outputs for original-source vs protected-source inputs.

### Blue Team (Detection)
- **Deep Scan**: analyzes uploaded media through a staged forensic pipeline:
1. Input normalization
2. FFT spectrum extraction
3. Frequency anomaly scoring
4. Noise consistency scoring
5. Boundary artifact scoring
6. Sharpness mismatch scoring
7. Verdict assembly

## System Design Overview

### Protection Flow
- Configurable presets are defined in `config.py`.
- The protection execution pipeline is implemented in `pipeline.py`.
- API output includes visual quality metrics (PSNR, SSIM) and identity similarity metrics (ArcFace, FaceNet).

### Swap Simulation Flow
- Implemented via `swapper.py`.
- Runs the same target swap twice: once with a clean source image and once with a protected source image.
- Returns comparative identity confidence and quality labels.

### Detection Flow
- Implemented in `detection/forensic_detector.py`.
- Uses combined forensic signals: frequency-domain irregularity, noise consistency mismatch, boundary blending artifacts, sharpness mismatch, compression blockiness cues, and color/chroma inconsistency.
- Returns `REAL`/`FAKE` verdict, confidence, fake probability, per-signal analysis, and summarized explanation with reasons.

## Tech Stack
- Python
- Flask
- PyTorch / TorchVision
- OpenCV
- NumPy
- HTML / CSS / JavaScript

## Repository Layout
```text
.
|- app.py
|- main.py
|- pipeline.py
|- config.py
|- swapper.py
|- detection/
|- models/
|- methods/
|- evaluation/
|- utils/
|- static/
|- templates/
|- docs/
`- README.md
```

## Getting Started

### Prerequisites
- Python 3.10+ recommended
- pip
- GPU optional (CUDA supported if available)

### Installation
```bash
git clone <your-repo-url>
cd <repository-root>
python -m venv venv
```

Windows (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install flask
```

macOS/Linux:
```bash
source venv/bin/activate
pip install -r requirements.txt
pip install flask
```

## Run Locally

Windows (PowerShell):
```powershell
$env:HOST="127.0.0.1"
$env:PORT="5003"
$env:UPLOAD_RETENTION_SECONDS="1800"
python .\app.py
```

macOS/Linux:
```bash
export HOST=127.0.0.1
export PORT=5003
export UPLOAD_RETENTION_SECONDS=1800
python app.py
```

Open in browser:
- `http://127.0.0.1:5003/`
- `http://127.0.0.1:5003/about`

## API Reference

### `POST /api/protect`
Form data:
- `image` (required)
- `preset` (optional, default: `balanced`)

Response fields:
- `original` (base64 image)
- `protected` (base64 image)
- `metrics.psnr`
- `metrics.ssim`
- `metrics.arcface_similarity`
- `metrics.facenet_similarity`
- `metrics.preset`
- `elapsed`

### `POST /api/swap`
Form data:
- `source_original` (required)
- `source_protected` (required)
- `target` (required)

Response fields:
- `clean_swap.image`
- `clean_swap.identity_confidence`
- `clean_swap.quality`
- `protected_swap.image`
- `protected_swap.identity_confidence`
- `protected_swap.quality`
- `elapsed`

### `POST /api/detect`
Form data:
- `image` (required)

Response fields:
- `label`
- `confidence`
- `fake_probability`
- `analysis`
- `analysis_extended`
- `explanation`
- `spectrum` (when available)
- `elapsed`

## Quick Validation
```bash
python -m py_compile app.py main.py pipeline.py swapper.py detection/forensic_detector.py
```

Health checks:
```bash
curl -I http://127.0.0.1:5003/
curl -I http://127.0.0.1:5003/about
```

Detection smoke test:
```bash
curl -F "image=@/absolute/path/to/sample.jpg" http://127.0.0.1:5003/api/detect
```

## Privacy and File Handling
- Uploaded files are saved temporarily in `uploads/`.
- Temporary upload files are cleaned after processing.
- Runtime/cache artifacts are excluded via `.gitignore`.

## Deployment Notes
- Local/dev server is started with `python app.py`.
- For production, run behind a reverse proxy + WSGI server.
- Configure `HOST`, `PORT`, and `UPLOAD_RETENTION_SECONDS` via environment variables.

## License
No license file is currently included. Add a project license before public/commercial release.
