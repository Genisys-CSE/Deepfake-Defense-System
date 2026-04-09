# DeepShield

DeepShield is a mission-style AI deepfake defense system with two active modules:
- Red Team: prevention (`/api/protect`) + attack simulation (`/api/swap`)
- Blue Team: forensic deepfake detection (`/api/detect`)

The frontend is a single-page mission flow on `/`, with a dedicated About page at `/about`.

## Final UI/Flow Status
- Cinematic dark mission UI is active (landing, Red Team, Blue Team)
- Red Team has progress feedback for protection and attack simulation
- Blue Team has staged pipeline animation + request progress + forensic reasoning output
- About page opens in a separate route and displays project vision + README content
- Dead nav actions were removed to avoid buttons going nowhere

## Backend Status
- Existing API contracts are unchanged:
  - `POST /api/protect`
  - `POST /api/swap`
  - `POST /api/detect`
- Detection now uses the stronger forensic detector in:
  - `detection/forensic_detector.py`
- Upload temp files are auto-cleaned after requests

## Setup (D Drive Only)
```powershell
cd D:\try_one
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```powershell
cd D:\try_one
.\venv\Scripts\Activate.ps1
$env:HOST="127.0.0.1"
$env:PORT="5003"
$env:UPLOAD_RETENTION_SECONDS="1800"
python .\app.py
```

Open:
- `http://127.0.0.1:5003` (main app)
- `http://127.0.0.1:5003/about` (about page)

## Quick Validation Commands
1. Syntax check:
```powershell
cd D:\try_one
python -m py_compile app.py main.py pipeline.py swapper.py detection\forensic_detector.py
```

2. Start server:
```powershell
cd D:\try_one
.\venv\Scripts\Activate.ps1
$env:PORT="5003"
python .\app.py
```

3. Health check in second terminal:
```powershell
Invoke-WebRequest http://127.0.0.1:5003 | Select-Object -ExpandProperty StatusCode
Invoke-WebRequest http://127.0.0.1:5003/about | Select-Object -ExpandProperty StatusCode
```

4. API smoke check (D-drive image path):
```powershell
curl.exe -s -F "image=@D:\try_one\test_images\sample.jpg" http://127.0.0.1:5003/api/detect
```

Expected JSON keys include:
- `label`
- `confidence`
- `fake_probability`
- `analysis`
- `explanation`
- `elapsed`

## Git Push Notes
- `.gitignore` already excludes runtime and heavy local artifacts (`venv/`, `model_cache/`, `uploads/*`, caches, logs).
- Keep `uploads/.gitkeep` so folder structure remains stable.
