"""
DeepShield v2 — Web Application
================================
Flask backend serving the 3-module web UI:
  Tab 1: Protect — Apply adversarial protection + show metrics
  Tab 2: Swap Test — POC face swap demonstrating protection effect
  Tab 3: Detect — Heuristic deepfake detection
"""

import os
import sys

# ── Redirect all model downloads to project-local cache (D: drive) ──
# This MUST run before any torch/torchvision imports.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_PROJECT_DIR, 'model_cache')
os.environ.setdefault('TORCH_HOME', os.path.join(_CACHE_DIR, 'torch'))
os.environ.setdefault('FACENET_CACHE', _CACHE_DIR)

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

import io
import base64
import time
import uuid
import traceback
import html
import re
import numpy as np
import cv2
import torch
from PIL import Image
from flask import Flask, render_template, request, jsonify

# Project imports
from models.loader import ModelLoader
from swapper import FaceSwapper
from detection.forensic_detector import DeepfakeDetector

app = Flask(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
UPLOAD_RETENTION_SECONDS = int(os.environ.get('UPLOAD_RETENTION_SECONDS', '1800'))
ALLOWED_UPLOAD_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# ── Global model state (loaded once) ───────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = None
swapper_instance = None
detector = DeepfakeDetector()


def get_loader():
    global loader
    if loader is None:
        loader = ModelLoader(device)
    return loader


def get_swapper():
    global swapper_instance
    if swapper_instance is None:
        swapper_instance = FaceSwapper(get_loader(), device)
    return swapper_instance


# ── Helpers ─────────────────────────────────────────────────────────────

def save_upload(file_storage):
    """Save uploaded file, return path."""
    cleanup_upload_dir(max_age_seconds=UPLOAD_RETENTION_SECONDS)
    ext = os.path.splitext(file_storage.filename or '')[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        ext = '.jpg'
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    file_storage.save(path)
    return path


def remove_file_safely(path):
    """Delete a file if it exists, ignore missing/locked file errors."""
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def cleanup_temp_paths(paths):
    """Best-effort deletion for a list of temp files."""
    for p in paths:
        remove_file_safely(p)


def cleanup_upload_dir(max_age_seconds=None):
    """
    Remove stale files from uploads.
    If max_age_seconds is None, remove all files.
    """
    now = time.time()
    for entry in os.scandir(UPLOAD_DIR):
        if not entry.is_file():
            continue
        if max_age_seconds is not None:
            try:
                age = now - entry.stat().st_mtime
            except OSError:
                continue
            if age < max_age_seconds:
                continue
        remove_file_safely(entry.path)


def markdown_to_basic_html(markdown_text):
    """
    Lightweight markdown-to-HTML renderer for the about page.
    Supports headings, paragraphs, lists, and fenced code blocks.
    """
    lines = markdown_text.replace('\r\n', '\n').split('\n')
    chunks = []
    in_code = False
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            chunks.append("</ul>")
            in_ul = False
        if in_ol:
            chunks.append("</ol>")
            in_ol = False

    for raw_line in lines:
        line = raw_line.rstrip('\n')
        stripped = line.strip()

        if stripped.startswith("```"):
            close_lists()
            if not in_code:
                chunks.append("<pre class='md-code'><code>")
                in_code = True
            else:
                chunks.append("</code></pre>")
                in_code = False
            continue

        if in_code:
            chunks.append(html.escape(line))
            continue

        if not stripped:
            close_lists()
            continue

        if stripped.startswith("#"):
            close_lists()
            level = min(6, len(stripped) - len(stripped.lstrip("#")))
            title = html.escape(stripped[level:].strip())
            chunks.append(f"<h{level}>{title}</h{level}>")
            continue

        if stripped.startswith("- "):
            if in_ol:
                chunks.append("</ol>")
                in_ol = False
            if not in_ul:
                chunks.append("<ul>")
                in_ul = True
            chunks.append(f"<li>{html.escape(stripped[2:].strip())}</li>")
            continue

        if re.match(r"^\d+\.\s+", stripped):
            if in_ul:
                chunks.append("</ul>")
                in_ul = False
            if not in_ol:
                chunks.append("<ol>")
                in_ol = True
            item_text = re.sub(r"^\d+\.\s+", "", stripped).strip()
            chunks.append(f"<li>{html.escape(item_text)}</li>")
            continue

        close_lists()
        chunks.append(f"<p>{html.escape(stripped)}</p>")

    close_lists()
    if in_code:
        chunks.append("</code></pre>")

    return "\n".join(chunks)


def load_readme_html():
    """Load README.md and render to lightweight safe HTML."""
    readme_path = os.path.join(_PROJECT_DIR, 'README.md')
    if not os.path.isfile(readme_path):
        return "<p>README.md not found.</p>"
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_text = f.read()
    except OSError:
        return "<p>README.md could not be read.</p>"
    return markdown_to_basic_html(readme_text)


def pil_to_base64(img_pil, fmt='JPEG'):
    """Convert PIL image to base64 data URI."""
    buf = io.BytesIO()
    save_kwargs = {}
    if fmt.upper() == 'JPEG':
        save_kwargs.update({'quality': 95, 'subsampling': 0, 'optimize': True})
    img_pil.save(buf, format=fmt, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def cv2_to_base64(img_cv2):
    """Convert OpenCV image to base64 data URI."""
    _, buf = cv2.imencode('.jpg', img_cv2, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(buf).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def compute_psnr_ssim(orig_path, prot_path):
    """Quick PSNR/SSIM computation."""
    orig = cv2.imread(orig_path)
    prot = cv2.imread(prot_path)
    if orig is None or prot is None:
        return None, None
    if orig.shape != prot.shape:
        prot = cv2.resize(prot, (orig.shape[1], orig.shape[0]))

    mse = np.mean((orig.astype(float) - prot.astype(float)) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 100

    # Simplified SSIM
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    ssim_vals = []
    for c in range(3):
        a, b = orig[:, :, c].astype(float), prot[:, :, c].astype(float)
        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
        s2_a = cv2.GaussianBlur(a*a, (11, 11), 1.5) - mu_a*mu_a
        s2_b = cv2.GaussianBlur(b*b, (11, 11), 1.5) - mu_b*mu_b
        s_ab = cv2.GaussianBlur(a*b, (11, 11), 1.5) - mu_a*mu_b
        ssim_map = ((2*mu_a*mu_b+C1)*(2*s_ab+C2)) / \
                   ((mu_a**2+mu_b**2+C1)*(s2_a+s2_b+C2))
        ssim_vals.append(ssim_map.mean())
    ssim = np.mean(ssim_vals)

    return round(psnr, 2), round(ssim, 4)


def run_full_protection(img_path, preset='balanced'):
    """
    Run the FULL DeepShield protection pipeline — multi-model adversarial
    attack with DCT frequency perturbation, EOT augmentations, ArcFace +
    FaceNet + ResNet50 ensemble. Takes 1-3 minutes on GPU.
    """
    from config import PRESETS
    from pipeline import ProtectionPipeline

    config = PRESETS.get(preset, PRESETS['balanced'])
    prot_path = img_path.replace('.', '_protected.')

    pipeline = ProtectionPipeline(config, device=str(device))
    metrics = pipeline.protect(img_path, prot_path)

    return prot_path, metrics


# ── Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about_page():
    vision = [
        "Prevent identity misuse before a deepfake is created.",
        "Keep protection visually clean for real users.",
        "Provide explainable forensic detection, not black-box verdicts.",
        "Make deployment practical with one integrated Red-Team + Blue-Team workflow.",
    ]
    return render_template(
        'about.html',
        vision=vision,
        readme_html=load_readme_html(),
    )


@app.route('/api/protect', methods=['POST'])
def api_protect():
    """Apply FULL DeepShield protection pipeline and return metrics."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    t0 = time.time()
    img_path = save_upload(request.files['image'])
    prot_path = None

    try:
        # Get preset from form data (default: balanced)
        preset = request.form.get('preset', 'balanced')

        # Run the full multi-model pipeline (1-3 min)
        try:
            prot_path, pipeline_metrics = run_full_protection(img_path, preset)
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Protection failed: {str(e)}'}), 500

        # Compute quality metrics
        psnr, ssim = compute_psnr_ssim(img_path, prot_path)

        # Load images once (converted copies are in memory)
        with Image.open(img_path) as img_file:
            orig_pil = img_file.convert('RGB')
        with Image.open(prot_path) as img_file:
            prot_pil = img_file.convert('RGB')

        # Compute identity disruption scores
        arcface_sim = None
        facenet_sim = None
        try:
            import torchvision.transforms as T
            from utils.face import detect_and_crop_face
            ldr = get_loader()

            to_t = T.ToTensor()
            mtcnn = ldr.get('mtcnn')

            orig_face, _, _ = detect_and_crop_face(orig_pil, mtcnn, margin=15, device=device)
            prot_face, _, _ = detect_and_crop_face(prot_pil, mtcnn, margin=15, device=device)

            if orig_face is not None and prot_face is not None:
                orig_t = orig_face.unsqueeze(0)
                prot_t = prot_face.unsqueeze(0)
            else:
                orig_t = to_t(orig_pil).unsqueeze(0).to(device)
                prot_t = to_t(prot_pil).unsqueeze(0).to(device)

            if orig_t.shape != prot_t.shape:
                prot_t = torch.nn.functional.interpolate(
                    prot_t, size=orig_t.shape[2:], mode='bilinear', align_corners=False)

            # ArcFace
            try:
                arcface = ldr.get('arcface')
                with torch.no_grad():
                    e1 = arcface(orig_t)
                    e2 = arcface(prot_t)
                arcface_sim = round(torch.nn.functional.cosine_similarity(
                    e1, e2, dim=1).item(), 4)
            except Exception:
                pass

            # FaceNet
            try:
                facenet = ldr.get('facenet_vggface2')
                o160 = torch.nn.functional.interpolate(orig_t, (160, 160),
                                                        mode='bilinear', align_corners=False)
                p160 = torch.nn.functional.interpolate(prot_t, (160, 160),
                                                        mode='bilinear', align_corners=False)
                with torch.no_grad():
                    fe1 = facenet((o160 - 0.5) * 2.0)
                    fe2 = facenet((p160 - 0.5) * 2.0)
                facenet_sim = round(torch.nn.functional.cosine_similarity(
                    fe1, fe2, dim=1).item(), 4)
            except Exception:
                pass
        except Exception as e:
            print(f"  Warning: metric computation failed: {e}")

        elapsed = round(time.time() - t0, 2)

        return jsonify({
            'original': pil_to_base64(orig_pil),
            'protected': pil_to_base64(prot_pil),
            'metrics': {
                'psnr': psnr,
                'ssim': ssim,
                'arcface_similarity': arcface_sim,
                'facenet_similarity': facenet_sim,
                'preset': preset,
            },
            'elapsed': elapsed,
        })
    finally:
        cleanup_temp_paths([img_path, prot_path])


@app.route('/api/swap', methods=['POST'])
def api_swap():
    """Run POC face swap — clean vs protected comparison."""
    required = ['source_original', 'source_protected', 'target']
    for key in required:
        if key not in request.files:
            return jsonify({'error': f'Missing: {key}'}), 400

    t0 = time.time()

    orig_path = save_upload(request.files['source_original'])
    prot_path = save_upload(request.files['source_protected'])
    target_path = save_upload(request.files['target'])

    try:
        sw = get_swapper()

        # Swap with clean original
        result_clean = sw.swap(orig_path, target_path, original_source_path=orig_path)
        # Swap with protected image (compare against original for identity score)
        result_prot = sw.swap(prot_path, target_path,
                              original_source_path=orig_path)

        fallback_target = None
        if 'result_image' not in result_clean or 'result_image' not in result_prot:
            with Image.open(target_path) as target_file:
                fallback_target = target_file.convert('RGB')

        response = {
            'clean_swap': {
                'image': pil_to_base64(result_clean.get('result_image', fallback_target)),
                'identity_confidence': result_clean.get('identity_confidence', 0),
                'quality': result_clean.get('quality', 'UNKNOWN'),
            },
            'protected_swap': {
                'image': pil_to_base64(result_prot.get('result_image', fallback_target)),
                'identity_confidence': result_prot.get('identity_confidence', 0),
                'quality': result_prot.get('quality', 'UNKNOWN'),
            },
            'elapsed': round(time.time() - t0, 2),
        }

        if 'error' in result_clean:
            response['clean_swap']['error'] = result_clean['error']
        if 'error' in result_prot:
            response['protected_swap']['error'] = result_prot['error']

        return jsonify(response)
    finally:
        cleanup_temp_paths([orig_path, prot_path, target_path])


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """Analyze image for deepfake indicators."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    t0 = time.time()
    img_path = save_upload(request.files['image'])
    try:
        result = detector.detect(img_path)
        if 'error' in result:
            return jsonify({'error': result['error']}), 400

        response = {
            'label': result.get('label', 'UNKNOWN'),
            'confidence': result.get('confidence', 0),
            'fake_probability': result.get('fake_probability', 0),
            'analysis': result.get('analysis', {}),
            'explanation': result.get('explanation', {}),
            'analysis_extended': result.get('analysis_extended', {}),
            'elapsed': round(time.time() - t0, 2),
        }

        # Include spectrum visualization
        if 'spectrum_image' in result and result['spectrum_image'] is not None:
            response['spectrum'] = cv2_to_base64(result['spectrum_image'])

        return jsonify(response)
    finally:
        cleanup_temp_paths([img_path])


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '5000'))
    cleanup_upload_dir(max_age_seconds=None)
    print("\n" + "=" * 50)
    print("  DeepShield v2 — Web Interface")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 50)
    print(f"\n  Open http://{host}:{port} in your browser\n")

    app.run(debug=False, host=host, port=port)
