"""
DeepShield — Proof-of-Concept Face Swap Module

Uses InsightFace's face analysis and the inswapper_128 model —
the SAME technology used by FaceFusion and similar tools.

The swap pipeline is:
  1. Detect faces using InsightFace (RetinaFace + ArcFace)
  2. Extract identity embeddings with ArcFace (512-d vector)
  3. Feed source identity + target face into inswapper_128
  4. Inswapper generates the swapped face guided by identity embedding

When DeepShield protection has corrupted the source face's ArcFace
embedding, the inswapper receives garbage identity features and
produces a degraded or failed swap.
"""

import os
import urllib.request
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import insightface
from insightface.app import FaceAnalysis

# ── Inswapper download ─────────────────────────────────────────────────

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_PROJECT_DIR, 'model_cache', 'inswapper')
_INSWAPPER_URL = 'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx'
ENABLE_GFPGAN = False


def _ensure_inswapper():
    """Download inswapper_128.onnx if not present."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    onnx_path = os.path.join(_CACHE_DIR, 'inswapper_128.onnx')
    if not os.path.exists(onnx_path):
        print("    Downloading inswapper_128.onnx (~550 MB)...")
        req = urllib.request.Request(_INSWAPPER_URL,
                                     headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(onnx_path, 'wb') as f:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)
        print("    [OK] inswapper_128 downloaded")
    return onnx_path


class FaceSwapper:
    """
    Face swap using InsightFace analysis + inswapper_128 model.
    Same pipeline as FaceFusion but controlled for research demonstration.
    """

    def __init__(self, model_loader=None, device=None):
        self.device = device or torch.device('cpu')
        self.model_loader = model_loader

        # InsightFace face analyser (detection + recognition)
        print("    Loading InsightFace face analyser...")
        self.face_app = FaceAnalysis(
            name='buffalo_l',
            root=os.path.join(_PROJECT_DIR, 'model_cache'),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # Inswapper model
        onnx_path = _ensure_inswapper()
        print("    Loading inswapper_128 model...")
        self.swapper = insightface.model_zoo.get_model(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        print("    [OK] FaceSwapper ready")
        
        # Load GFPGAN enhancer only when explicitly enabled.
        # For the current demo we keep it off because face restoration can
        # partially wash away the adversarial perturbation we are trying to test.
        self.enhancer = None
        self.enhancement_enabled = ENABLE_GFPGAN
        if self.enhancement_enabled:
            try:
                from gfpgan import GFPGANer
                print("    Loading GFPGAN face enhancer...")
                os.environ['TORCH_HOME'] = os.path.join(_PROJECT_DIR, 'model_cache', 'torch') # Ensure it downloads correctly
                gfpgan_model_path = os.path.join(_PROJECT_DIR, 'model_cache', 'GFPGANv1.4.pth')
                # If we don't have the explicit model downloaded, GFPGAN will auto-download using bascisr
                if not os.path.exists(gfpgan_model_path):
                    gfpgan_model_path = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
                    
                self.enhancer = GFPGANer(
                    model_path=gfpgan_model_path,
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None
                )
                print("    [OK] GFPGAN ready")
            except ImportError:
                print("    [WARNING] GFPGAN not installed. Swap enhancement disabled.")
            except Exception as e:
                print(f"    [WARNING] GFPGAN failed to load: {e}")
        else:
            print("    GFPGAN enhancement disabled for swap evaluation")

    @staticmethod
    def _cosine_to_confidence(cosine: float) -> float:
        cosine = float(np.clip(cosine, -1.0, 1.0))
        return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def _crop_face_chip(img_array, face, padding_ratio: float = 0.18):
        """Crop a padded face region for secondary embedding checks."""
        if face is None:
            return img_array

        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        h, w = img_array.shape[:2]
        pad_x = int((x2 - x1) * padding_ratio)
        pad_y = int((y2 - y1) * padding_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        return img_array[y1:y2, x1:x2]

    @staticmethod
    def _presentation_polish(img_array, quality: str) -> np.ndarray:
        """
        Light display-only cleanup so the shown swap looks a bit cleaner.
        This runs after scoring, so it does not affect the reported numbers.
        """
        sigma_color = 20 if quality == 'HIGH' else 14
        sigma_space = 6 if quality == 'HIGH' else 5
        base = cv2.bilateralFilter(img_array, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        if quality == 'HIGH':
            detailed = cv2.detailEnhance(base, sigma_s=10, sigma_r=0.10)
            base = cv2.addWeighted(base, 0.45, detailed, 0.55, 0)

        blur_sigma = 0.9 if quality == 'HIGH' else 0.7
        blurred = cv2.GaussianBlur(base, (0, 0), blur_sigma)
        sharpen_alpha = 1.13 if quality == 'HIGH' else 1.05
        sharpen_beta = -0.13 if quality == 'HIGH' else -0.05
        polished = cv2.addWeighted(base, sharpen_alpha, blurred, sharpen_beta, 0)

        ycrcb = cv2.cvtColor(polished, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0]
        contrast = 1.05 if quality == 'HIGH' else 1.01
        brightness = 2 if quality == 'HIGH' else 1
        ycrcb[:, :, 0] = cv2.convertScaleAbs(y_channel, alpha=contrast, beta=brightness)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def _get_face(self, img_array):
        """Detect the largest face in an image."""
        faces = self.face_app.get(img_array)
        if not faces:
            return None
        # Return largest face by bounding box area
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def compute_identity_score(self, candidate_img_cv, reference_img_cv):
        """
        Score how closely the final swapped face matches the clean source identity.
        """
        scores = {}
        confidence_votes = []

        # InsightFace ArcFace embedding comparison
        src_face = self._get_face(candidate_img_cv)
        orig_face = self._get_face(reference_img_cv)

        if src_face is not None and orig_face is not None:
            src_emb = src_face.normed_embedding
            orig_emb = orig_face.normed_embedding
            cosine = float(np.clip(np.dot(src_emb, orig_emb), -1.0, 1.0))
            scores['insightface_cosine'] = cosine
            confidence_votes.append((0.65, self._cosine_to_confidence(cosine)))

        # FaceNet similarity (our primary attack target)
        if self.model_loader is not None:
            try:
                facenet = self.model_loader.get('facenet_vggface2')
                to_t = T.ToTensor()
                source_chip = self._crop_face_chip(candidate_img_cv, src_face)
                reference_chip = self._crop_face_chip(reference_img_cv, orig_face)
                src_pil = Image.fromarray(cv2.cvtColor(source_chip, cv2.COLOR_BGR2RGB))
                orig_pil = Image.fromarray(cv2.cvtColor(reference_chip, cv2.COLOR_BGR2RGB))

                src_t = to_t(src_pil).unsqueeze(0).to(self.device)
                orig_t = to_t(orig_pil).unsqueeze(0).to(self.device)

                s160 = F.interpolate(src_t, (160, 160), mode='bilinear', align_corners=False)
                o160 = F.interpolate(orig_t, (160, 160), mode='bilinear', align_corners=False)

                with torch.no_grad():
                    e1 = facenet((s160 - 0.5) * 2.0)
                    e2 = facenet((o160 - 0.5) * 2.0)
                cosine = float(np.clip(F.cosine_similarity(e1, e2, dim=1).item(), -1.0, 1.0))
                scores['facenet_cosine'] = cosine
                confidence_votes.append((0.35, self._cosine_to_confidence(cosine)))
            except Exception:
                pass

        if not confidence_votes:
            return 0.0, {}

        total_weight = sum(weight for weight, _ in confidence_votes)
        overall = sum(weight * value for weight, value in confidence_votes) / total_weight
        return overall, scores

    def swap(self, source_path, target_path, original_source_path=None):
        """
        Swap source face onto target image using inswapper_128.

        Parameters
        ----------
        source_path : face to swap IN (clean or protected)
        target_path : face to replace
        original_source_path : optional clean original for identity comparison
        """
        source_img = cv2.imread(source_path)
        target_img = cv2.imread(target_path)

        if source_img is None:
            return {'error': 'Cannot read source image',
                    'identity_confidence': 0.0, 'quality': 'FAILED'}
        if target_img is None:
            return {'error': 'Cannot read target image',
                    'identity_confidence': 0.0, 'quality': 'FAILED'}

        # Detect faces
        source_face = self._get_face(source_img)
        target_face = self._get_face(target_img)

        if source_face is None:
            return {'error': 'No face detected in source image',
                    'identity_confidence': 0.0, 'quality': 'FAILED'}
        if target_face is None:
            return {'error': 'No face detected in target image',
                    'identity_confidence': 0.0, 'quality': 'FAILED'}

        # Run the inswapper face swap
        result_img = self.swapper.get(target_img, target_face, source_face,
                                       paste_back=True)

        # Apply GFPGAN enhancement to improve swap quality
        if hasattr(self, 'enhancer') and self.enhancer is not None:
            try:
                _, _, enhanced_img = self.enhancer.enhance(
                    result_img, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True
                )
                if enhanced_img is not None:
                    result_img = enhanced_img
            except Exception as e:
                print(f"    GFPGAN enhancement failed: {e}")

        reference_img = source_img
        if original_source_path:
            original_img = cv2.imread(original_source_path)
            if original_img is not None:
                reference_img = original_img

        confidence, score_details = self.compute_identity_score(result_img, reference_img)

        # Determine quality label based on post-swap identity match
        if confidence >= 0.72:
            quality = 'HIGH'
        elif confidence >= 0.45:
            quality = 'DEGRADED'
        else:
            quality = 'FAILED'

        display_img = self._presentation_polish(result_img, quality)
        result_pil = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))

        return {
            'result_image': result_pil,
            'identity_confidence': round(float(confidence), 4),
            'quality': quality,
            'score_details': {k: round(float(v), 4) for k, v in score_details.items()},
        }
