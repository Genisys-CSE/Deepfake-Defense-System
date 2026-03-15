"""
DeepShield — Protection Pipeline

Orchestrates the full protection flow:
  1. Face detection & cropping
  2. DCT frequency perturbation (base layer)
  3. Multi-model ensemble adversarial attack (main layer)
  4. Evaluation
  5. Soft-mask paste-back & save
"""

import torch
import numpy as np
from PIL import Image

from config import ProtectionConfig
from models.loader import ModelLoader
from methods.frequency import protect_frequency
from methods.adversarial import protect_adversarial
from evaluation.metrics import compute_metrics, print_metrics
from utils.face import detect_and_crop_face, paste_face_back
from utils.image import load_image, save_image


def set_seed(seed=0):
    """Deterministic seeding for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ProtectionPipeline:
    """
    End-to-end deepfake protection.

    Usage
    -----
        pipeline = ProtectionPipeline(config)
        pipeline.protect('input.jpg', 'protected.jpg')
    """

    def __init__(self, config: ProtectionConfig, seed: int = 0):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loader = ModelLoader(self.device)
        set_seed(seed)
        print(f"  Device: {self.device}")

    def _preload_models(self):
        """Load all required models upfront so progress is visible."""
        print("\n📦 Loading models …")
        # Always need MTCNN + LPIPS
        self.loader.get('mtcnn')
        self.loader.get('lpips')

        for name in self.config.identity_models:
            self.loader.get(name)
        for name in self.config.surrogate_models:
            self.loader.get(name)

    def protect(self, input_path, output_path):
        """
        Protect a single image.

        Parameters
        ----------
        input_path  : str  path to input image
        output_path : str  path to save protected image
        """
        self._preload_models()
        cfg = self.config

        # ── 1. Load & detect face ───────────────────────────────────────
        print("\n🔍 Detecting face …")
        img_pil = load_image(input_path)
        mtcnn = self.loader.get('mtcnn')
        face_tensor, img_pil, bbox = detect_and_crop_face(
            img_pil, mtcnn, margin=cfg.face_margin, device=self.device
        )

        if face_tensor is None:
            print("  ✗ No face found. Cannot protect this image.")
            return None

        print(f"  ✓ Face detected: {face_tensor.shape[1]}×{face_tensor.shape[2]}px")

        # ── 2. DCT frequency perturbation (base layer) ─────────────────
        protected = face_tensor.clone()

        if cfg.use_frequency:
            print("\n📡 Applying DCT frequency perturbation …")
            protected = protect_frequency(protected, self.device, cfg)
            print("  ✓ Frequency layer applied")

        # ── 3. Multi-model adversarial attack (main layer) ──────────────
        if cfg.use_adversarial:
            print("\n⚔️  Running multi-model adversarial attack …")
            protected = protect_adversarial(
                protected, self.device, cfg, self.loader
            )
            print("  ✓ Adversarial layer applied")

        # ── 4. Evaluate ─────────────────────────────────────────────────
        print("\n📊 Evaluating protection effectiveness …")
        metrics = compute_metrics(
            face_tensor, protected, self.loader, cfg
        )
        print_metrics(metrics)

        # ── 5. Paste back & save ────────────────────────────────────────
        print("\n💾 Saving protected image …")
        final_img = paste_face_back(img_pil, protected, bbox)
        save_image(final_img, output_path)

        return metrics

    def evaluate(self, path_a, path_b):
        """
        Compare two images (e.g. original vs deepfake output).

        Parameters
        ----------
        path_a : str  path to first image
        path_b : str  path to second image
        """
        self._preload_models()
        cfg = self.config
        mtcnn = self.loader.get('mtcnn')

        print(f"\n🔍 Comparing:")
        print(f"  A: {path_a}")
        print(f"  B: {path_b}")

        img_a = load_image(path_a)
        img_b = load_image(path_b)

        face_a, _, _ = detect_and_crop_face(img_a, mtcnn, margin=cfg.face_margin,
                                             device=self.device)
        face_b, _, _ = detect_and_crop_face(img_b, mtcnn, margin=cfg.face_margin,
                                             device=self.device)

        if face_a is None or face_b is None:
            print("  ✗ Could not detect faces in both images.")
            return None

        metrics = compute_metrics(face_a, face_b, self.loader, cfg)
        print_metrics(metrics)
        return metrics
