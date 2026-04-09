"""
DeepShield — Protection Pipeline

Orchestrates the full protection flow:

  1. Preload models (with VRAM check)
  2. Face detection & cropping  (MTCNN)
  3. DCT frequency perturbation (base layer — JPEG-robust)
  4. Multi-model adversarial attack (main layer — PGD + EOT)
  5. Evaluation metrics
  6. Soft-mask paste-back & save
"""

import sys
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


def set_seed(seed: int = 0) -> None:
    """Deterministic seeding for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ProtectionPipeline:
    """
    End-to-end deepfake protection pipeline.

    Usage
    -----
        pipeline = ProtectionPipeline(config)
        pipeline.protect('input.jpg', 'protected.jpg')
    """

    def __init__(self, config: ProtectionConfig, seed: int = 0,
                 device: str | None = None):
        self.config = config

        # Device selection: explicit flag > CUDA > CPU
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.loader = ModelLoader(self.device)
        set_seed(seed)

        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            print(f"  Device: {self.device}  ({gpu_name}, {vram_mb} MB VRAM)")
        else:
            print(f"  Device: {self.device}  (CPU — this will be slower)")

    # ── Pre-load models ─────────────────────────────────────────────────

    def _preload_models(self) -> None:
        """Load all required models upfront so download progress is visible."""
        print("\n📦 Loading models …")
        try:
            # Always need MTCNN + LPIPS
            self.loader.get('mtcnn')
            self.loader.get('lpips')

            for name in self.config.identity_models:
                self.loader.get(name)
            for name in self.config.surrogate_models:
                self.loader.get(name)

        except torch.cuda.OutOfMemoryError:
            print("\n  ⚠ CUDA out of memory while loading models.")
            print("    Falling back to CPU (will be slower).")
            self.loader.clear()
            self.device = torch.device('cpu')
            self.loader = ModelLoader(self.device)
            # Retry on CPU
            self.loader.get('mtcnn')
            self.loader.get('lpips')
            for name in self.config.identity_models:
                self.loader.get(name)
            for name in self.config.surrogate_models:
                self.loader.get(name)

    # ── Protect ─────────────────────────────────────────────────────────

    def protect(self, input_path: str, output_path: str) -> dict | None:
        """
        Protect a single image against deepfake creation.

        Parameters
        ----------
        input_path  : Path to source image (any format PIL can read).
        output_path : Path to save the protected image.

        Returns
        -------
        metrics : dict of evaluation metrics, or None on failure.
        """
        self._preload_models()
        cfg = self.config

        # ── 1. Load & detect face ───────────────────────────────────────
        print("\n🔍 Detecting face …")
        try:
            img_pil = load_image(input_path)
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            return None

        mtcnn = self.loader.get('mtcnn')
        face_tensor, img_pil, bbox = detect_and_crop_face(
            img_pil, mtcnn, margin=cfg.face_margin, device=self.device,
        )

        if face_tensor is None:
            print("  ✗ No face detected — cannot protect this image.")
            print("    Make sure the image contains a clearly visible face.")
            return None

        print(f"  ✓ Face detected: {face_tensor.shape[2]}×{face_tensor.shape[1]}px")

        # ── 2. DCT frequency perturbation (base layer) ─────────────────
        protected = face_tensor.clone()

        if cfg.use_frequency:
            print("\n📡 Applying DWT frequency perturbation …")
            protected = protect_frequency(protected, cfg.freq_epsilon)
            print("  ✓ Frequency layer applied")

        # ── 3. Multi-model adversarial attack (main layer) ──────────────
        if cfg.use_adversarial:
            print("\n⚔️  Running multi-model adversarial attack …")
            try:
                protected = protect_adversarial(
                    protected, self.device, cfg, self.loader,
                )
            except torch.cuda.OutOfMemoryError:
                print("\n  ⚠ CUDA out of memory during adversarial attack.")
                print("    Retrying on CPU …")
                self.loader.clear()
                self.device = torch.device('cpu')
                self.loader = ModelLoader(self.device)
                self._preload_models()
                protected = protected.to(self.device)
                face_tensor = face_tensor.to(self.device)
                protected = protect_adversarial(
                    protected, self.device, cfg, self.loader,
                )
            print("  ✓ Adversarial layer applied")

        # ── 4. Evaluate ─────────────────────────────────────────────────
        print("\n📊 Evaluating protection effectiveness …")
        metrics = compute_metrics(face_tensor, protected, self.loader, cfg)
        print_metrics(metrics)

        # ── 5. Paste back & save ────────────────────────────────────────
        print("\n💾 Saving protected image …")
        final_img = paste_face_back(img_pil, protected, bbox)
        save_image(final_img, output_path)

        # Free GPU memory after processing
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return metrics

    # ── Evaluate (compare two images) ───────────────────────────────────

    def evaluate(self, path_a: str, path_b: str) -> dict | None:
        """
        Compare two images (e.g. original vs deepfake output).

        Parameters
        ----------
        path_a : Path to the first image (e.g. original).
        path_b : Path to the second image (e.g. deepfake output).

        Returns
        -------
        metrics : dict of comparison metrics, or None on failure.
        """
        self._preload_models()
        cfg = self.config
        mtcnn = self.loader.get('mtcnn')

        print(f"\n🔍 Comparing:")
        print(f"  A: {path_a}")
        print(f"  B: {path_b}")

        try:
            img_a = load_image(path_a)
            img_b = load_image(path_b)
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            return None

        face_a, _, _ = detect_and_crop_face(
            img_a, mtcnn, margin=cfg.face_margin, device=self.device,
        )
        face_b, _, _ = detect_and_crop_face(
            img_b, mtcnn, margin=cfg.face_margin, device=self.device,
        )

        if face_a is None or face_b is None:
            print("  ✗ Could not detect faces in both images.")
            return None

        metrics = compute_metrics(face_a, face_b, self.loader, cfg)
        print_metrics(metrics)

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return metrics
