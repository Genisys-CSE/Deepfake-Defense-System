"""
DeepShield — Model Loader & Registry

Lazy-loads models on first access, caches them, and manages GPU memory.
Supports:  FaceNet (vggface2 / casia-webface), ResNet50, VGG19,
           MobileNetV2, MTCNN, LPIPS.

All models are loaded in eval mode with gradients disabled.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from facenet_pytorch import MTCNN, InceptionResnetV1
import lpips

# Modern weights API (torchvision >= 0.13).  Falls back to legacy
# ``pretrained=True`` on older installs so users never hit import errors.
try:
    from torchvision.models import (
        ResNet50_Weights, VGG19_Weights, MobileNet_V2_Weights,
    )
    _USE_MODERN_WEIGHTS = True
except ImportError:
    _USE_MODERN_WEIGHTS = False


class ModelLoader:
    """
    Lazy-loading model registry with automatic GPU placement.

    Usage
    -----
        loader = ModelLoader(device)
        facenet  = loader.get('facenet_vggface2')
        resnet50 = loader.get('resnet50')
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._cache: dict[str, nn.Module] = {}

    # ── Public API ──────────────────────────────────────────────────────

    def get(self, name: str) -> nn.Module:
        """Return a cached model, loading it on first call."""
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def preload(self, names: list[str]) -> None:
        """Pre-load a list of model names (useful for progress reporting)."""
        for n in names:
            self.get(n)

    def clear(self) -> None:
        """Free all cached models and reclaim CUDA memory."""
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Internal ────────────────────────────────────────────────────────

    @staticmethod
    def _freeze(model: nn.Module) -> nn.Module:
        """Set eval mode and disable gradients for all parameters."""
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def _load(self, name: str) -> nn.Module:
        print(f"    Loading {name} …")

        # ── Face detector ───────────────────────────────────────────────
        if name == 'mtcnn':
            return MTCNN(keep_all=False, device=self.device)

        # ── Identity models (FaceNet / InceptionResNetV1) ───────────────
        if name == 'facenet_vggface2':
            m = InceptionResnetV1(pretrained='vggface2')
            return self._freeze(m).to(self.device)

        if name == 'facenet_casia':
            m = InceptionResnetV1(pretrained='casia-webface')
            return self._freeze(m).to(self.device)

        # ── Surrogate feature encoders ──────────────────────────────────
        if name == 'resnet50':
            # Truncate at global average-pool (remove final FC) → (B, 2048, 1, 1)
            if _USE_MODERN_WEIGHTS:
                resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                resnet = models.resnet50(pretrained=True)
            m = nn.Sequential(*list(resnet.children())[:-1])
            return self._freeze(m).to(self.device)

        if name == 'vgg19':
            # .features gives the convolutional backbone only
            if _USE_MODERN_WEIGHTS:
                m = models.vgg19(weights=VGG19_Weights.DEFAULT).features
            else:
                m = models.vgg19(pretrained=True).features
            return self._freeze(m).to(self.device)

        if name == 'mobilenet_v2':
            # Lightweight encoder — web face-swap tools commonly use
            # mobile architectures; attacking this improves transferability.
            if _USE_MODERN_WEIGHTS:
                mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            else:
                mobilenet = models.mobilenet_v2(pretrained=True)
            m = nn.Sequential(
                mobilenet.features,
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            return self._freeze(m).to(self.device)

        # ── Perceptual quality loss ─────────────────────────────────────
        if name == 'lpips':
            # 'alex' is ~10× lighter than 'vgg' and nearly as accurate
            m = lpips.LPIPS(net='alex')
            return self._freeze(m).to(self.device)

        # ── ArcFace (InsightFace w600k_r50 — exact FaceFusion model) ───
        if name == 'arcface':
            from models.arcface_loader import load_arcface
            return load_arcface(self.device)

        raise ValueError(
            f"Unknown model '{name}'.  Available: mtcnn, facenet_vggface2, "
            f"facenet_casia, resnet50, vgg19, mobilenet_v2, lpips, arcface"
        )
