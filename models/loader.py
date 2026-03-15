"""
DeepShield — Model Loader & Registry

Lazy-loads models on first access, caches them, and manages GPU memory.
Supports:  FaceNet (vggface2 / casia-webface), ResNet50, VGG19, LPIPS.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from facenet_pytorch import MTCNN, InceptionResnetV1
import lpips


class ModelLoader:
    """
    Lazy-loading model registry.

    Usage
    -----
        loader = ModelLoader(device)
        facenet  = loader.get('facenet_vggface2')
        resnet50 = loader.get('resnet50')
    """

    def __init__(self, device):
        self.device = device
        self._cache = {}

    # ── Public API ──────────────────────────────────────────────────────

    def get(self, name: str) -> nn.Module:
        """Return a cached model, loading it on first call."""
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def preload(self, names):
        """Pre-load a list of model names (useful for progress reporting)."""
        for n in names:
            self.get(n)

    def clear(self):
        """Free all cached models and CUDA memory."""
        self._cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Internal ────────────────────────────────────────────────────────

    def _freeze(self, model):
        """Set eval mode and disable gradients."""
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def _load(self, name: str) -> nn.Module:
        print(f"    Loading {name} …")

        # ── Face detector ───────────────────────────────────────────────
        if name == 'mtcnn':
            return MTCNN(keep_all=False, device=self.device)

        # ── Identity models (FaceNet/InceptionResNetV1) ─────────────────
        elif name == 'facenet_vggface2':
            m = InceptionResnetV1(pretrained='vggface2')
            return self._freeze(m).to(self.device)

        elif name == 'facenet_casia':
            m = InceptionResnetV1(pretrained='casia-webface')
            return self._freeze(m).to(self.device)

        # ── Surrogate feature encoders ──────────────────────────────────
        elif name == 'resnet50':
            # Truncate at global avgpool (remove final FC) → (B, 2048, 1, 1)
            resnet = models.resnet50(pretrained=True)
            m = nn.Sequential(*list(resnet.children())[:-1])
            return self._freeze(m).to(self.device)

        elif name == 'vgg19':
            # .features gives the convolutional trunk only
            m = models.vgg19(pretrained=True).features
            return self._freeze(m).to(self.device)

        elif name == 'mobilenet_v2':
            # Lightweight model — web faceswap tools often use mobile
            # architectures. Attacking this improves transfer.
            mobilenet = models.mobilenet_v2(pretrained=True)
            # Truncate: features + avgpool, drop classifier
            m = nn.Sequential(mobilenet.features,
                              nn.AdaptiveAvgPool2d((1, 1)))
            return self._freeze(m).to(self.device)

        # ── Perceptual loss ─────────────────────────────────────────────
        elif name == 'lpips':
            # 'alex' is ~10x lighter than 'vgg' and almost as accurate
            m = lpips.LPIPS(net='alex')
            return self._freeze(m).to(self.device)

        else:
            raise ValueError(f"Unknown model: {name}")
