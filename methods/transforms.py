"""
DeepShield — Enhanced Expectation-Over-Transformation (EOT)

Simulates real-world transformations the image will undergo (social media
compression, deepfake preprocessing, etc.) to make perturbations robust.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  Differentiable JPEG Simulation (Straight-Through Estimator)
# ═══════════════════════════════════════════════════════════════════════

class _StraightThroughJPEG(torch.autograd.Function):
    """
    Forward: real JPEG encode/decode via OpenCV.
    Backward: straight-through (identity gradient) so PGD can optimise
              through JPEG compression.
    """

    @staticmethod
    def forward(ctx, x, quality):
        device = x.device
        result = []
        for i in range(x.shape[0]):
            img_np = (x[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img_np = np.transpose(img_np, (1, 2, 0))                   # CHW → HWC
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            _, enc = cv2.imencode('.jpg', img_bgr,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is None:
                dec = img_bgr
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            dec = np.transpose(dec, (2, 0, 1)).astype(np.float32) / 255.0
            result.append(torch.from_numpy(dec).to(device))

        return torch.stack(result)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def _jpeg_compress(x, quality):
    """Differentiable JPEG compression with STE."""
    return _StraightThroughJPEG.apply(x, quality)


# ═══════════════════════════════════════════════════════════════════════
#  Full EOT Pipeline
# ═══════════════════════════════════════════════════════════════════════

def apply_eot(x):
    """
    Apply a random combination of realistic transformations.

    Parameters
    ----------
    x : (B, C, H, W)  batch of images in [0, 1]

    Returns
    -------
    x : (B, C, H, W)  transformed images, clamped to [0, 1]
    """
    # 1) Random affine: rotation ±5°, scale 0.95–1.05
    x = T.RandomAffine(degrees=5, scale=(0.95, 1.05))(x)

    # 2) JPEG compression (quality 70–95)
    quality = int(torch.randint(70, 96, (1,)).item())
    x = _jpeg_compress(x, quality)

    # 3) Color jitter: brightness/contrast/saturation
    if torch.rand(1).item() < 0.5:
        x = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)(x)

    # 4) Gaussian noise  (σ ∈ [0.005, 0.025])
    sigma = 0.005 + torch.rand(1).item() * 0.02
    noise = torch.randn_like(x) * sigma
    x = x + noise

    # 5) Random Gaussian blur (50% chance, simulates rescaling)
    if torch.rand(1).item() < 0.3:
        x = T.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))(x)

    return torch.clamp(x, 0.0, 1.0)
