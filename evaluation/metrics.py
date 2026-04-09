"""
DeepShield — Comprehensive Evaluation Metrics

Evaluates protection effectiveness across multiple dimensions:

  Identity disruption  — Multi-model FaceNet cosine similarity
  Feature disruption   — Per-layer surrogate-model cosine similarity
  Visual quality       — LPIPS, SSIM, PSNR
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def compute_metrics(face_orig: torch.Tensor,
                    face_prot: torch.Tensor,
                    model_loader,
                    config) -> dict:
    """
    Compute a comprehensive set of metrics between the original and
    protected face tensors.

    Parameters
    ----------
    face_orig    : Tensor (C, H, W) in [0, 1].
    face_prot    : Tensor (C, H, W) in [0, 1].
    model_loader : ModelLoader instance.
    config       : ProtectionConfig instance.

    Returns
    -------
    metrics : dict mapping ``category/name`` → float.
    """
    device = face_orig.device
    metrics: dict[str, float] = {}

    with torch.no_grad():
        orig_batch = face_orig.unsqueeze(0)
        prot_batch = face_prot.unsqueeze(0)

        # ── Identity similarity (FaceNet) ───────────────────────────────
        for model_name in config.identity_models:
            id_model = model_loader.get(model_name)
            orig_fn = F.interpolate(orig_batch, size=(160, 160),
                                    mode='bilinear', align_corners=False)
            prot_fn = F.interpolate(prot_batch, size=(160, 160),
                                    mode='bilinear', align_corners=False)
            orig_emb = id_model((orig_fn - 0.5) * 2.0)
            prot_emb = id_model((prot_fn - 0.5) * 2.0)
            cos = F.cosine_similarity(orig_emb, prot_emb, dim=1).item()
            pretty = model_name.replace('facenet_', 'FaceNet/')
            metrics[f'identity/{pretty}'] = cos

        # ── Surrogate feature similarity ────────────────────────────────
        normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        for model_name in config.surrogate_models:
            if model_name == 'resnet50':
                model = model_loader.get('resnet50')
                _orig = normalize(F.interpolate(
                    orig_batch, (224, 224), mode='bilinear', align_corners=False))
                _prot = normalize(F.interpolate(
                    prot_batch, (224, 224), mode='bilinear', align_corners=False))
                cos = F.cosine_similarity(
                    model(_orig).flatten(1), model(_prot).flatten(1), dim=1,
                ).item()
                metrics['surrogate/ResNet50'] = cos

            elif model_name == 'vgg19':
                from models.feature_extractor import MultiLayerFeatureExtractor
                vgg = model_loader.get('vgg19')
                extractor = MultiLayerFeatureExtractor(vgg, config.vgg_layers)
                orig_feats = extractor.extract(orig_batch)
                prot_feats = extractor.extract(prot_batch)
                for layer in config.vgg_layers:
                    o = orig_feats[layer].flatten(1)
                    p = prot_feats[layer].flatten(1)
                    cos = F.cosine_similarity(o, p, dim=1).item()
                    metrics[f'surrogate/VGG19_{layer}'] = cos

            elif model_name == 'mobilenet_v2':
                model = model_loader.get('mobilenet_v2')
                _orig = normalize(F.interpolate(
                    orig_batch, (224, 224), mode='bilinear', align_corners=False))
                _prot = normalize(F.interpolate(
                    prot_batch, (224, 224), mode='bilinear', align_corners=False))
                cos = F.cosine_similarity(
                    model(_orig).flatten(1), model(_prot).flatten(1), dim=1,
                ).item()
                metrics['surrogate/MobileNetV2'] = cos

        # ── LPIPS ──────────────────────────────────────────────────────
        lpips_model = model_loader.get('lpips')
        lpips_val = lpips_model(
            (prot_batch - 0.5) * 2.0,
            (orig_batch - 0.5) * 2.0,
        ).item()
        metrics['quality/LPIPS'] = lpips_val

    # ── SSIM & PSNR (CPU, numpy) ────────────────────────────────────────
    orig_np = (face_orig.detach().cpu().numpy().transpose(1, 2, 0) * 255
               ).clip(0, 255).astype(np.uint8)
    prot_np = (face_prot.detach().cpu().numpy().transpose(1, 2, 0) * 255
               ).clip(0, 255).astype(np.uint8)

    metrics['quality/SSIM'] = float(ssim(
        orig_np, prot_np, channel_axis=2, data_range=255,
    ))

    mse = float(np.mean((orig_np.astype(float) - prot_np.astype(float)) ** 2))
    metrics['quality/PSNR'] = (
        10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
    )

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics with visual bars and status labels."""

    print("\n" + "=" * 60)
    print("  DeepShield — Protection Evaluation")
    print("=" * 60)

    # Group by category
    categories: dict[str, list] = {}
    for key, val in metrics.items():
        cat, name = key.split('/', 1)
        categories.setdefault(cat, []).append((name, val))

    labels = {
        'identity':  '🎭 Identity Disruption  (lower = better protection)',
        'surrogate': '🧠 Feature Disruption   (lower = better protection)',
        'quality':   '🖼️  Visual Quality        (higher = less visible)',
    }

    for cat in ('identity', 'surrogate', 'quality'):
        if cat not in categories:
            continue
        print(f"\n  {labels.get(cat, cat)}")
        print("  " + "-" * 54)

        for name, val in categories[cat]:
            if cat in ('identity', 'surrogate'):
                # Cosine similarity: 1.0 = identical, 0.0 = orthogonal
                bar_val = max(0.0, min(1.0, val))
                bar_len = int(bar_val * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                if val < 0.5:
                    status = "✓ STRONG"
                elif val < 0.7:
                    status = "✓ DISRUPTED"
                elif val < 0.9:
                    status = "~ PARTIAL"
                else:
                    status = "✗ WEAK"
                print(f"    {name:30s}  {val:+.4f}  [{bar}]  {status}")

            elif name == 'LPIPS':
                quality = (
                    "✓ invisible" if val < 0.05
                    else "~ slight" if val < 0.15
                    else "⚠ noticeable"
                )
                print(f"    {name:30s}  {val:.4f}  {quality}")

            elif name == 'SSIM':
                quality = (
                    "✓ excellent" if val > 0.95
                    else "~ good" if val > 0.85
                    else "⚠ degraded"
                )
                print(f"    {name:30s}  {val:.4f}  {quality}")

            elif name == 'PSNR':
                quality = (
                    "✓ invisible" if val > 38
                    else "~ good" if val > 32
                    else "⚠ visible"
                )
                print(f"    {name:30s}  {val:.2f} dB  {quality}")

    print("\n" + "=" * 60)
