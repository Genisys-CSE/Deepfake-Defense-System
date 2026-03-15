"""
DeepShield — Comprehensive Evaluation Metrics

Evaluates protection effectiveness across multiple dimensions:
  - Identity disruption (multi-model FaceNet cosine similarity)
  - Feature disruption  (multi-model surrogate cosine similarity)
  - Visual quality      (LPIPS, SSIM, PSNR)
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from skimage.metrics import structural_similarity as ssim


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def compute_metrics(face_orig, face_prot, model_loader, config):
    """
    Compute a comprehensive set of metrics between the original and
    protected face tensors.

    Parameters
    ----------
    face_orig    : (C, H, W) in [0, 1]
    face_prot    : (C, H, W) in [0, 1]
    model_loader : ModelLoader
    config       : ProtectionConfig

    Returns
    -------
    metrics : dict[str → float]
    """
    device = face_orig.device
    metrics = {}

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
            pretty_name = model_name.replace('facenet_', 'FaceNet/')
            metrics[f'identity/{pretty_name}'] = cos

        # ── Surrogate feature similarity ────────────────────────────────
        normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        for model_name in config.surrogate_models:
            if model_name == 'resnet50':
                model = model_loader.get('resnet50')
                orig_surr = F.interpolate(orig_batch, size=(224, 224),
                                          mode='bilinear', align_corners=False)
                prot_surr = F.interpolate(prot_batch, size=(224, 224),
                                          mode='bilinear', align_corners=False)
                orig_feat = model(normalize(orig_surr)).flatten(1)
                prot_feat = model(normalize(prot_surr)).flatten(1)
                cos = F.cosine_similarity(orig_feat, prot_feat, dim=1).item()
                metrics['surrogate/ResNet50'] = cos

            elif model_name == 'vgg19':
                from models.feature_extractor import MultiLayerFeatureExtractor
                vgg = model_loader.get('vgg19')
                extractor = MultiLayerFeatureExtractor(vgg, config.vgg_layers)
                orig_feats = extractor.extract(orig_batch)
                prot_feats = extractor.extract(prot_batch)
                for layer_name in config.vgg_layers:
                    o = orig_feats[layer_name].flatten(1)
                    p = prot_feats[layer_name].flatten(1)
                    cos = F.cosine_similarity(o, p, dim=1).item()
                    metrics[f'surrogate/VGG19_{layer_name}'] = cos

            elif model_name == 'mobilenet_v2':
                model = model_loader.get('mobilenet_v2')
                orig_surr = F.interpolate(orig_batch, size=(224, 224),
                                          mode='bilinear', align_corners=False)
                prot_surr = F.interpolate(prot_batch, size=(224, 224),
                                          mode='bilinear', align_corners=False)
                orig_feat = model(normalize(orig_surr)).flatten(1)
                prot_feat = model(normalize(prot_surr)).flatten(1)
                cos = F.cosine_similarity(orig_feat, prot_feat, dim=1).item()
                metrics['surrogate/MobileNetV2'] = cos

        # ── LPIPS ───────────────────────────────────────────────────────
        lpips_model = model_loader.get('lpips')
        lpips_val = lpips_model(
            (prot_batch - 0.5) * 2.0,
            (orig_batch - 0.5) * 2.0
        ).item()
        metrics['quality/LPIPS'] = lpips_val

    # ── SSIM & PSNR (on CPU numpy) ─────────────────────────────────────
    orig_np = (face_orig.detach().cpu().numpy().transpose(1, 2, 0) * 255
               ).clip(0, 255).astype(np.uint8)
    prot_np = (face_prot.detach().cpu().numpy().transpose(1, 2, 0) * 255
               ).clip(0, 255).astype(np.uint8)

    metrics['quality/SSIM'] = ssim(orig_np, prot_np,
                                    channel_axis=2, data_range=255)

    mse = np.mean((orig_np.astype(float) - prot_np.astype(float)) ** 2)
    if mse > 0:
        metrics['quality/PSNR'] = 10 * np.log10(255.0 ** 2 / mse)
    else:
        metrics['quality/PSNR'] = float('inf')

    return metrics


def print_metrics(metrics):
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 56)
    print("  DeepShield — Evaluation Results")
    print("=" * 56)

    # Group by category
    categories = {}
    for key, val in metrics.items():
        cat, name = key.split('/', 1)
        categories.setdefault(cat, []).append((name, val))

    labels = {
        'identity':  '🎭  Identity Disruption  (lower = better protection)',
        'surrogate': '🧠  Feature Disruption   (lower = better protection)',
        'quality':   '🖼️   Visual Quality        (higher = less visible)',
    }

    for cat in ['identity', 'surrogate', 'quality']:
        if cat not in categories:
            continue
        print(f"\n  {labels.get(cat, cat)}")
        print("  " + "-" * 50)
        for name, val in categories[cat]:
            if cat in ('identity', 'surrogate'):
                # Cosine similarity: 1.0 = identical
                bar_len = int(val * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                status = "✓ DISRUPTED" if val < 0.7 else ("~ PARTIAL" if val < 0.9 else "✗ WEAK")
                print(f"    {name:30s}  {val:+.4f}  [{bar}]  {status}")
            elif name == 'LPIPS':
                print(f"    {name:30s}  {val:.4f}")
            elif name == 'SSIM':
                print(f"    {name:30s}  {val:.4f}")
            elif name == 'PSNR':
                print(f"    {name:30s}  {val:.2f} dB")

    print("\n" + "=" * 56)
