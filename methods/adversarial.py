"""
DeepShield V3.0 — The FaceFusion Breaker

Direct implementation of the ultimate 3-step plan:
1. Alignment-Aware EOT (Simulates FaceFusion Affine Warp)
2. RetinaFace/YOLO Backbone Disruption (Massive ResNet/VGG adversarial scatter)
3. ArcFace/Inswapper survival through aggressive perceptual bounding.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

from methods.transforms import apply_eot
from models.feature_extractor import (
    MultiLayerFeatureExtractor,
    ResNetFeatureExtractor,
)


def _build_focus_mask(height: int,
                      width: int,
                      device: torch.device,
                      dtype: torch.dtype,
                      floor: float,
                      power: float) -> torch.Tensor:
    """Keep perturbations strongest on central identity features."""
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    radial = torch.sqrt((xx / 0.92) ** 2 + ((yy + 0.08) / 1.08) ** 2)
    center = torch.clamp(1.0 - radial, min=0.0).pow(power)
    return (floor + (1.0 - floor) * center).unsqueeze(0)


def _desaturate_delta(delta: torch.Tensor, chroma_scale: float) -> torch.Tensor:
    """Suppress color blotches, which are the most visible artifacts."""
    if delta.shape[0] != 3 or chroma_scale >= 1.0:
        return delta

    luma = (
        0.299 * delta[0:1]
        + 0.587 * delta[1:2]
        + 0.114 * delta[2:3]
    )
    luma_rgb = luma.repeat(3, 1, 1)
    return luma_rgb + chroma_scale * (delta - luma_rgb)


def protect_adversarial(face_tensor: torch.Tensor,
                        device: torch.device,
                        config,
                        model_loader) -> torch.Tensor:
    """
    Run the DeepShield V3.0 (Anti-FaceFusion) adversarial attack.
    """
    # Increased attack intensity for FaceFusion specifically
    epsilon     = config.epsilon
    steps       = config.steps
    lr          = config.lr
    eot_samples = config.eot_samples
    
    # We drop MTCNN entirely. It doesn't work for FaceFusion.
    lam_arc       = getattr(config, 'lambda_arcface', 5.0)
    lam_detector  = getattr(config, 'lambda_landmark', 5.0) # Redirected to YOLO/RetinaFace backbones
    lam_perc      = getattr(config, 'lambda_perceptual', 15.0)
    arcface_focus = getattr(config, 'arcface_focus', 1.0)
    facenet_focus = getattr(config, 'facenet_focus', 1.0)

    face_tensor = face_tensor.to(device)

    # ── 1. Load FaceFusion specific targets ────────────────────────────
    arcface_model = model_loader.get('arcface')
    facenet_model = model_loader.get('facenet_vggface2') # Added FaceNet for strong identity disruption
    lpips_model = model_loader.get('lpips')
    
    # FaceFusion uses YOLOv8 and RetinaFace. Both are built on deep CNN backbones.
    # By maximizing the feature displacement in ResNet and VGG, we literally 
    # blind the face detector. If FaceFusion can't detect a face, the swap crashes.
    vgg_extractor = MultiLayerFeatureExtractor(
        model_loader.get('vgg19'), 
        ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4'] # Deep layers to blast YOLO
    )
    resnet_extractor = ResNetFeatureExtractor(model_loader.get('resnet50'))

    # ── Compute clean features ───────────────────────────────
    with torch.no_grad():
        face_batch = face_tensor.unsqueeze(0)
        orig_arc_emb = arcface_model(face_batch).detach()
        
        # Resize for FaceNet
        orig_160 = F.interpolate(face_batch, (160, 160), mode='bilinear', align_corners=False)
        orig_facenet_emb = facenet_model((orig_160 - 0.5) * 2.0).detach()
        
        orig_vgg = {k: v.detach() for k, v in vgg_extractor.extract(face_batch).items()}
        orig_resnet = resnet_extractor.extract(face_batch).detach()

    # ── Optimisation setup (TI-MI-FGSM) ─────────────────────────────────
    delta = (torch.randn_like(face_tensor) * 0.001).to(device) # Tiny initial noise to break symmetry
    delta.requires_grad = True
    momentum = torch.zeros_like(face_tensor, device=device)
    alpha = epsilon / (steps * 0.2) # Robust step size for sign-based FGSM
    decay = 1.0 # Momentum decay factor (mu)
    focus_mask = _build_focus_mask(
        face_tensor.shape[1],
        face_tensor.shape[2],
        device,
        face_tensor.dtype,
        getattr(config, 'delta_mask_floor', 0.45),
        getattr(config, 'delta_mask_power', 1.6),
    )
    lpips_budget = getattr(config, 'lpips_budget', 0.065)
    best_delta = None
    best_attack_score = float('inf')
    best_lpips = float('inf')
    fallback_delta = delta.detach().clone()
    fallback_total = float('inf')
    
    # Translation Invariance Kernel (Clustered Blobs)
    ti_kernel = T.GaussianBlur(kernel_size=5, sigma=1.5).to(device)

    print(f"  Running V3.1 TI-MI-FGSM Break ({steps} steps, config: ε={epsilon * 255:.1f}/255)")

    for step in tqdm(range(steps), desc="  MI-FGSM Optimization", ncols=80):
        if delta.grad is not None:
            delta.grad.zero_()
            
        protected = torch.clamp(face_tensor + (delta * focus_mask), 0.0, 1.0)
        prot_batch = protected.unsqueeze(0)

        # ── Visual Quality Loss (Strict) ──────────────────────────────
        lpips_val = lpips_model((prot_batch - 0.5) * 2.0, (face_batch - 0.5) * 2.0).mean()

        # ── Alignment-Aware EOT Loop ───────────────────────────────────
        # apply_eot() now contains Kornia Affine transforms simulating
        # FaceFusion's exact 5-point facial alignment mathematical warp.
        acc_arc_loss = 0.0
        acc_detector_loss = 0.0

        for _ in range(eot_samples):
            prot_eot = apply_eot(prot_batch)

            # Target 1: Destroy Identity! We target BOTH ArcFace (zero grads but kept for legacy) 
            # and FaceNet (very strong gradients)
            prot_arc_emb = arcface_model(prot_eot)
            
            # FaceNet requires 160x160 input
            prot_eot_160 = F.interpolate(prot_eot, (160, 160), mode='bilinear', align_corners=False)
            prot_facenet_emb = facenet_model((prot_eot_160 - 0.5) * 2.0)
            
            # Minimize cosine similarity (maximize distance), with extra emphasis
            # on ArcFace because it drives the swap identity embedding directly.
            arcface_cos = F.cosine_similarity(orig_arc_emb, prot_arc_emb, dim=1).mean()
            facenet_cos = F.cosine_similarity(orig_facenet_emb, prot_facenet_emb, dim=1).mean()
            acc_arc_loss += (
                arcface_focus * arcface_cos + facenet_focus * facenet_cos
            ) / max(arcface_focus + facenet_focus, 1e-6)

            # Target 2: Destroy YOLOv8 / RetinaFace (for target images)
            prot_feats = vgg_extractor.extract(prot_eot)
            for layer_name, orig_feat in orig_vgg.items():
                acc_detector_loss += F.cosine_similarity(orig_feat.flatten(1), prot_feats[layer_name].flatten(1), dim=1).mean()
            
            acc_detector_loss += F.cosine_similarity(orig_resnet, resnet_extractor.extract(prot_eot), dim=1).mean()

        acc_arc_loss /= max(eot_samples, 1)
        # Normalize detector loss by number of layers + resnet
        acc_detector_loss /= (eot_samples * 5)

        # ── V3.0 Final Loss ─────────────────────────────────────────────
        # We minimize arc_loss and detector_loss (pushes cosine sim to -1)
        loss = (lam_arc      * acc_arc_loss
              + lam_detector * acc_detector_loss
              + lam_perc     * lpips_val)

        attack_score = float((acc_arc_loss + 0.35 * acc_detector_loss).detach().item())
        lpips_scalar = float(lpips_val.detach().item())
        total_scalar = float(loss.detach().item())

        if total_scalar < fallback_total:
            fallback_total = total_scalar
            fallback_delta = delta.detach().clone()

        if (lpips_scalar <= lpips_budget and
                (attack_score < best_attack_score or
                 (attack_score == best_attack_score and lpips_scalar < best_lpips))):
            best_attack_score = attack_score
            best_lpips = lpips_scalar
            best_delta = delta.detach().clone()

        loss.backward()

        # ── Translation-Invariant Momentum Update (TI-MI-FGSM) ──────────
        with torch.no_grad():
            raw_grad = delta.grad.data
            
            # Apply TI smoothing to group gradients into indestructible blobs
            smoothed_grad = ti_kernel(raw_grad.unsqueeze(0)).squeeze(0)
            
            # L1 normalize gradient to stabilize momentum
            grad_l1_norm = torch.mean(torch.abs(smoothed_grad))
            normalized_grad = smoothed_grad / (grad_l1_norm + 1e-8)
            
            # Accumulate Momentum
            momentum = decay * momentum + normalized_grad
            
            # Update delta (Minimize loss -> step opposite to gradient)
            delta.data = delta.data - alpha * torch.sign(momentum)

            # Project into L∞ epsilon ball and [0,1] valid image range
            delta.data.clamp_(-epsilon, epsilon)
            valid = torch.clamp(face_tensor + delta.data, 0.0, 1.0)
            delta.data.copy_(valid - face_tensor)

    # ── Final smoothing ─────────────────────────────────────────────────
    with torch.no_grad():
        chosen_delta = best_delta if best_delta is not None else fallback_delta
        final_delta = T.GaussianBlur(
            kernel_size=getattr(config, 'final_blur_kernel', 3),
            sigma=getattr(config, 'final_blur_sigma', 1.2)
        )((chosen_delta * focus_mask).unsqueeze(0)).squeeze(0)
        final_delta = _desaturate_delta(
            final_delta,
            getattr(config, 'delta_chroma_scale', 0.25),
        )
        final_delta.clamp_(-epsilon, epsilon)
        protected_face = torch.clamp(face_tensor + final_delta, 0.0, 1.0)

    return protected_face
