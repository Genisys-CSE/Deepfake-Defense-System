"""
DeepShield — Multi-Model Ensemble Adversarial Attack (PGD + EOT)

This is the core of the protection system.  It runs Projected Gradient
Descent with Expectation Over Transformation to craft perturbations that
simultaneously disrupt:

  1. Identity embeddings   (FaceNet × 2 models)
  2. CNN feature maps      (VGG19 multi-layer + ResNet50)
  3. Texture statistics    (gram matrices from VGG19)
  4. While preserving      visual quality (LPIPS constraint)

Key improvements over the prototype:
  - Multi-model ensemble for better transfer to unknown deepfake models
  - Multi-layer features (not just final embedding)
  - Cosine similarity loss (vs MSE) for meaningful gradients
  - Gram-matrix texture disruption
  - Cosine-annealing LR schedule
  - Smoothing applied only at end (never during optimisation)
  - Enhanced EOT with JPEG, color jitter, blur
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

from methods.transforms import apply_eot
from methods.texture import texture_loss
from models.feature_extractor import (
    MultiLayerFeatureExtractor, ResNetFeatureExtractor, MobileNetFeatureExtractor
)


def protect_adversarial(face_tensor, device, config, model_loader):
    """
    Run the multi-model ensemble adversarial attack.

    Parameters
    ----------
    face_tensor  : (C, H, W) in [0, 1]
    device       : torch.device
    config       : ProtectionConfig
    model_loader : ModelLoader instance

    Returns
    -------
    protected : (C, H, W) in [0, 1], clamped
    """
    epsilon        = config.epsilon
    steps          = config.steps
    lr             = config.lr
    eot_samples    = config.eot_samples
    lam_id         = config.lambda_identity
    lam_feat       = config.lambda_feature
    lam_tex        = config.lambda_texture
    lam_perc       = config.lambda_perceptual
    use_texture    = config.use_texture

    face_tensor = face_tensor.to(device)

    # ── Load models ─────────────────────────────────────────────────────
    identity_models = []
    for name in config.identity_models:
        identity_models.append(model_loader.get(name))

    surrogate_extractors = []
    vgg_extractor = None

    for name in config.surrogate_models:
        model = model_loader.get(name)
        if name == 'vgg19':
            vgg_extractor = MultiLayerFeatureExtractor(model, config.vgg_layers)
            surrogate_extractors.append(('vgg19', vgg_extractor))
        elif name == 'resnet50':
            resnet_extractor = ResNetFeatureExtractor(model)
            surrogate_extractors.append(('resnet50', resnet_extractor))
        elif name == 'mobilenet_v2':
            mobile_extractor = MobileNetFeatureExtractor(model)
            surrogate_extractors.append(('mobilenet_v2', mobile_extractor))

    lpips_model = model_loader.get('lpips')

    # ── Compute original (clean) embeddings ─────────────────────────────
    with torch.no_grad():
        face_batch = face_tensor.unsqueeze(0)

        # Identity embeddings
        orig_id_embeddings = []
        for id_model in identity_models:
            face_fn = F.interpolate(face_batch, size=(160, 160),
                                    mode='bilinear', align_corners=False)
            face_fn_norm = (face_fn - 0.5) * 2.0
            emb = id_model(face_fn_norm).detach()
            orig_id_embeddings.append(emb)

        # Surrogate features
        orig_surrogate_features = {}
        for name, extractor in surrogate_extractors:
            if name == 'vgg19':
                feats = extractor.extract(face_batch)
                orig_surrogate_features['vgg19'] = {
                    k: v.detach() for k, v in feats.items()
                }
            elif name in ('resnet50', 'mobilenet_v2'):
                feat = extractor.extract(face_batch).detach()
                orig_surrogate_features[name] = feat

    # ── Optimisation setup ──────────────────────────────────────────────
    delta = torch.zeros_like(face_tensor, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * 0.1
    )

    # ── PGD Loop ────────────────────────────────────────────────────────
    print(f"  Running ensemble PGD+EOT  ({steps} steps, {eot_samples} EOT, "
          f"ε={epsilon * 255:.1f}/255)")

    for step in tqdm(range(steps), desc="  DeepShield PGD", ncols=80):
        optimizer.zero_grad()

        acc_id_loss   = 0.0
        acc_feat_loss = 0.0
        acc_tex_loss  = 0.0
        acc_perc_loss = 0.0

        for _ in range(eot_samples):
            protected = torch.clamp(face_tensor + delta, 0.0, 1.0)
            prot_batch = protected.unsqueeze(0)

            # Apply EOT augmentation
            prot_eot = apply_eot(prot_batch)

            # ── Identity loss (cosine similarity → minimise) ────────────
            for i, id_model in enumerate(identity_models):
                prot_fn = F.interpolate(prot_eot, size=(160, 160),
                                        mode='bilinear', align_corners=False)
                prot_fn_norm = (prot_fn - 0.5) * 2.0
                prot_emb = id_model(prot_fn_norm)
                id_sim = F.cosine_similarity(
                    orig_id_embeddings[i], prot_emb, dim=1
                ).mean()
                acc_id_loss += id_sim

            # ── Feature loss (cosine similarity → minimise) ─────────────
            for name, extractor in surrogate_extractors:
                if name == 'vgg19':
                    prot_feats = extractor.extract(prot_eot)
                    for layer_name, orig_feat in orig_surrogate_features['vgg19'].items():
                        prot_feat = prot_feats[layer_name]
                        # Flatten spatial dims for cosine sim
                        o = orig_feat.flatten(1)
                        p = prot_feat.flatten(1)
                        feat_sim = F.cosine_similarity(o, p, dim=1).mean()
                        acc_feat_loss += feat_sim

                    # ── Texture loss (gram matrix → maximise distance) ──
                    if use_texture:
                        tex = texture_loss(
                            orig_surrogate_features['vgg19'], prot_feats
                        )
                        acc_tex_loss += tex

                elif name in ('resnet50', 'mobilenet_v2'):
                    prot_feat = extractor.extract(prot_eot)
                    orig_feat = orig_surrogate_features[name]
                    feat_sim = F.cosine_similarity(
                        orig_feat, prot_feat, dim=1
                    ).mean()
                    acc_feat_loss += feat_sim

            # ── Perceptual loss (LPIPS → keep visually similar) ─────────
            lpips_val = lpips_model(
                (protected.unsqueeze(0) - 0.5) * 2.0,
                (face_tensor.unsqueeze(0) - 0.5) * 2.0
            ).mean()
            acc_perc_loss += lpips_val

        # Average over EOT samples
        n = eot_samples
        n_id = eot_samples * len(identity_models)
        # Feature losses are accumulated across layers and models
        # We just normalise by EOT samples
        acc_id_loss   /= n_id
        acc_feat_loss /= n
        acc_tex_loss  /= n
        acc_perc_loss /= n

        # ── Combined loss ───────────────────────────────────────────────
        # Minimise identity sim + feature sim + texture_loss + perceptual
        loss = (lam_id   * acc_id_loss
              + lam_feat * acc_feat_loss
              + lam_tex  * acc_tex_loss
              + lam_perc * acc_perc_loss)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # ── Project delta into L∞ ball ──────────────────────────────────
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            valid = torch.clamp(face_tensor + delta.data, 0.0, 1.0)
            delta.data = valid - face_tensor

    # ── Final smoothing (only here — never during optimisation) ─────────
    with torch.no_grad():
        final_delta = T.GaussianBlur(
            kernel_size=config.final_blur_kernel,
            sigma=config.final_blur_sigma
        )(delta.data.unsqueeze(0)).squeeze(0)
        final_delta = torch.clamp(final_delta, -epsilon, epsilon)
        protected_face = torch.clamp(face_tensor + final_delta, 0.0, 1.0)

    return protected_face
