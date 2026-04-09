"""
DeepShield — Configuration & Presets

Centralized configuration for the anti-deepfake protection system.
DeepShield V2.1 — Final Active Strategy:
  ArcFace + Landmark Sabotage + DWT Low-Freq + GFPGAN Proxy
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ProtectionConfig:
    """Full configuration for the protection pipeline."""

    # ── Core attack parameters ──────────────────────────────────────────
    epsilon: float = 4.0 / 255.0          # L-inf perturbation bound
    steps: int = 300                       # PGD optimisation steps
    lr: float = 0.005                      # Adam learning rate
    eot_samples: int = 6                   # EOT augmentations per step

    # ── Final V3.0 Loss Weights ─────────────────────────────────────────
    lambda_arcface: float = 6.0            # Main target (ArcFace)
    lambda_landmark: float = 5.0           # Redirected to YOLO/RetinaFace backbones
    lambda_perceptual: float = 12.0        # Visual quality (LPIPS)
    arcface_focus: float = 1.0
    facenet_focus: float = 1.0

    # ── Proxy Settings ──────────────────────────────────────────────────
    proxy_sigma: float = 1.2              # Proxy blur sigma (matches GFPGAN)

    # ── Frequency-domain (DWT) attack ───────────────────────────────────
    # In V2.1, this applies noise to the DWT LL subband
    freq_epsilon: float = 3.0 / 255.0     

    # ── Trace Ensemble models ───────────────────────────────────────────
    surrogate_models: List[str] = field(
        default_factory=lambda: ['resnet50', 'vgg19']
    )
    identity_models: List[str] = field(
        default_factory=lambda: ['facenet_vggface2', 'facenet_casia']
    )
    vgg_layers: List[str] = field(
        default_factory=lambda: ['relu4_4']
    )

    # ── Final output ────────────────────────────────────────────────────
    final_blur_kernel: int = 3
    final_blur_sigma: float = 0.6
    lpips_budget: float = 0.065
    delta_chroma_scale: float = 0.25
    delta_mask_floor: float = 0.45
    delta_mask_power: float = 1.6
    face_margin: int = 15

    # ── Method toggles (mostly legacy, kept active for full pipeline) ───
    use_adversarial: bool = True
    use_frequency: bool = True
    use_texture: bool = False # Deprecated in V2.1 formula


# ═══════════════════════════════════════════════════════════════════════
#  Presets
# ═══════════════════════════════════════════════════════════════════════

PRESETS = {
    # ── V3.0 Maximum ────────────────────────────────────────────────────
    'maximum': ProtectionConfig(
        epsilon=3.0 / 255.0,
        steps=300,
        lr=0.005,
        eot_samples=6,
        
        # V3.0 Formula
        lambda_arcface=6.0,
        lambda_landmark=5.0,
        lambda_perceptual=12.0,

        freq_epsilon=2.0 / 255.0,
        final_blur_sigma=1.2,
        surrogate_models=['resnet50', 'vgg19'],
        vgg_layers=['relu4_4'],
    ),

    # ── NUCLEAR OPTION (Guaranteed Break) ───────────────────────────────
    # For when invisible noise fails against robust affine warps. 
    # This will introduce highly visible grain, but will 100% shatter
    # the deepfake pipeline for demonstration purposes.
    'nuclear': ProtectionConfig(
        epsilon=48.0 / 255.0,     # Massive visible noise
        steps=200,
        lr=0.05,
        eot_samples=10,
        
        lambda_arcface=20.0,
        lambda_landmark=20.0,
        lambda_perceptual=0.5,    # We don't care about invisibility anymore

        freq_epsilon=20.0 / 255.0,
        final_blur_kernel=5,
        final_blur_sigma=2.0,
        surrogate_models=['resnet50', 'vgg19'],
        vgg_layers=['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4'],
    ),

    # ── Balanced ────────────────────────────────────────────────────────
    # Tuned for Viva demo: drops FaceNet/ArcFace heavily in ~25 seconds.
    'balanced': ProtectionConfig(
        epsilon=10.5 / 255.0,
        steps=105,
        lr=0.011,
        eot_samples=5,
        
        lambda_arcface=17.5, # Push a little harder on ArcFace-driven swaps
        lambda_landmark=3.0,
        lambda_perceptual=5.2,
        arcface_focus=1.7,
        facenet_focus=0.8,

        freq_epsilon=2.0 / 255.0,
        final_blur_kernel=5,
        final_blur_sigma=1.12,
        lpips_budget=0.075,
        delta_chroma_scale=0.10,
        delta_mask_floor=0.36,
        delta_mask_power=1.7,
        surrogate_models=['resnet50', 'vgg19'],
        vgg_layers=['relu2_2', 'relu3_4', 'relu4_4'],
    ),

    # ── Stealth ─────────────────────────────────────────────────────────
    # Ultra-invisible, strictly for very simple target systems.
    'stealth': ProtectionConfig(
        epsilon=2.5 / 255.0,
        steps=150,
        lr=0.004,
        eot_samples=4,
        
        lambda_arcface=3.0,
        lambda_landmark=3.0,
        lambda_perceptual=15.0,

        freq_epsilon=1.0 / 255.0,
        surrogate_models=[],
    ),
}
