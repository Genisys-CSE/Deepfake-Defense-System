"""
DeepShield — Configuration & Presets

Centralized configuration for the anti-deepfake protection system.
Three presets: maximum, balanced, stealth.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ProtectionConfig:
    """Full configuration for the protection pipeline."""

    # ── Core attack parameters ──────────────────────────────────────────
    epsilon: float = 4.0 / 255.0          # L-inf perturbation bound (pixel space)
    steps: int = 200                       # PGD optimisation steps
    lr: float = 0.005                      # Adam learning rate
    eot_samples: int = 6                   # EOT augmentations per step

    # ── Loss weights ────────────────────────────────────────────────────
    lambda_identity: float = 1.0           # Identity-embedding disruption
    lambda_feature: float = 5.0            # Multi-layer feature disruption
    lambda_texture: float = 2.0            # Gram-matrix texture disruption
    lambda_perceptual: float = 20.0        # LPIPS perceptual constraint

    # ── Frequency-domain attack ─────────────────────────────────────────
    freq_epsilon: float = 3.0 / 255.0     # Frequency perturbation budget
    freq_radius: float = 0.25             # High-freq band radius (0→all, 1→highest)
    use_dct: bool = True                   # Use DCT (JPEG-robust) vs raw FFT

    # ── Model ensemble ──────────────────────────────────────────────────
    surrogate_models: List[str] = field(
        default_factory=lambda: ['resnet50', 'vgg19']
    )
    identity_models: List[str] = field(
        default_factory=lambda: ['facenet_vggface2', 'facenet_casia']
    )

    # VGG19 layers for multi-layer feature extraction
    vgg_layers: List[str] = field(
        default_factory=lambda: ['relu2_2', 'relu3_4', 'relu4_4']
    )

    # ── Final output polish ─────────────────────────────────────────────
    final_blur_kernel: int = 3
    final_blur_sigma: float = 0.6

    # ── Face detection ──────────────────────────────────────────────────
    face_margin: int = 15

    # ── Method toggles ──────────────────────────────────────────────────
    use_adversarial: bool = True
    use_frequency: bool = True
    use_texture: bool = True


# ═══════════════════════════════════════════════════════════════════════
#  Presets
# ═══════════════════════════════════════════════════════════════════════

PRESETS = {
    # ── Maximum disruption ──────────────────────────────────────────────
    # Slight visible artifacts acceptable.  Targets every feature space
    # with high budget.  Best for images you want to make deepfake-proof.
    'maximum': ProtectionConfig(
        epsilon=6.0 / 255.0,
        steps=300,
        lr=0.008,
        eot_samples=8,
        lambda_identity=1.5,
        lambda_feature=8.0,
        lambda_texture=3.0,
        lambda_perceptual=12.0,
        freq_epsilon=4.0 / 255.0,
        surrogate_models=['resnet50', 'vgg19', 'mobilenet_v2'],
        identity_models=['facenet_vggface2', 'facenet_casia'],
        use_adversarial=True,
        use_frequency=True,
        use_texture=True,
    ),

    # ── Balanced (default) ──────────────────────────────────────────────
    # Good disruption with minimal visual impact.  Recommended for most
    # use-cases (social media uploads, profile photos).
    'balanced': ProtectionConfig(
        epsilon=4.0 / 255.0,
        steps=200,
        lr=0.005,
        eot_samples=6,
        lambda_identity=1.0,
        lambda_feature=5.0,
        lambda_texture=2.0,
        lambda_perceptual=30.0,        # stronger visual constraint
        freq_epsilon=1.5 / 255.0,      # gentler frequency noise
        final_blur_kernel=3,
        final_blur_sigma=0.8,          # smoother final output
        surrogate_models=['resnet50', 'vgg19', 'mobilenet_v2'],
        identity_models=['facenet_vggface2', 'facenet_casia'],
        use_adversarial=True,
        use_frequency=True,
        use_texture=True,
    ),

    # ── Stealth ─────────────────────────────────────────────────────────
    # Invisible perturbation.  Lower epsilon & stronger perceptual
    # constraint.  Effective against basic deepfake tools.
    'stealth': ProtectionConfig(
        epsilon=2.0 / 255.0,
        steps=150,
        lr=0.003,
        eot_samples=4,
        lambda_identity=0.8,
        lambda_feature=3.0,
        lambda_texture=1.0,
        lambda_perceptual=35.0,
        freq_epsilon=1.5 / 255.0,
        surrogate_models=['resnet50', 'mobilenet_v2'],
        identity_models=['facenet_vggface2'],
        use_adversarial=True,
        use_frequency=True,
        use_texture=False,
    ),
}
