"""
DeepShield V3.0 — The "Anti-Inswapper" EOT Pipeline

Implements severe Alignment-Aware EOT using Kornia.
FaceFusion uses 5-point facial landmarks to calculate an Affine Transform
matrix and warps the face to exactly 112x112 or 128x128. Simple random
resizing (like Differentiable Augmentation) fails against this.

We simulate FaceFusion's exact Affine Warp here so the adversarial
noise survives the actual face swap preprocessing.
"""

import torch
import torch.nn.functional as F
import kornia.augmentation as K

class AlignmentAwareEOT(torch.nn.Module):
    """
    Simulates FaceFusion / InsightFace alignment.
    Uses Kornia for highly differentiable spatial transforms.
    """
    def __init__(self):
        super().__init__()
        # FaceFusion typically applies a similarity transform (rotation + scale + translation)
        # We aggressively simulate all possible affine variances to make the noise bulletproof.
        self.affine = K.RandomAffine(
            degrees=[-15.0, 15.0],        # Face roll
            translate=[0.05, 0.05],       # Face bounding box shift
            scale=[0.85, 1.15],           # Scale estimation errors
            shear=[-5.0, 5.0],            # Minor perspective distortion
            p=1.0,
            keepdim=True
        )
        
        # Simulates CodeFormer / GFPGAN resizing artifacts
        self.blur = K.RandomGaussianBlur(
            kernel_size=(3, 3), 
            sigma=(0.1, 1.5), 
            p=0.5,
            keepdim=True
        )

        # Simulates internal Deepfake Generator latent space noise compression
        self.noise = K.RandomGaussianNoise(
            mean=0.0, std=0.02, p=0.3, keepdim=True
        )
        
    def forward(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the FaceFusion-simulated pipeline.
        face_tensor: (1, 3, H, W)
        """
        x = self.affine(face_tensor)
        x = self.blur(x)
        x = self.noise(x)
        
        # Simulate InsightFace resizing to 128x128 (inswapper resolution)
        # then back to 112x112 (arcface resolution)
        x_128 = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        x_112 = F.interpolate(x_128, size=(112, 112), mode='bilinear', align_corners=False)
        
        return torch.clamp(x_112, 0.0, 1.0)


def apply_eot(face_tensor: torch.Tensor) -> torch.Tensor:
    """Wrapper function to align with the rest of the pipeline."""
    # Initialize only once per process usually, but doing it here for simplicity
    eot_module = AlignmentAwareEOT().to(face_tensor.device)
    return eot_module(face_tensor)
