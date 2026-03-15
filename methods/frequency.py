"""
DeepShield — DCT Frequency-Domain Perturbation

Adds structured perturbations in the DCT (Discrete Cosine Transform)
domain, targeting mid-to-high frequency bands.

Why DCT instead of FFT:
  JPEG compression uses DCT internally.  By adding perturbations that
  align with DCT coefficients, our noise survives JPEG recompression —
  critical since every social-media upload re-encodes images as JPEG.
  Plain FFT perturbations often get washed away by JPEG quantisation.

The perturbation is applied per-channel on 8×8 blocks (matching JPEG
block structure) for maximum robustness.
"""

import torch
import numpy as np
from scipy.fft import dctn, idctn


def _dct2(block):
    """2-D DCT of a single block (numpy)."""
    return dctn(block, type=2, norm='ortho')


def _idct2(block):
    """2-D inverse DCT of a single block (numpy)."""
    return idctn(block, type=2, norm='ortho')


def _create_freq_mask(block_h, block_w, radius):
    """
    Create a smooth mask that selects mid-to-high frequency DCT
    coefficients.  Low freqs (top-left corner) are left alone.

    Parameters
    ----------
    block_h, block_w : block dimensions  (typically 8)
    radius           : 0.0–1.0   (0 = all freqs, 1 = highest only)

    Returns
    -------
    mask : (block_h, block_w) float array in [0, 1]
    """
    Y, X = np.ogrid[:block_h, :block_w]
    max_dist = np.sqrt((block_h - 1) ** 2 + (block_w - 1) ** 2)
    dist = np.sqrt(Y ** 2 + X ** 2)

    cutoff = radius * max_dist
    # Smooth taper from 0 → 1 over a transition zone
    taper_width = max(0.1 * max_dist, 1.0)
    mask = np.clip((dist - cutoff) / taper_width, 0.0, 1.0)
    return mask.astype(np.float32)


def protect_frequency(face_tensor, device, config):
    """
    Apply DCT-domain frequency perturbation.

    Parameters
    ----------
    face_tensor : (C, H, W) in [0, 1], float32 on GPU
    device      : torch.device
    config      : ProtectionConfig

    Returns
    -------
    protected : (C, H, W) in [0, 1], same device
    """
    epsilon = config.freq_epsilon
    radius = config.freq_radius
    block_size = 8          # JPEG uses 8×8 blocks

    face_np = face_tensor.detach().cpu().numpy()       # (C, H, W)
    C, H, W = face_np.shape

    # Pad to multiple of block_size
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        face_np = np.pad(face_np, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

    _, H_pad, W_pad = face_np.shape
    freq_mask = _create_freq_mask(block_size, block_size, radius)

    protected_np = face_np.copy()

    for c in range(C):
        for y in range(0, H_pad, block_size):
            for x in range(0, W_pad, block_size):
                block = face_np[c, y:y + block_size, x:x + block_size]

                # DCT
                dct_block = _dct2(block)

                # Add structured perturbation to masked (mid-high) frequencies
                # Random magnitude within epsilon budget, random sign
                pert_mag = epsilon * np.random.uniform(0.3, 1.0, size=dct_block.shape)
                pert_sign = np.sign(np.random.randn(*dct_block.shape))
                perturbation = freq_mask * pert_mag * pert_sign

                dct_block_pert = dct_block + perturbation

                # Inverse DCT
                block_pert = _idct2(dct_block_pert)
                protected_np[c, y:y + block_size, x:x + block_size] = block_pert

    # Remove padding
    protected_np = protected_np[:, :H, :W]

    # Clamp the perturbation to the epsilon budget in pixel space
    orig_np = face_tensor.detach().cpu().numpy()
    delta = protected_np - orig_np
    delta = np.clip(delta, -epsilon, epsilon)
    protected_np = np.clip(orig_np + delta, 0.0, 1.0)

    # Smooth block boundary artifacts with a light Gaussian blur
    import cv2
    for c in range(C):
        protected_np[c] = cv2.GaussianBlur(
            protected_np[c], (3, 3), sigmaX=0.5
        )

    # Re-clamp after blur
    delta = protected_np - orig_np
    delta = np.clip(delta, -epsilon, epsilon)
    protected_np = np.clip(orig_np + delta, 0.0, 1.0)

    return torch.from_numpy(protected_np).float().to(device)
