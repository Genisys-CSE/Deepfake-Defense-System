"""
DeepShield — Discrete Wavelet Transform (DWT) Attack

Implements frequency-domain perturbation using a native, differentiable
2D Haar Wavelet Transform in PyTorch.

By injecting adversarial noise specifically into the LL (Low-Low) subband,
we create perturbations that survive low-pass filtering and generative
reconstruction (like GFPGAN), which typically discard high-frequency details.
"""

import torch
import torch.nn.functional as F


def _build_ll_focus_mask(height: int, width: int,
                         device: torch.device,
                         dtype: torch.dtype) -> torch.Tensor:
    """Favor central facial structure and reduce border visibility."""
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    radial = torch.sqrt((xx / 0.9) ** 2 + ((yy + 0.05) / 1.05) ** 2)
    mask = torch.clamp(1.0 - radial, min=0.0).pow(1.5)
    return (0.4 + 0.6 * mask).unsqueeze(0).unsqueeze(0)


def haar_dwt2d(x: torch.Tensor):
    """
    Computes the 2D Haar Discrete Wavelet Transform.
    
    Parameters
    ----------
    x : Tensor (B, C, H, W)
    
    Returns
    -------
    ll, lh, hl, hh : Four Tensors of shape (B, C, H//2, W//2)
    """
    # Ensure dimensions are divisible by 2 for the transform
    B, C, H, W = x.shape
    if H % 2 != 0 or W % 2 != 0:
        x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
        
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    
    # Haar filters (normalized by 2)
    ll = (x00 + x01 + x10 + x11) / 2.0
    lh = (x00 - x01 + x10 - x11) / 2.0
    hl = (x00 + x01 - x10 - x11) / 2.0
    hh = (x00 - x01 - x10 + x11) / 2.0
    
    return ll, lh, hl, hh

def haar_idwt2d(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """
    Computes the 2D Inverse Haar Discrete Wavelet Transform.
    
    Parameters
    ----------
    ll, lh, hl, hh : Four Tensors of shape (B, C, H//2, W//2)
    original_shape : Tuple (B, C, H, W) for optional unpadding
    
    Returns
    -------
    out : Tensor (B, C, H, W)
    """
    B, C, h, w = ll.shape
    out = torch.zeros((B, C, h * 2, w * 2), device=ll.device, dtype=ll.dtype)
    
    out[:, :, 0::2, 0::2] = (ll + lh + hl + hh) / 2.0
    out[:, :, 0::2, 1::2] = (ll - lh + hl - hh) / 2.0
    out[:, :, 1::2, 0::2] = (ll + lh - hl - hh) / 2.0
    out[:, :, 1::2, 1::2] = (ll - lh - hl + hh) / 2.0
    
    # Trim padding if original dimensions were odd
    _, _, OH, OW = original_shape
    return out[:, :, :OH, :OW]

def protect_frequency(face_tensor: torch.Tensor, epsilon: float = 3.0 / 255.0) -> torch.Tensor:
    """
    Apply DWT-based static noise to the low-frequency band.
    
    This acts as a base layer of robust noise before the main PGD
    adversarial attack runs.
    
    Parameters
    ----------
    face_tensor : Tensor of shape (C, H, W) in [0, 1].
    epsilon     : Magnitude of the injected noise.
    
    Returns
    -------
    protected : Tensor of shape (C, H, W) in [0, 1].
    """
    if epsilon <= 0:
        return face_tensor
        
    device = face_tensor.device
    x = face_tensor.unsqueeze(0)  # (1, C, H, W)
    
    # 1. Forward DWT
    ll, lh, hl, hh = haar_dwt2d(x)
    
    # 2. Inject smooth shared noise into the LL band.
    # A single low-frequency luminance-biased map is much less visible
    # than per-channel sign noise while still surviving swap preprocessing.
    noise = torch.randn(
        (ll.shape[0], 1, ll.shape[2], ll.shape[3]),
        device=device,
        dtype=ll.dtype,
    )
    noise = F.avg_pool2d(noise, kernel_size=5, stride=1, padding=2)
    noise = noise / (noise.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)
    noise = noise.repeat(1, ll.shape[1], 1, 1)
    noise = noise * _build_ll_focus_mask(ll.shape[2], ll.shape[3], device, ll.dtype)
    noise = noise * epsilon
    ll = ll + noise
    
    # 3. Inverse DWT
    protected_x = haar_idwt2d(ll, lh, hl, hh, x.shape)
    
    protected_tensor = torch.clamp(protected_x.squeeze(0), 0.0, 1.0)
    return protected_tensor
