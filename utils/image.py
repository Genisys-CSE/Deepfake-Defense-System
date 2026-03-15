"""
DeepShield — Image I/O and Tensor Utilities
"""

import os
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image


def load_image(path):
    """Load an image as a PIL RGB Image."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert('RGB')


def image_to_tensor(img_pil, device='cpu'):
    """PIL Image → (C, H, W) float32 tensor in [0, 1]."""
    return T.ToTensor()(img_pil).to(device)


def tensor_to_numpy(tensor):
    """(C, H, W) tensor in [0, 1] → (H, W, C) uint8 numpy array."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return (arr * 255).clip(0, 255).astype(np.uint8)


def save_image(img_pil, path):
    """Save a PIL image, creating parent dirs if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    img_pil.save(path)
    print(f"  ✓ Saved to {path}")
