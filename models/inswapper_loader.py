"""
DeepShield V3.0 — FaceFusion Inswapper Loader (Target Protection)

Downloads inswapper_128.onnx (the literal FaceFusion engine).
We convert it to a differentiable PyTorch model.
BY ATTACKING THE SWAPPER ITSELF, we break BOTH Source and Target manipulations.
"""

import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'model_cache', 'inswapper',
)

# Using huggingface mirror as it's more stable than the GitHub releases for 128
_INSWAPPER_URL = 'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx'

def _download_inswapper(cache_dir: str) -> str:
    """Download inswapper_128.onnx to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    onnx_path = os.path.join(cache_dir, 'inswapper_128.onnx')

    if not os.path.exists(onnx_path):
        print(f"    Downloading FaceFusion inswapper_128 (~550 MB) ...")
        # Added headers to handle basic 403/404 blocks from direct bots
        req = urllib.request.Request(_INSWAPPER_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(onnx_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"    [OK] Inswapper download complete")

    return onnx_path

class InswapperPyTorch(nn.Module):
    """
    Wraps the converted Inswapper model.
    FaceFusion uses Inswapper by feeding a 128x128 image and a 512-d ArcFace vector.
    We just need it to be differentiable so we can extract its UNet latent features.
    """
    def __init__(self, torch_model: nn.Module):
        super().__init__()
        self.swapper = torch_model

    def forward(self, target_img: torch.Tensor, identity_vector: torch.Tensor) -> torch.Tensor:
        """
        FaceFusion input format:
        target_img: (B, 3, 128, 128)
        identity_vector: (B, 512)
        """
        # Resize safely to 128x128 for the swapper network
        x = F.interpolate(target_img, size=(128, 128), mode='bilinear', align_corners=False)
        # FaceFusion typically uses [0, 1] for inswapper inputs
        return self.swapper(x, identity_vector)

def load_inswapper(device: torch.device, cache_dir: str = _DEFAULT_CACHE_DIR) -> nn.Module:
    """
    Load inswapper_128.onnx -> convert to PyTorch.
    Returns the raw PyTorch model because ONNX conversions of multi-input 
    UNets can be tricky and we only want to extract its intermediate gradients.
    """
    import onnx
    from onnx2torch import convert

    onnx_path = _download_inswapper(cache_dir)

    print("    Converting Inswapper ONNX -> PyTorch ...")
    onnx_model = onnx.load(onnx_path)
    
    # We convert it, but because INSWAPPER requires two inputs (Image, ArcFace Embedding),
    # we return the raw converted torch_model directly.
    torch_model = convert(onnx_model)
    torch_model.eval()
    torch_model.to(device)
    print("    [OK] Inswapper loaded (differentiable PyTorch)")

    return torch_model
