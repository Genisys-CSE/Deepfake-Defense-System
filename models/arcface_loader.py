"""
DeepShield — ArcFace Model Loader

Downloads InsightFace's buffalo_l model pack (which contains w600k_r50.onnx)
and converts it to a differentiable PyTorch model using onnx2torch.

This is THE model FaceFusion, DeepFaceLab, Roop, and virtually all modern
face-swap tools use for face recognition and alignment.  Attacking this model
directly is the key to making adversarial perturbations effective.

Input:  112×112 face crop, normalised to [0, 1]
Output: 512-D L2-normalised embedding
"""

import os
import zipfile
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F


# Where to cache the ONNX model
_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'model_cache', 'arcface',
)

# InsightFace buffalo_l model pack URL (GitHub releases)
_BUFFALO_L_URL = (
    'https://github.com/deepinsight/insightface/releases/download/'
    'v0.7/buffalo_l.zip'
)


def _download_arcface_onnx(cache_dir: str) -> str:
    """
    Download the buffalo_l.zip, extract w600k_r50.onnx.

    Returns the path to the ONNX file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    onnx_path = os.path.join(cache_dir, 'w600k_r50.onnx')

    if os.path.exists(onnx_path):
        return onnx_path

    zip_path = os.path.join(cache_dir, 'buffalo_l.zip')

    if not os.path.exists(zip_path):
        print(f"    Downloading ArcFace model pack (~325 MB ...")
        urllib.request.urlretrieve(_BUFFALO_L_URL, zip_path)
        print("    [OK] Download complete")

    print(f"    Extracting w600k_r50.onnx ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find the recognition model inside the zip
        for name in zf.namelist():
            if name.endswith('w600k_r50.onnx'):
                # Extract to cache_dir with flat name
                data = zf.read(name)
                with open(onnx_path, 'wb') as f:
                    f.write(data)
                print(f"    [OK] Extracted to {onnx_path}")
                break
        else:
            raise FileNotFoundError(
                "w600k_r50.onnx not found in buffalo_l.zip. "
                "Available files: " + str(zf.namelist())
            )

    # Clean up zip to save disk space
    try:
        os.remove(zip_path)
    except OSError:
        pass

    return onnx_path


class ArcFacePyTorch(nn.Module):
    """
    Wraps the onnx2torch-converted ArcFace model with correct
    preprocessing for 112×112 face inputs.

    Input:  Tensor (B, 3, H, W) with values in [0, 1]
    Output: Tensor (B, 512) — L2-normalised face embedding
    """

    def __init__(self, torch_model: nn.Module):
        super().__init__()
        self.backbone = torch_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ArcFace expects 112×112, normalised to [-1, 1]
        x = F.interpolate(x, size=(112, 112),
                          mode='bilinear', align_corners=False)
        x = (x - 0.5) / 0.5  # [0,1] → [-1,1]

        emb = self.backbone(x)

        # L2 normalise
        emb = F.normalize(emb, p=2, dim=1)
        return emb


def load_arcface(device: torch.device,
                 cache_dir: str = _DEFAULT_CACHE_DIR) -> ArcFacePyTorch:
    """
    Load ArcFace w600k_r50 as a differentiable PyTorch model.

    1. Downloads the ONNX model from InsightFace releases (first run only)
    2. Converts ONNX → PyTorch via onnx2torch
    3. Wraps in ArcFacePyTorch with correct preprocessing

    Returns
    -------
    model : ArcFacePyTorch — ready for forward pass + gradient computation
    """
    import onnx
    from onnx2torch import convert

    onnx_path = _download_arcface_onnx(cache_dir)

    print("    Converting ArcFace ONNX -> PyTorch ...")
    onnx_model = onnx.load(onnx_path)
    torch_model = convert(onnx_model)
    torch_model.eval()
    torch_model.to(device)

    model = ArcFacePyTorch(torch_model).to(device)
    model.eval()
    print("    [OK] ArcFace loaded (differentiable PyTorch)")

    return model
