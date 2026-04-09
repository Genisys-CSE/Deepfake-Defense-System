"""
DeepShield — Gram-Matrix Texture Disruption

Computes gram matrices from VGG19 feature maps and provides a loss that
maximises the difference in texture statistics between the original and
perturbed images.

Why this works
--------------
Deepfake generators (GANs, autoencoders, diffusion models) rely on
texture / style consistency during face rendering.  The gram matrix
captures the correlation structure of CNN feature maps — the same
representation used in Neural Style Transfer.  By disrupting these
statistics we directly poison the texture-rendering pipeline of the
deepfake model.
"""

import torch
import torch.nn.functional as F


def gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    """
    Compute the (normalised) Gram matrix of a feature map.

    Parameters
    ----------
    feature_map : Tensor of shape (B, C, H, W).

    Returns
    -------
    gram : Tensor of shape (B, C, C), normalised by spatial size.
    """
    B, C, H, W = feature_map.shape
    features = feature_map.view(B, C, H * W)               # (B, C, N)
    gram = torch.bmm(features, features.transpose(1, 2))    # (B, C, C)
    return gram / (C * H * W)


def texture_loss(orig_features: dict, pert_features: dict) -> torch.Tensor:
    """
    Gram-matrix texture disruption loss.

    Returns the **negative** mean Frobenius distance between the gram
    matrices of each layer.  Minimising this loss pushes the textures
    apart (maximises gram-matrix divergence).

    Parameters
    ----------
    orig_features : dict of {layer_name: Tensor (B, C, H, W)}
    pert_features : dict of {layer_name: Tensor (B, C, H, W)}

    Returns
    -------
    loss : scalar Tensor (negative; minimise = maximise gram distance).
    """
    total = torch.tensor(0.0, device=next(iter(orig_features.values())).device)
    n_layers = 0

    for layer_name in orig_features:
        if layer_name not in pert_features:
            continue
        orig_gram = gram_matrix(orig_features[layer_name])
        pert_gram = gram_matrix(pert_features[layer_name])

        # negative Frobenius distance → minimise loss = maximise distance
        dist = F.mse_loss(pert_gram, orig_gram)
        total = total - dist
        n_layers += 1

    if n_layers == 0:
        return total

    return total / n_layers
