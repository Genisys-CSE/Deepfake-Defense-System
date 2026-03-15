"""
DeepShield — Gram-Matrix Texture Disruption

Computes gram matrices from VGG19 feature maps and provides a loss
function that maximises the difference in texture statistics between
the original and perturbed images.

Why this works:
  Deepfake generators (GANs, autoencoders) rely on texture/style
  consistency during face rendering.  The gram matrix captures the
  correlation structure of CNN feature maps — the same representation
  used in Neural Style Transfer.  By disrupting gram-matrix statistics
  we directly attack the texture-rendering pipeline.
"""

import torch
import torch.nn.functional as F


def gram_matrix(feature_map):
    """
    Compute the (normalised) Gram matrix of a feature map.

    Parameters
    ----------
    feature_map : (B, C, H, W)

    Returns
    -------
    gram : (B, C, C)   normalised by spatial size
    """
    B, C, H, W = feature_map.shape
    features = feature_map.view(B, C, H * W)                # (B, C, N)
    gram = torch.bmm(features, features.transpose(1, 2))     # (B, C, C)
    return gram / (C * H * W)                                 # normalise


def texture_loss(orig_features, pert_features):
    """
    Compute the gram-matrix texture disruption loss.

    We want to MAXIMISE the difference between gram matrices, so we return
    the *negative* Frobenius distance.  Minimising this loss pushes the
    gram matrices apart.

    Parameters
    ----------
    orig_features : dict[str → Tensor]  layer_name → (B, C, H, W)
    pert_features : dict[str → Tensor]  layer_name → (B, C, H, W)

    Returns
    -------
    loss : scalar   (negative; minimise to maximise gram distance)
    """
    total = 0.0
    n_layers = 0

    for layer_name in orig_features:
        if layer_name not in pert_features:
            continue
        orig_gram = gram_matrix(orig_features[layer_name])
        pert_gram = gram_matrix(pert_features[layer_name])

        # Frobenius distance (sum of squared differences)
        dist = F.mse_loss(pert_gram, orig_gram)
        total += -dist          # negative → minimise loss = maximise distance
        n_layers += 1

    if n_layers == 0:
        return torch.tensor(0.0)

    return total / n_layers
