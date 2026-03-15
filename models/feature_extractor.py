"""
DeepShield — Multi-Layer Feature Extractor

Extracts features from multiple intermediate layers of VGG19 simultaneously.
This enables:
  1. Multi-layer feature disruption (attack different abstraction levels)
  2. Gram-matrix texture disruption (style transfer attack)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


# VGG19 layer name → sequential-index mapping (after each ReLU)
VGG19_LAYER_MAP = {
    'relu1_1':  1,  'relu1_2':  3,
    'relu2_1':  6,  'relu2_2':  8,
    'relu3_1': 11,  'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
    'relu4_1': 20,  'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
    'relu5_1': 29,  'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class MultiLayerFeatureExtractor:
    """
    Run a forward pass through VGG19 and collect activations at specified layers.

    Parameters
    ----------
    vgg_model   : nn.Module   (models.vgg19(...).features, already on device)
    layer_names : list[str]   e.g. ['relu2_2', 'relu3_4', 'relu4_4']
    """

    def __init__(self, vgg_model, layer_names):
        self.model = vgg_model
        self.layer_indices = {}
        for name in layer_names:
            if name not in VGG19_LAYER_MAP:
                raise ValueError(
                    f"Unknown VGG19 layer '{name}'. "
                    f"Available: {list(VGG19_LAYER_MAP.keys())}"
                )
            self.layer_indices[name] = VGG19_LAYER_MAP[name]

        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self._max_idx = max(self.layer_indices.values())

    def extract(self, x, target_size=224):
        """
        Extract multi-layer features.

        Parameters
        ----------
        x           : (B, 3, H, W) in [0, 1]
        target_size : resize to this before feeding to VGG

        Returns
        -------
        features : dict[str → Tensor]   layer_name → (B, C, H', W')
        """
        x = F.interpolate(x, size=(target_size, target_size),
                          mode='bilinear', align_corners=False)
        x = self.normalize(x)

        features = {}
        h = x
        for i, layer in enumerate(self.model):
            h = layer(h)
            for name, idx in self.layer_indices.items():
                if i == idx:
                    features[name] = h
            if i >= self._max_idx:
                break

        return features


class ResNetFeatureExtractor:
    """
    Extract the final feature vector from a truncated ResNet50.
    """

    def __init__(self, resnet_model):
        self.model = resnet_model
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def extract(self, x, target_size=224):
        """
        Parameters
        ----------
        x : (B, 3, H, W) in [0, 1]

        Returns
        -------
        feat : (B, 2048) flattened feature vector
        """
        x = F.interpolate(x, size=(target_size, target_size),
                          mode='bilinear', align_corners=False)
        x = self.normalize(x)
        return self.model(x).flatten(1)


class MobileNetFeatureExtractor:
    """
    Extract features from a truncated MobileNetV2.
    Web faceswap tools often use mobile-architecture encoders.
    """

    def __init__(self, mobilenet_model):
        self.model = mobilenet_model
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def extract(self, x, target_size=224):
        x = F.interpolate(x, size=(target_size, target_size),
                          mode='bilinear', align_corners=False)
        x = self.normalize(x)
        return self.model(x).flatten(1)
