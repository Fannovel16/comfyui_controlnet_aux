"""
Standalone DSINE Encoder that bypasses ComfyUI import conflicts.
This module provides a clean EfficientNet-B5 encoder without any problematic torch.hub dependencies.
"""
import torch
import torch.nn as nn
import timm

INPUT_CHANNELS_DICT = {
    0: [1280, 112, 40, 24, 16],
    1: [1280, 112, 40, 24, 16],
    2: [1408, 120, 48, 24, 16],
    3: [1536, 136, 48, 32, 24],
    4: [1792, 160, 56, 32, 24],
    5: [2048, 176, 64, None, None],  # Modern timm EfficientNet-B5: features[10,7,5]
    6: [2304, 200, 72, 40, 32],
    7: [2560, 224, 80, 48, 32]
}

class StandaloneEncoder(nn.Module):
    """
    Standalone EfficientNet encoder that works reliably in PyTorch 2.7 without hanging.
    This bypasses all problematic torch.hub imports and ComfyUI conflicts.
    """
    def __init__(self, B=5, pretrained=True):
        super(StandaloneEncoder, self).__init__()

        basemodel_name = f'tf_efficientnet_b{B}.ap_in1k'
        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = timm.create_model(basemodel_name, pretrained=False, num_classes=0)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features