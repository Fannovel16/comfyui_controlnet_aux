import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        basemodel_name = 'tf_efficientnet_b5'
        print('Loading base model ()...'.format(basemodel_name), end='')
        try:
            from custom_timm import create_model
        except ImportError:
            from src.custom_timm import create_model
        basemodel = create_model(basemodel_name, pretrained=False, num_classes=0)
        print('Done.')


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


