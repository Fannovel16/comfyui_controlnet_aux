import torch
import torch.nn as nn
from custom_controlnet_aux.metric3d.mono.utils.comm import get_func

class DensePredModel(nn.Module):
    def __init__(self, cfg) -> None:
        super(DensePredModel, self).__init__()

        self.encoder = get_func('custom_controlnet_aux.metric3d.mono.model.' + cfg.model.backbone.prefix + cfg.model.backbone.type)(**cfg.model.backbone)
        self.decoder = get_func('custom_controlnet_aux.metric3d.mono.model.' + cfg.model.decode_head.prefix + cfg.model.decode_head.type)(cfg)

    def forward(self, input, **kwargs):
        # [f_32, f_16, f_8, f_4]
        features = self.encoder(input)
        out = self.decoder(features, **kwargs)
        return out