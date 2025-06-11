import torch
import torch.nn as nn
from ...utils.comm import get_func

class DensePredModel(nn.Module):
    def __init__(self, cfg) -> None:
        super(DensePredModel, self).__init__()

        # Use direct imports instead of get_func to avoid module resolution issues
        
        # Handle different backbone types
        backbone_type = cfg.model.backbone.type
        if backbone_type == 'vit_small_reg':
            from ..backbones.ViT_DINO_reg import vit_small_reg
            self.encoder = vit_small_reg(**cfg.model.backbone)
        elif backbone_type == 'vit_large_reg':
            from ..backbones.ViT_DINO_reg import vit_large_reg
            self.encoder = vit_large_reg(**cfg.model.backbone)
        elif backbone_type == 'vit_giant2_reg':
            from ..backbones.ViT_DINO_reg import vit_giant2_reg
            self.encoder = vit_giant2_reg(**cfg.model.backbone)
        elif backbone_type == 'vit_large':
            from ..backbones.ViT_DINO import vit_large
            self.encoder = vit_large(**cfg.model.backbone)
        elif backbone_type == 'convnext_large':
            from ..backbones.ConvNeXt import convnext_large
            self.encoder = convnext_large(**cfg.model.backbone)
        else:
            raise NotImplementedError(f"Backbone {backbone_type} not implemented")
            
        # Handle decode head
        decode_head_type = cfg.model.decode_head.type
        if decode_head_type == 'RAFTDepthNormalDPT5':
            from ..decode_heads.RAFTDepthNormalDPTDecoder5 import RAFTDepthNormalDPT5
            self.decoder = RAFTDepthNormalDPT5(cfg)
        else:
            # For other decode head types, we'd need to check what file they're in
            raise NotImplementedError(f"Decode head {decode_head_type} not implemented")

    def forward(self, input, **kwargs):
        # [f_32, f_16, f_8, f_4]
        features = self.encoder(input)
        out = self.decoder(features, **kwargs)
        return out