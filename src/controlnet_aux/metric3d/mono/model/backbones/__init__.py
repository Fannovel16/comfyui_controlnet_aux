from .ConvNeXt import convnext_xlarge
from .ConvNeXt import convnext_small
from .ConvNeXt import convnext_base
from .ConvNeXt import convnext_large
from .ConvNeXt import convnext_tiny
from .ViT_DINO import vit_large
from .ViT_DINO_reg import vit_small_reg, vit_large_reg, vit_giant2_reg

__all__ = [
    'convnext_xlarge', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_tiny', 'vit_small_reg', 'vit_large_reg', 'vit_giant2_reg'
]
