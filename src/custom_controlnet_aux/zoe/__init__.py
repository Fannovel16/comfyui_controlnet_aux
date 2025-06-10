# Modern ZoeDepth implementation using HuggingFace transformers
# This replaces the problematic torch.hub-based approach with PyTorch 2.7 compatible code

from .transformers_impl import ZoeDetector, ZoeDepthAnythingDetector