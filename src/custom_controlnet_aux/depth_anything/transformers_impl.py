"""
Modern DepthAnything implementation using HuggingFace transformers.
Replaces legacy torch.hub.load DINOv2 backbone with transformers pipeline.
"""

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from custom_controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad

class DepthAnythingDetector:
    """DepthAnything depth estimation using HuggingFace transformers."""
    
    def __init__(self, model_name="LiheYoung/depth-anything-large-hf"):
        """Initialize DepthAnything with specified model."""
        self.pipe = pipeline(task="depth-estimation", model=model_name)
        self.device = "cpu"

    @classmethod  
    def from_pretrained(cls, pretrained_model_or_path=None, filename="depth_anything_vitl14.pth"):
        """Create DepthAnything from pretrained model, mapping legacy names to HuggingFace models."""
        
        # Map legacy checkpoint names to modern HuggingFace models
        model_mapping = {
            "depth_anything_vitl14.pth": "LiheYoung/depth-anything-large-hf",
            "depth_anything_vitb14.pth": "LiheYoung/depth-anything-base-hf", 
            "depth_anything_vits14.pth": "LiheYoung/depth-anything-small-hf"
        }
        
        model_name = model_mapping.get(filename, "LiheYoung/depth-anything-large-hf")
        return cls(model_name=model_name)
    
    def to(self, device):
        """Move model to specified device."""
        self.pipe.model = self.pipe.model.to(device) 
        self.device = device
        return self
        
    def __call__(self, input_image, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        """Perform depth estimation on input image."""
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        if isinstance(input_image, np.ndarray):
            pil_image = Image.fromarray(input_image)
        else:
            pil_image = input_image
        
        with torch.no_grad():
            result = self.pipe(pil_image)
            depth = result["depth"]
            
            if isinstance(depth, Image.Image):
                depth_array = np.array(depth, dtype=np.float32)
            else:
                depth_array = np.array(depth)
                
            # Normalize depth values to 0-255 range
            depth_min = depth_array.min()
            depth_max = depth_array.max()
            if depth_max > depth_min:
                depth_array = (depth_array - depth_min) / (depth_max - depth_min) * 255.0
            else:
                depth_array = np.zeros_like(depth_array)
                
            depth_image = depth_array.astype(np.uint8)

        detected_map = remove_pad(HWC3(depth_image))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map