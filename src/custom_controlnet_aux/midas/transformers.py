"""
MiDaS implementation using HuggingFace transformers for PyTorch 2.7 compatibility.
"""
import numpy as np
import torch
import cv2
from PIL import Image
from typing import Union

# Import utilities
from ..util import HWC3, common_input_validate, resize_image_with_pad


class MidasDetector:
    
    def __init__(self, model_name="Intel/dpt-large"):
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        
        self.model_name = model_name
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.device = "cpu"

    @classmethod  
    def from_pretrained(cls, pretrained_model_or_path=None, model_type="dpt_hybrid", filename="dpt_hybrid-midas-501f0c75.pt"):
        # Map legacy model types to HuggingFace models
        model_mapping = {
            "dpt_large": "Intel/dpt-large",
            "dpt_hybrid": "Intel/dpt-hybrid-midas", 
            "midas_v21": "Intel/dpt-large",
            "midas_v21_small": "Intel/dpt-large"
        }
        
        # Use filename for model selection if provided
        if filename and isinstance(filename, str):
            if "dpt_large" in filename.lower():
                model_name = "Intel/dpt-large"
            elif "dpt_hybrid" in filename.lower():
                model_name = "Intel/dpt-hybrid-midas"
            else:
                model_name = model_mapping.get(model_type, "Intel/dpt-large")
        else:
            model_name = model_mapping.get(model_type, "Intel/dpt-large")
        
        return cls(model_name)

    def to(self, device):
        self.model = self.model.to(device) 
        self.device = device
        return self

    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1, depth_and_normal=False, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        # Convert to PIL for processor
        pil_image = Image.fromarray(detected_map.astype(np.uint8))
        
        # Process with HuggingFace pipeline
        with torch.no_grad():
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth
            
            # Normalize depth
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(detected_map.shape[0], detected_map.shape[1]),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            if depth_and_normal:
                depth_np = depth.cpu().numpy()
                x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
                y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
                z = np.ones_like(x) * a
                x[depth_pt < bg_th] = 0
                y[depth_pt < bg_th] = 0
                normal = np.stack([x, y, z], axis=2)
                normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
                normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)[:, :, ::-1]
        
        depth_image = HWC3(depth_image)
        if depth_and_normal:
            normal_image = HWC3(normal_image)

        depth_image = remove_pad(depth_image)
        if depth_and_normal:
            normal_image = remove_pad(normal_image)
        
        if output_type == "pil":
            depth_image = Image.fromarray(depth_image)
            if depth_and_normal:
                normal_image = Image.fromarray(normal_image)
        
        if depth_and_normal:
            return depth_image, normal_image
        else:
            return depth_image