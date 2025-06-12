"""
ZoeDepth implementation using HuggingFace transformers.
Uses official Intel models for depth estimation.
"""

import numpy as np
import torch
from PIL import Image
from transformers import pipeline, AutoImageProcessor, ZoeDepthForDepthEstimation

# Local utility functions
def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def safer_memory(x):
    return np.ascontiguousarray(x.copy()).copy()

def resize_image_with_pad(input_image, resolution, upscale_method="INTER_CUBIC", skip_hwc3=False, mode='edge'):
    import cv2
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    
    upscale_methods = {"INTER_NEAREST": cv2.INTER_NEAREST, "INTER_LINEAR": cv2.INTER_LINEAR, 
                      "INTER_AREA": cv2.INTER_AREA, "INTER_CUBIC": cv2.INTER_CUBIC, 
                      "INTER_LANCZOS4": cv2.INTER_LANCZOS4}
    method = upscale_methods.get(upscale_method, cv2.INTER_CUBIC)
    
    img = cv2.resize(img, (W_target, H_target), interpolation=method if k > 1 else cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad

def common_input_validate(input_image, output_type, **kwargs):
    import warnings
    if "img" in kwargs:
        warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
        input_image = kwargs.pop("img")
    
    if "return_pil" in kwargs:
        warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
        output_type = "pil" if kwargs["return_pil"] else "np"
    
    if type(output_type) is bool:
        warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
        if output_type:
            output_type = "pil"

    if input_image is None:
        raise ValueError("input_image must be defined.")

    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)
        output_type = output_type or "pil"
    else:
        output_type = output_type or "np"
    
    return (input_image, output_type)


class ZoeDetector:
    """ZoeDepth depth estimation using HuggingFace transformers."""
    
    def __init__(self, model_name="Intel/zoedepth-nyu-kitti"):
        """Initialize ZoeDepth with specified model."""
        self.pipe = pipeline(task="depth-estimation", model=model_name)
        self.device = "cpu"

    @classmethod  
    def from_pretrained(cls, pretrained_model_or_path="Intel/zoedepth-nyu-kitti", filename=None, **kwargs):
        """Create ZoeDetector from pretrained model."""
        return cls(model_name=pretrained_model_or_path)
    
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
                
            vmin = np.percentile(depth_array, 2)
            vmax = np.percentile(depth_array, 85)
            
            depth_array = depth_array - vmin
            depth_array = depth_array / (vmax - vmin)
            depth_array = 1.0 - depth_array
            depth_image = (depth_array * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(depth_image))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map


class ZoeDepthAnythingDetector:
    """ZoeDepthAnything implementation using HuggingFace transformers."""
    
    def __init__(self, model_name="Intel/zoedepth-nyu-kitti"):
        """Initialize ZoeDepthAnything detector."""
        self.pipe = pipeline(task="depth-estimation", model=model_name)
        self.device = "cpu"

    @classmethod  
    def from_pretrained(cls, pretrained_model_or_path="Intel/zoedepth-nyu-kitti", filename=None, **kwargs):
        """Create from pretrained model."""
        return cls(model_name=pretrained_model_or_path)
    
    def to(self, device):
        """Move model to specified device."""
        self.pipe.model = self.pipe.model.to(device) 
        self.device = device
        return self
        
    def __call__(self, input_image, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        """Perform depth estimation."""
        detector = ZoeDetector(model_name="Intel/zoedepth-nyu-kitti")
        detector.pipe = self.pipe
        detector.device = self.device
        
        return detector(input_image, detect_resolution, output_type, upscale_method, **kwargs)