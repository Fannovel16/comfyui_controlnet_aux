import numpy as np
import torch
from einops import repeat
from PIL import Image
from custom_controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, DEPTH_ANYTHING_V2_MODEL_NAME_DICT
from custom_controlnet_aux.depth_anything_v2.dpt import DepthAnythingV2
import cv2
import torch.nn.functional as F


# https://github.com/DepthAnything/Depth-Anything-V2/blob/main/app.py
model_configs = {
    'depth_anything_v2_vits.pth': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'depth_anything_v2_vitb.pth': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'depth_anything_v2_vitl.pth': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'depth_anything_v2_vitg.pth': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    'depth_anything_v2_metric_vkitti_vitl.pth': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'depth_anything_v2_metric_hypersim_vitl.pth': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

class DepthAnythingV2Detector:
    def __init__(self, model, filename):
        self.model = model
        self.device = "cpu"
        self.filename = filename
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=None, filename="depth_anything_v2_vits.pth"):
        if pretrained_model_or_path is None:
            pretrained_model_or_path = DEPTH_ANYTHING_V2_MODEL_NAME_DICT[filename]
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        model = DepthAnythingV2(**model_configs[filename])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.eval()
        return cls(model, filename)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self
    
    def __call__(self, input_image, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", max_depth=20.0, **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)

        depth = self.model.infer_image(cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR), input_size=518, max_depth=max_depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        if 'metric' in self.filename:
            depth = 255 - depth
        
        detected_map = repeat(depth, "h w -> h w 3")
        detected_map, remove_pad = resize_image_with_pad(detected_map, detect_resolution, upscale_method)
        detected_map = remove_pad(detected_map)
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map