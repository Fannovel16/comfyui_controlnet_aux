import os

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, HF_MODEL_NAME
from .zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from .zoedepth.utils.config import get_config


class ZoeDetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="ZoeD_M12_N.pt"):
        model_path = custom_hf_download(pretrained_model_or_path, filename)
            
        conf = get_config("zoedepth", "infer")
        model = ZoeDepth.build_from_config(conf)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
        model.eval()

        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self
    
    def __call__(self, input_image, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        device = next(iter(self.model.parameters())).device
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().to(device)
            image_depth = image_depth / 255.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model.infer(image_depth)

            depth = depth[0, 0].cpu().numpy()

            vmin = np.percentile(depth, 2)
            vmax = np.percentile(depth, 85)

            depth -= vmin
            depth /= vmax - vmin
            depth = 1.0 - depth
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(depth_image))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
