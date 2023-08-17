import os

import cv2
import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image

from ..util import HWC3, resize_image
from .zoedepth.models.zoedepth.zoedepth_v1 import ZoeDepth
from .zoedepth.utils.config import get_config


class ZoeDetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None):
        filename = filename or "ZoeD_M12_N.pt"

        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)
            
        conf = get_config("zoedepth", "infer")
        model = ZoeDepth.build_from_config(conf)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
        model.eval()

        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self
    
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type=None):
        device = next(iter(self.model.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
            output_type = output_type or "pil"
        else:
            output_type = output_type or "np"
        
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        assert input_image.ndim == 3
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

        detected_map = depth_image
        detected_map = HWC3(detected_map)      
         
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
