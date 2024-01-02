import os

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, HF_MODEL_NAME
from .api import MiDaSInference


class MidasDetector:
    def __init__(self, model):
        self.model = model
        
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, model_type="dpt_hybrid", filename="dpt_hybrid-midas-501f0c75.pt"):
        subfolder = "annotator/ckpts" if pretrained_model_or_path == "lllyasviel/ControlNet" else ''
        model_path = custom_hf_download(pretrained_model_or_path, filename, subfolder=subfolder)
        model = MiDaSInference(model_type=model_type, model_path=model_path)
        return cls(model)
        

    def to(self, device):
        self.model.to(device)
        return self
    
    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1, depth_and_normal=False, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        device = next(iter(self.model.parameters())).device
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        image_depth = detected_map
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float()
            image_depth = image_depth.to(device)
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model(image_depth)[0]

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
