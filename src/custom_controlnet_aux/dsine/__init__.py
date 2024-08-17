import os
import types
import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image

from custom_controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, DIFFUSION_EDGE_MODEL_NAME
from .models.dsine_arch import DSINE
from custom_controlnet_aux.dsine.utils.utils import get_intrins_from_fov

# load model
def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model

def get_pad(orig_H, orig_W):
    if orig_W % 64 == 0:
        l = 0
        r = 0
    else:
        new_W = 64 * ((orig_W // 64) + 1)
        l = (new_W - orig_W) // 2
        r = (new_W - orig_W) - l

    if orig_H % 64 == 0:
        t = 0
        b = 0
    else:
        new_H = 64 * ((orig_H // 64) + 1)
        t = (new_H - orig_H) // 2
        b = (new_H - orig_H) - t
    return l, r, t, b

class DsineDetector:
    def __init__(self, model):
        self.model = model
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=DIFFUSION_EDGE_MODEL_NAME, filename="dsine.pt"):
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        model = DSINE()
        model = load_checkpoint(model_path, model)
        model.eval()

        return cls(model)

    def to(self, device):
        self.model.to(device)
        self.model.pixel_coords = self.model.pixel_coords.to(device)
        self.device = device
        return self


    def __call__(self, input_image, fov=60.0, iterations=5, detect_resolution=512, output_type="pil", upscale_method="INTER_CUBIC", **kwargs):
        self.model.num_iter = iterations
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        orig_H, orig_W = input_image.shape[:2]
        l, r, t, b = get_pad(orig_H, orig_W)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method, mode="constant")
        with torch.no_grad():
            input_image = torch.from_numpy(input_image).float().to(self.device)
            input_image = input_image / 255.0
            input_image = rearrange(input_image, 'h w c -> 1 c h w')
            input_image = self.norm(input_image)
            
            intrins = get_intrins_from_fov(new_fov=fov, H=orig_H, W=orig_W, device=self.device).unsqueeze(0)
            intrins[:, 0, 2] += l
            intrins[:, 1, 2] += t

            normal = self.model(input_image, intrins)
            normal = normal[-1][0]
            normal = ((normal + 1) * 0.5).clip(0, 1)
            
            normal = rearrange(normal, 'c h w -> h w c').cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = HWC3(normal_image)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
    