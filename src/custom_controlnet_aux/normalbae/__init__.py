import os
import types
import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image

from custom_controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, HF_MODEL_NAME
from .nets.NNET import NNET


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

class NormalBaeDetector:
    def __init__(self, model):
        self.model = model
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="scannet.pt"):
        model_path = custom_hf_download(pretrained_model_or_path, filename)

        args = types.SimpleNamespace()
        args.mode = 'client'
        args.architecture = 'BN'
        args.pretrained = 'scannet'
        args.sampling_ratio = 0.4
        args.importance_ratio = 0.7
        model = NNET(args)
        model = load_checkpoint(model_path, model)
        model.eval()

        return cls(model)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self


    def __call__(self, input_image, detect_resolution=512, output_type="pil", upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        image_normal = detected_map
        with torch.no_grad():
            image_normal = torch.from_numpy(image_normal).float().to(self.device)
            image_normal = image_normal / 255.0
            image_normal = rearrange(image_normal, 'h w c -> 1 c h w')
            image_normal = self.norm(image_normal)

            normal = self.model(image_normal)
            normal = normal[0][-1][:, :3]
            # d = torch.sum(normal ** 2.0, dim=1, keepdim=True) ** 0.5
            # d = torch.maximum(d, torch.ones_like(d) * 1e-5)
            # normal /= d
            normal = ((normal + 1) * 0.5).clip(0, 1)

            normal = rearrange(normal[0], 'c h w -> h w c').cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(normal_image))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
    