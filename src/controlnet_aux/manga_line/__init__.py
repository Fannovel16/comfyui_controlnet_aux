# MangaLineExtraction_PyTorch
# https://github.com/ljsabc/MangaLineExtraction_PyTorch

#NOTE: This preprocessor is designed to work with lineart_anime ControlNet so the result will be white lines on black canvas

import torch
import numpy as np
import os
import cv2
from einops import rearrange
from .model_torch import res_skip
from PIL import Image
import warnings

from controlnet_aux.util import HWC3, resize_image_with_pad, common_input_validate, custom_hf_download, HF_MODEL_NAME

class LineartMangaDetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="erika.pth"):
        model_path = custom_hf_download(pretrained_model_or_path, filename)

        net = res_skip()
        ckpt = torch.load(model_path)
        for key in list(ckpt.keys()):
            if 'module.' in key:
                ckpt[key.replace('module.', '')] = ckpt[key]
                del ckpt[key]
        net.load_state_dict(ckpt)
        net.eval()
        return cls(net)

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, input_image, detect_resolution=512, output_type="pil", upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, 256 * int(np.ceil(float(detect_resolution) / 256.0)), upscale_method)
        device = next(iter(self.model.parameters())).device

        img = cv2.cvtColor(detected_map, cv2.COLOR_RGB2GRAY)
        with torch.no_grad():
            image_feed = torch.from_numpy(img).float().to(device)
            image_feed = rearrange(image_feed, 'h w -> 1 1 h w')

            line = self.model(image_feed)
            line = line.cpu().numpy()[0,0,:,:]
            line[line > 255] = 255
            line[line < 0] = 0

            line = line.astype(np.uint8)
        
        detected_map = HWC3(line)
        detected_map = remove_pad(255 - detected_map)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
