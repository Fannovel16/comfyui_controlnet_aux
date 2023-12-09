import functools
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image

from controlnet_aux.util import HWC3, resize_image_with_pad, common_input_validate, annotator_ckpts_path, custom_hf_download

class DenseposeDetector:
    def __init__(self, model):
        self.dense_pose_estimation = model
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=annotator_ckpts_path):
        torchscript_model_path = custom_hf_download(pretrained_model_or_path, filename, cache_dir=cache_dir)
        densepose = torch.jit.load(torchscript_model_path, map_location="cpu")
        return cls(densepose)

    def to(self, device):
        self.dense_pose_estimation.to(device)
        self.device = device
        return self
    
    def __call__(self, input_image, detect_resolution=512, output_type="pil", upscale_method="INTER_CUBIC", cmap="viridis", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        H, W  = input_image.shape[:2]
        input_image = rearrange(torch.from_numpy(input_image).to(self.device), 'h w c -> c h w')
        detected_map = self.dense_pose_estimation(input_image)[-1 if cmap=="viridis" else -2]
        if detected_map.all() == -1:
            detected_map = np.zeros([H, W, 3], dtype=np.uint8)
        else:
            detected_map = detected_map.cpu().detach().numpy()
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        detected_map = remove_pad(HWC3(detected_map))
        return detected_map
