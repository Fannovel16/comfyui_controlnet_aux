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
from .utils import infernce

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
        H, W, C = input_image.shape
        detected_maps = infernce(self.dense_pose_estimation, input_image, self.device)
        detected_map = detected_maps[-1 if cmap=="viridis" else -2]
        
        if detected_map.all() == -1:
            detected_map = np.zeros([H, W, 3], dtype=np.uint8)
        else:
            detected_map = detected_map
        detected_map, remove_pad = resize_image_with_pad(detected_map, detect_resolution, upscale_method)
        detected_map = remove_pad(detected_map)
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        return detected_map
