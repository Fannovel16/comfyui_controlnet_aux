import os
from .api import make_detectron2_model, semantic_run
from pathlib import Path

import os
import warnings

import cv2
import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from PIL import Image

from ..util import HWC3, nms, resize_image


class OneformerSegmentor:
    configs = {
        "coco": {
            "name": "150_16_swin_l_oneformer_coco_100ep.pth",
            "config": Path(os.path.dirname(__file__), 'configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml')
        },
        "ade20k": {
            "name": "250_16_swin_l_oneformer_ade20k_160k.pth",
            "config": Path(os.path.dirname(__file__), 'configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml')
        }
    }

    def __init__(self, netNetwork, metadata):
        self.model = netNetwork
        self.metadata = metadata

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None, config_file_name=None):
        filename = filename or "250_16_swin_l_oneformer_ade20k_160k.pth"
        config_path = config_path or cls.configs["ada20k" if "ada20k" in filename else "coco"]

        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)

        netNetwork, metadata = make_detectron2_model(config_path, model_path)

        return cls(netNetwork, metadata)

    def to(self, device):
        self.model.model.to(device)
        return self
    
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        
        detected_map = semantic_run(img, self.model, self.metadata)
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
