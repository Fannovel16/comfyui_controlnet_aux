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

from ..util import HWC3, resize_image
from huggingface_hub import hf_hub_download

class LineartMangaDetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=None, filename=None, cache_dir=None):
        filename = filename or "erika.pth"

        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)

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

    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"
        
        device = next(iter(self.model.parameters())).device

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        H, W, C = input_image.shape
        Hn = 256 * int(np.ceil(float(H) / 256.0))
        Wn = 256 * int(np.ceil(float(W) / 256.0))
        img = cv2.resize(input_image, (Wn, Hn), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        with torch.no_grad():
            image_feed = torch.from_numpy(img).float().to(device)
            image_feed = rearrange(image_feed, 'h w -> 1 1 h w')

            line = self.model(image_feed)
            line = line.cpu().numpy()[0,0,:,:]
            line[line > 255] = 255
            line[line < 0] = 0

            line = line.astype(np.uint8)
            line = cv2.resize(line, (W, H), interpolation=cv2.INTER_CUBIC)
        
        detected_map = line

        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        detected_map = 255 - detected_map
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
