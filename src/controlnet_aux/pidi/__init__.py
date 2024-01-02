import os
import warnings

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from controlnet_aux.util import HWC3, nms, resize_image_with_pad, safe_step,common_input_validate, custom_hf_download, HF_MODEL_NAME
from .model import pidinet


class PidiNetDetector:
    def __init__(self, netNetwork):
        self.netNetwork = netNetwork

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="table5_pidinet.pth"):
        model_path = custom_hf_download(pretrained_model_or_path, filename)

        netNetwork = pidinet()
        netNetwork.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path)['state_dict'].items()})
        netNetwork.eval()

        return cls(netNetwork)

    def to(self, device):
        self.netNetwork.to(device)
        return self
    
    def __call__(self, input_image, detect_resolution=512, safe=False, output_type="pil", scribble=False, apply_filter=False, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        device = next(iter(self.netNetwork.parameters())).device
        
        detected_map = detected_map[:, :, ::-1].copy()
        with torch.no_grad():
            image_pidi = torch.from_numpy(detected_map).float().to(device)
            image_pidi = image_pidi / 255.0
            image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
            edge = self.netNetwork(image_pidi)[-1]
            edge = edge.cpu().numpy()
            if apply_filter:
                edge = edge > 0.5 
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge[0, 0]

        if scribble:
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        detected_map = HWC3(remove_pad(detected_map))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
