import os
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

from custom_controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, HF_MODEL_NAME
from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines


class MLSDdetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="mlsd_large_512_fp32.pth"):
        subfolder = "annotator/ckpts" if pretrained_model_or_path == "lllyasviel/ControlNet" else ''
        model_path = custom_hf_download(pretrained_model_or_path, filename, subfolder=subfolder)
        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()

        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self
    
    def __call__(self, input_image, thr_v=0.1, thr_d=0.1, detect_resolution=512, output_type="pil", upscale_method="INTER_AREA", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        img = detected_map
        img_output = np.zeros_like(img)
        try:
            with torch.no_grad():
                lines = pred_lines(img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d)
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
        except Exception as e:
            pass

        detected_map = remove_pad(HWC3(img_output[:, :, 0]))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
