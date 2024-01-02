import os
from .api import make_detectron2_model, semantic_run
from pathlib import Path
import warnings
from controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, HF_MODEL_NAME
import numpy as np
import cv2
from PIL import Image

DEFAULT_CONFIGS = {
    "coco": {
        "name": "150_16_swin_l_oneformer_coco_100ep.pth",
        "config": os.path.join(os.path.dirname(__file__), 'configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml')
    },
    "ade20k": {
        "name": "250_16_swin_l_oneformer_ade20k_160k.pth",
        "config": os.path.join(os.path.dirname(__file__), 'configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml')
    }
}
class OneformerSegmentor:
    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata

    def to(self, device):
        self.model.model.to(device)
        return self
    
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="250_16_swin_l_oneformer_ade20k_160k.pth", config_path = None):
        config_path = config_path or DEFAULT_CONFIGS["ade20k" if "ade20k" in filename else "coco"]["config"]
        model_path = custom_hf_download(pretrained_model_or_path, filename)

        model, metadata = make_detectron2_model(config_path, model_path)

        return cls(model, metadata)
    
    def __call__(self, input_image=None, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        detected_map = semantic_run(input_image, self.model, self.metadata)
        detected_map = remove_pad(HWC3(detected_map))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
