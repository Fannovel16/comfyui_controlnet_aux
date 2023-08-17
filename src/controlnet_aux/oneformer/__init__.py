import os
from .api import make_detectron2_model, semantic_run
from pathlib import Path
from huggingface_hub import hf_hub_download
import warnings
from ..util import HWC3, resize_image
import numpy as np
import cv2
from PIL import Image

DEFAULT_CONFIGS = {
    "coco": {
        "name": "150_16_swin_l_oneformer_coco_100ep.pth",
        "config": Path(os.path.dirname(__file__), 'configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml')
    },
    "ade20k": {
        "name": "250_16_swin_l_oneformer_ade20k_160k.pth",
        "config": Path(os.path.dirname(__file__), 'configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml')
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
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None, config_path = None):
        filename = filename or "250_16_swin_l_oneformer_ade20k_160k.pth"
        config_path = config_path or DEFAULT_CONFIGS["ade20k" if "ade20k" in filename else "coco"]["config"]

        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)

        model, metadata = make_detectron2_model(config_path, model_path)

        return cls(model, metadata)
    
    def __call__(self, input_image=None, detect_resolution=512, image_resolution=512, output_type=None, **kwargs):
        if "img" in kwargs:
            warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
            input_image = kwargs.pop("img")
        
        if input_image is None:
            raise ValueError("input_image must be defined.")

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
            output_type = output_type or "pil"
        else:
            output_type = output_type or "np"
        
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        detected_map = semantic_run(input_image, self.model, self.metadata)
        detected_map = HWC3(detected_map)      
         
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
