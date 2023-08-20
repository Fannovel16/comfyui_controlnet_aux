import os
from .inference import init_segmentor, inference_segmentor, show_result_pyplot
import warnings
import cv2
import numpy as np
from PIL import Image
from ..util import HWC3, resize_image
from huggingface_hub import hf_hub_download
import torch

from custom_mmpkg.mmseg.core.evaluation import get_palette

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "upernet_global_small.py")



class UniformerSegmentor:
    def __init__(self, netNetwork):
        self.model = netNetwork
    
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None):
        filename = filename or "upernet_global_small.pth"

        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)

        netNetwork = init_segmentor(config_file, model_path, device="cpu")
        netNetwork.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path)['state_dict'].items()})
        netNetwork.eval()

        return cls(netNetwork)

    def to(self, device):
        self.model.to(device)
        return self

    def _inference(self, img):
        if next(self.model.parameters()).device.type == 'mps':
            # adaptive_avg_pool2d can fail on MPS, workaround with CPU
            import torch.nn.functional
            
            orig_adaptive_avg_pool2d = torch.nn.functional.adaptive_avg_pool2d
            def cpu_if_exception(input, *args, **kwargs):
                try:
                    return orig_adaptive_avg_pool2d(input, *args, **kwargs)
                except:
                    return orig_adaptive_avg_pool2d(input.cpu(), *args, **kwargs).to(input.device)
            
            try:
                torch.nn.functional.adaptive_avg_pool2d = cpu_if_exception
                result = inference_segmentor(self.model, img)
            finally:
                torch.nn.functional.adaptive_avg_pool2d = orig_adaptive_avg_pool2d
        else:
            result = inference_segmentor(self.model, img)

        res_img = show_result_pyplot(self.model, img, result, get_palette('ade'), opacity=1)
        return res_img

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

        detected_map = self._inference(input_image)
        detected_map = HWC3(detected_map)      
         
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map

