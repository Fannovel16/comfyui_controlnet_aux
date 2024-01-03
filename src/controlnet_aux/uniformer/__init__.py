import os
from .inference import init_segmentor, inference_segmentor, show_result_pyplot
import warnings
import cv2
import numpy as np
from PIL import Image
from controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, HF_MODEL_NAME
import torch

from custom_mmpkg.custom_mmseg.core.evaluation import get_palette

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "upernet_global_small.py")



class UniformerSegmentor:
    def __init__(self, netNetwork):
        self.model = netNetwork
    
    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=HF_MODEL_NAME, filename="upernet_global_small.pth"):
        model_path = custom_hf_download(pretrained_model_or_path, filename)

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

    def __call__(self, input_image=None, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        detected_map = self._inference(input_image)
        detected_map = remove_pad(HWC3(detected_map))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map

