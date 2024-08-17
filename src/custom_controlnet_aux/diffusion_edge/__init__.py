from custom_controlnet_aux.diffusion_edge.model import DiffusionEdge, prepare_args
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from custom_controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, DIFFUSION_EDGE_MODEL_NAME

class DiffusionEdgeDetector:
    def __init__(self, model):
        self.model = model
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=DIFFUSION_EDGE_MODEL_NAME, filename="diffusion_edge_indoor.pt"):
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        model = DiffusionEdge(prepare_args(model_path))
        return cls(model)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self
    
    def __call__(self, input_image, detect_resolution=512, patch_batch_size=8, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        with torch.no_grad():
            input_image = rearrange(torch.from_numpy(input_image), "h w c -> 1 c h w")
            input_image = input_image.float() / 255.
            line = self.model(input_image, patch_batch_size)
            line = rearrange(line, "1 c h w -> h w c")
        
        detected_map = line.cpu().numpy().__mul__(255.).astype(np.uint8)
        detected_map = remove_pad(HWC3(detected_map))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map