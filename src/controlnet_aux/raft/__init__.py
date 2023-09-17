import os
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ..util import resize_image_with_pad, common_input_validate, HWC3
from torchvision.models.optical_flow import raft_small, raft_large
from einops import rearrange
from torchvision.utils import flow_to_image

class RaftOpticalFlowEmbedder:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None):
        filename = filename or "raft_large_C_T_SKHT_V2-ff5fadd5.pth"

        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir)

        model = raft_large() if "large" in filename else raft_small()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return cls(model)
    
    def to(self, device):
        self.model.to(device)
        return self
    
    def __call__(self, input_images, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", num_flow_updates=12, **kwargs):
        input_images, output_type = common_input_validate(input_images, output_type, **kwargs)
        assert input_images.ndim == 4, f"RAFT requires multiple images so ndim should be 4 instead of {input_images.ndim}"
        assert len(input_images) >= 2, f"RAFT requires at least two images to work with, only found {len(input_images)}"
        
        input_images = [resize_image_with_pad(input_image[1:], detect_resolution, upscale_method)[0] for input_image in input_images]
        input_images = np.stack(input_images, axis=0)
        _, remove_pad = resize_image_with_pad(input_images[0], detect_resolution, upscale_method)

        device = next(iter(self.model.parameters())).device
        model = self.model
        
        with torch.no_grad():
            images = torch.from_numpy(input_images).float().to(device)
            images = images / 255.0
            images = rearrange(images, 'n h w c -> n c h w')
            idxes = np.arange(len(images) - 1)
            flow_prediction = model(images[idxes], images[idxes + 1], num_flow_updates=num_flow_updates)[-1]
            #https://huggingface.co/CiaraRowles/TemporalNet2/blob/main/temporalvideo.py#L237
            flow_images = flow_to_image(flow_prediction)
            six_channel_images = torch.cat((images[idxes], flow_images), dim=1) #NCHW
            #https://huggingface.co/CiaraRowles/TemporalNet2/blob/main/temporalvideo.py#L124
            six_channel_images = rearrange(six_channel_images, "n c h w -> n h w c").cpu().numpy()
            six_channel_images = (six_channel_images * 255.0).clip(0, 255).astype(np.uint8)
        
        detected_maps = np.stack([remove_pad(image) for image in six_channel_images], axis=0)

        if output_type == "pil":
            detected_maps = [Image.fromarray(detected_map[:, :, :, :3]) for detected_map in detected_maps]
            
        return detected_maps