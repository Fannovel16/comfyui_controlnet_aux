import numpy as np
import torch
from einops import repeat
from PIL import Image
from custom_controlnet_aux.util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, DEPTH_ANYTHING_MODEL_NAME
from custom_controlnet_aux.depth_anything.depth_anything.dpt import DPT_DINOv2
from custom_controlnet_aux.depth_anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
import torch.nn.functional as F

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

#https://huggingface.co/LiheYoung/depth_anything_vitl14/raw/main/config.json
DPT_CONFIGS = {
    "depth_anything_vitl14.pth": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024], "use_bn": False, "use_clstoken": False},
    "depth_anything_vitb14.pth": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768], "use_bn": False, "use_clstoken": False},
    "depth_anything_vits14.pth": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "use_bn": False, "use_clstoken": False}
}

class DepthAnythingDetector:
    def __init__(self, model):
        self.model = model
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=DEPTH_ANYTHING_MODEL_NAME, filename="depth_anything_vitl14.pth"):
        model_path = custom_hf_download(pretrained_model_or_path, filename, subfolder="checkpoints", repo_type="space")
        model = DPT_DINOv2(**DPT_CONFIGS[filename], localhub=True)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        return cls(model)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self
    
    def __call__(self, input_image, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        t, remove_pad = resize_image_with_pad(np.zeros_like(input_image), detect_resolution, upscale_method)
        t = remove_pad(t)

        h, w = t.shape[:2]
        h, w = int(h), int(w)
        image = transform({'image': input_image / 255.})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            depth = self.model(image)
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        detected_map = repeat(depth, "h w -> h w 3").cpu().numpy().astype(np.uint8)
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map