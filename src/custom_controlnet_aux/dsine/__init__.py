import os
import types
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
from huggingface_hub import hf_hub_download

from .models.dsine_arch import DSINE
from .utils.utils import get_intrins_from_fov

# Local constants
DIFFUSION_EDGE_MODEL_NAME = "hr16/Diffusion-Edge"

# Local utility functions
def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def safer_memory(x):
    return np.ascontiguousarray(x.copy()).copy()

def resize_image_with_pad(input_image, resolution, upscale_method="INTER_CUBIC", skip_hwc3=False, mode='edge'):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    
    # Get upscale method
    upscale_methods = {"INTER_NEAREST": cv2.INTER_NEAREST, "INTER_LINEAR": cv2.INTER_LINEAR, 
                      "INTER_AREA": cv2.INTER_AREA, "INTER_CUBIC": cv2.INTER_CUBIC, 
                      "INTER_LANCZOS4": cv2.INTER_LANCZOS4}
    method = upscale_methods.get(upscale_method, cv2.INTER_CUBIC)
    
    img = cv2.resize(img, (W_target, H_target), interpolation=method if k > 1 else cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad

def common_input_validate(input_image, output_type, **kwargs):
    if "img" in kwargs:
        warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
        input_image = kwargs.pop("img")
    
    if "return_pil" in kwargs:
        warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
        output_type = "pil" if kwargs["return_pil"] else "np"
    
    if type(output_type) is bool:
        warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
        if output_type:
            output_type = "pil"

    if input_image is None:
        raise ValueError("input_image must be defined.")

    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)
        output_type = output_type or "pil"
    else:
        output_type = output_type or "np"
    
    return (input_image, output_type)

def custom_hf_download(pretrained_model_or_path, filename, subfolder=''):
    """Download model files from HuggingFace Hub"""
    annotator_ckpts_path = os.path.join(Path(__file__).parents[3], 'ckpts')
    local_dir = os.path.join(annotator_ckpts_path, pretrained_model_or_path)
    model_path = Path(local_dir).joinpath(*subfolder.split('/'), filename).__str__()

    if not os.path.exists(model_path):
        print(f"Downloading {filename} from {pretrained_model_or_path}")
        model_path = hf_hub_download(
            repo_id=pretrained_model_or_path,
            filename=filename,
            subfolder=subfolder,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
    
    print(f"model_path is {model_path}")
    return model_path

# load model
def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    # Load compatible weights only
    model_state = model.state_dict()
    compatible_dict = {}
    skipped_keys = []
    
    for k, v in load_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                compatible_dict[k] = v
            else:
                skipped_keys.append(f"{k}: checkpoint {v.shape} vs model {model_state[k].shape}")
        else:
            skipped_keys.append(f"{k}: not found in model")
    
    print(f"Loading checkpoint: {len(compatible_dict)} compatible, {len(skipped_keys)} skipped")
    if skipped_keys:
        print("Skipped keys with shape mismatches:")
        for key in skipped_keys[:5]:  # Show first 5 mismatches
            print(f"  {key}")
        if len(skipped_keys) > 5:
            print(f"  ... and {len(skipped_keys) - 5} more")
    
    model.load_state_dict(compatible_dict, strict=False)
    return model

def get_pad(orig_H, orig_W):
    if orig_W % 64 == 0:
        l = 0
        r = 0
    else:
        new_W = 64 * ((orig_W // 64) + 1)
        l = (new_W - orig_W) // 2
        r = (new_W - orig_W) - l

    if orig_H % 64 == 0:
        t = 0
        b = 0
    else:
        new_H = 64 * ((orig_H // 64) + 1)
        t = (new_H - orig_H) // 2
        b = (new_H - orig_H) - t
    return l, r, t, b

class DsineDetector:
    def __init__(self, model):
        self.model = model
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=DIFFUSION_EDGE_MODEL_NAME, filename="dsine.pt"):
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        model = DSINE()
        model = load_checkpoint(model_path, model)
        model.eval()

        return cls(model)

    def to(self, device):
        self.model.to(device)
        self.model.pixel_coords = self.model.pixel_coords.to(device)
        self.device = device
        return self


    def __call__(self, input_image, fov=60.0, iterations=5, detect_resolution=512, output_type="pil", upscale_method="INTER_CUBIC", **kwargs):
        self.model.num_iter = iterations
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        orig_H, orig_W = input_image.shape[:2]
        l, r, t, b = get_pad(orig_H, orig_W)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method, mode="constant")
        with torch.no_grad():
            input_image = torch.from_numpy(input_image).float().to(self.device)
            input_image = input_image / 255.0
            input_image = rearrange(input_image, 'h w c -> 1 c h w')
            input_image = self.norm(input_image)
            
            intrins = get_intrins_from_fov(new_fov=fov, H=orig_H, W=orig_W, device=self.device).unsqueeze(0)
            intrins[:, 0, 2] += l
            intrins[:, 1, 2] += t

            normal = self.model(input_image, intrins)
            normal = normal[-1][0]
            normal = ((normal + 1) * 0.5).clip(0, 1)
            
            normal = rearrange(normal, 'c h w -> h w c').cpu().numpy()
            normal_image = (normal * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = HWC3(normal_image)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map
    