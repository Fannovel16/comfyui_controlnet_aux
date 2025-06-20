
import torch
import os
from pathlib import Path

CODE_SPACE=Path(os.path.dirname(os.path.abspath(__file__)))

from custom_mmpkg.custom_mmcv.utils import Config, DictAction
from .mono.model.monodepth_model import get_configured_monodepth_model
from .mono.utils.running import load_ckpt
from .mono.utils.do_test import transform_test_data_scalecano, get_prediction
import numpy as np
from .mono.utils.visualization import vis_surface_normal
from einops import repeat
from PIL import Image
from ..util import HWC3, common_input_validate, resize_image_with_pad, custom_hf_download, METRIC3D_MODEL_NAME
import re
import matplotlib.pyplot as plt

def load_model(model_selection, model_path):
    if model_selection == "vit-small":
        cfg = Config.fromfile(CODE_SPACE / 'mono/configs/HourglassDecoder/vit.raft5.small.py')
    elif model_selection == "vit-large":
        cfg = Config.fromfile(CODE_SPACE / 'mono/configs/HourglassDecoder/vit.raft5.large.py')
    elif model_selection == "vit-giant2":
        cfg = Config.fromfile(CODE_SPACE / 'mono/configs/HourglassDecoder/vit.raft5.giant2.py')
    else:
        raise NotImplementedError(f"metric3d model: {model_selection}")
    model = get_configured_monodepth_model(cfg, )
    model, _,  _, _ = load_ckpt(model_path, model, strict_match=False)
    model.eval()
    model = model
    return model, cfg

def gray_to_colormap(img, cmap='rainbow'):
    """
    Transfer gray map to matplotlib colormap
    """
    assert img.ndim == 2

    img[img<0] = 0
    mask_invalid = img < 1e-10
    img = img / (img.max() + 1e-8)
    norm = plt.Normalize(vmin=0, vmax=1.1)  # Use plt.Normalize instead of matplotlib.colors.Normalize
    cmap_m = plt.get_cmap(cmap)  # Access the colormap directly from plt
    map = plt.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0
    return colormap

def predict_depth_normal(model, cfg, np_img, fx=1000.0, fy=1000.0, state_cache={}):
    intrinsic = [fx, fy, np_img.shape[1]/2, np_img.shape[0]/2]
    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(np_img, intrinsic, cfg.data_basic, device=next(model.parameters()).device)

    with torch.no_grad():
        pred_depth, confidence, output = get_prediction(
            model = model,
            input = rgb_input.unsqueeze(0),
            cam_model = cam_models_stacks,
            pad_info = pad,
            scale_info = label_scale_factor,
            gt_depth = None,
            normalize_scale = cfg.data_basic.depth_range[1],
            ori_shape=[np_img.shape[0], np_img.shape[1]],
        )

        pred_normal = output['normal_out_list'][0][:, :3, :, :] 
        H, W = pred_normal.shape[2:]
        pred_normal = pred_normal[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
        pred_depth = pred_depth[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3] ]

    pred_depth = pred_depth.squeeze().cpu().numpy()
    pred_color = gray_to_colormap(pred_depth, 'Greys')

    pred_normal = torch.nn.functional.interpolate(pred_normal, [np_img.shape[0], np_img.shape[1]], mode='bilinear').squeeze()
    pred_normal = pred_normal.permute(1,2,0)
    pred_color_normal = vis_surface_normal(pred_normal)
    pred_normal = pred_normal.cpu().numpy()
    
    # Storing depth and normal map in state for potential 3D reconstruction
    state_cache['depth'] = pred_depth
    state_cache['normal'] = pred_normal
    state_cache['img'] = np_img
    state_cache['intrinsic'] = intrinsic
    state_cache['confidence'] = confidence 

    return pred_color, pred_color_normal, state_cache

class Metric3DDetector:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=METRIC3D_MODEL_NAME, filename="metric_depth_vit_small_800k.pth"):
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        backbone = re.findall(r"metric_depth_vit_(\w+)_", model_path)[0]
        model, cfg = load_model(f'vit-{backbone}', model_path)
        return cls(model, cfg)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self
    
    def __call__(self, input_image, detect_resolution=512, fx=1000, fy=1000, output_type=None, upscale_method="INTER_CUBIC", depth_and_normal=True, **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)

        depth_map, normal_map, _ = predict_depth_normal(self.model, self.cfg, input_image, fx=fx, fy=fy)
        # ControlNet uses inverse depth and normal
        depth_map, normal_map = depth_map, 255 - normal_map 
        depth_map, remove_pad = resize_image_with_pad(depth_map, detect_resolution, upscale_method)
        normal_map, _ = resize_image_with_pad(normal_map, detect_resolution, upscale_method)
        depth_map, normal_map = remove_pad(depth_map), remove_pad(normal_map)
        
        if output_type == "pil":
            depth_map = Image.fromarray(depth_map)
            normal_map = Image.fromarray(normal_map)
        
        if depth_and_normal:
            return depth_map, normal_map
        else:
            return depth_map
