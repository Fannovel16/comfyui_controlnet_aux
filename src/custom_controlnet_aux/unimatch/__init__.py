import os
import warnings

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from custom_controlnet_aux.util import resize_image_with_pad,common_input_validate, custom_hf_download, UNIMATCH_MODEL_NAME
from .utils.flow_viz import save_vis_flow_tofile, flow_to_image
from .unimatch.unimatch import UniMatch
import torch.nn.functional as F
from argparse import Namespace

def inference_flow(model,
                   image1, #np array of HWC
                   image2,
                   padding_factor=8,
                   inference_size=None,
                   attn_type='swin',
                   attn_splits_list=None,
                   corr_radius_list=None,
                   prop_radius_list=None,
                   num_reg_refine=1,
                   pred_bidir_flow=False,
                   pred_bwd_flow=False,
                   fwd_bwd_consistency_check=False,
                   device="cpu",
                   **kwargs
                   ):
    fixed_inference_size = inference_size
    transpose_img = False
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # the model is trained with size: width > height
    if image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True

    nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                    int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
    # resize to nearest size or specified size
    inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size
    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)
    if pred_bwd_flow:
        image1, image2 = image2, image1

    results_dict = model(image1, image2,
                            attn_type=attn_type,
                            attn_splits_list=attn_splits_list,
                            corr_radius_list=corr_radius_list,
                            prop_radius_list=prop_radius_list,
                            num_reg_refine=num_reg_refine,
                            task='flow',
                            pred_bidir_flow=pred_bidir_flow,
                            )
    flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
    
    # resize back
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

    if transpose_img:
        flow_pr = torch.transpose(flow_pr, -2, -1)

    flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

    vis_image = flow_to_image(flow)

    # also predict backward flow
    if pred_bidir_flow:
        assert flow_pr.size(0) == 2  # [2, H, W, 2]
        flow_bwd = flow_pr[1].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
        vis_image = flow_to_image(flow_bwd)
        flow = flow_bwd
    return flow, vis_image

MODEL_CONFIGS = {
    "gmflow-scale1": Namespace(
        num_scales=1,
        upsample_factor=8,

        attn_type="swin",
        feature_channels=128,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        
        attn_splits_list=[2],
        corr_radius_list=[-1],
        prop_radius_list=[-1],

        reg_refine=False,
        num_reg_refine=1
    ),
    "gmflow-scale2": Namespace(
        num_scales=2,
        upsample_factor=4,
        padding_factor=32,

        attn_type="swin",
        feature_channels=128,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        
        attn_splits_list=[2, 8],
        corr_radius_list=[-1, 4],
        prop_radius_list=[-1, 1],

        reg_refine=False,
        num_reg_refine=1
    ),
    "gmflow-scale2-regrefine6": Namespace(
        num_scales=2,
        upsample_factor=4,
        padding_factor=32,

        attn_type="swin",
        feature_channels=128,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        
        attn_splits_list=[2, 8],
        corr_radius_list=[-1, 4],
        prop_radius_list=[-1, 1],

        reg_refine=True,
        num_reg_refine=6
    )
}

class UnimatchDetector:
    def __init__(self, unimatch, config_args):
        self.unimatch = unimatch
        self.config_args = config_args
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=UNIMATCH_MODEL_NAME, filename="gmflow-scale2-regrefine6-mixdata.pth"):
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        config_args = None
        for key in list(MODEL_CONFIGS.keys())[::-1]:
            if key in filename:
                config_args = MODEL_CONFIGS[key]
                break
        assert config_args, f"Couldn't find hardcoded Unimatch config for {filename}"
        
        model = UniMatch(feature_channels=config_args.feature_channels,
                        num_scales=config_args.num_scales,
                        upsample_factor=config_args.upsample_factor,
                        num_head=config_args.num_head,
                        ffn_dim_expansion=config_args.ffn_dim_expansion,
                        num_transformer_layers=config_args.num_transformer_layers,
                        reg_refine=config_args.reg_refine,
                        task='flow')

        sd = torch.load(model_path, map_location="cpu")
        model.load_state_dict(sd['model'])
        return cls(model, config_args)

    def to(self, device):
        self.unimatch.to(device)
        self.device = device
        return self
    
    def __call__(self, image1, image2, detect_resolution=512, output_type="pil", upscale_method="INTER_CUBIC", pred_bwd_flow=False, pred_bidir_flow=False, **kwargs):
        assert image1.shape == image2.shape, f"[Unimatch] image1 and image2 must have the same size, got {image1.shape} and {image2.shape}"

        image1, output_type = common_input_validate(image1, output_type, **kwargs)
        #image1, remove_pad = resize_image_with_pad(image1, detect_resolution, upscale_method)
        image2, output_type = common_input_validate(image2, output_type, **kwargs)
        #image2, remove_pad = resize_image_with_pad(image2, detect_resolution, upscale_method)
        with torch.no_grad():
            flow, vis_image = inference_flow(self.unimatch, image1, image2, device=self.device, pred_bwd_flow=pred_bwd_flow, pred_bidir_flow=pred_bidir_flow, **vars(self.config_args))
        
        if output_type == "pil":
            vis_image = Image.fromarray(vis_image)

        return flow, vis_image
