import cv2
import numpy as np
from PIL import Image
from custom_controlnet_aux.util import resize_image_with_pad, common_input_validate, HWC3, custom_hf_download, MESH_GRAPHORMER_MODEL_NAME
from custom_controlnet_aux.mesh_graphormer.pipeline import MeshGraphormerMediapipe, args
import random, torch

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

class MeshGraphormerDetector:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path=MESH_GRAPHORMER_MODEL_NAME, filename="graphormer_hand_state_dict.bin", hrnet_filename="hrnetv2_w64_imagenet_pretrained.pth", detect_thr=0.6, presence_thr=0.6):
        args.resume_checkpoint = custom_hf_download(pretrained_model_or_path, filename)
        args.hrnet_checkpoint = custom_hf_download(pretrained_model_or_path, hrnet_filename)
        pipeline = MeshGraphormerMediapipe(args, detect_thr=detect_thr, presence_thr=presence_thr)
        return cls(pipeline)
    
    def to(self, device):
        self.pipeline._model.to(device)
        self.pipeline.mano_model.to(device)
        self.pipeline.mano_model.layer.to(device)
        return self

    def __call__(self, input_image=None, mask_bbox_padding=30, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", seed=88, **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        set_seed(seed, 0)
        depth_map, mask, info = self.pipeline.get_depth(input_image, mask_bbox_padding)
        if depth_map is None:
            depth_map = np.zeros_like(input_image)
            mask = np.zeros_like(input_image)

        #The hand is small
        depth_map, mask = HWC3(depth_map), HWC3(mask)
        depth_map, remove_pad = resize_image_with_pad(depth_map, detect_resolution, upscale_method)
        depth_map = remove_pad(depth_map)
        if output_type == "pil":
            depth_map = Image.fromarray(depth_map)
            mask = Image.fromarray(mask)
            
        return depth_map, mask, info
