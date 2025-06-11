from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT, MAX_RESOLUTION, run_script
import comfy.model_management as model_management
import numpy as np
import torch
from einops import rearrange
import os, sys
import subprocess, threading
import scipy.ndimage
import cv2
import torch.nn.functional as F

def install_deps():
    try:
        import mediapipe
    except ImportError:
        run_script([sys.executable, '-s', '-m', 'pip', 'install', 'mediapipe'])
        run_script([sys.executable, '-s', '-m', 'pip', 'install', '--upgrade', 'protobuf'])
    
    try:
        import trimesh
    except ImportError:
        run_script([sys.executable, '-s', '-m', 'pip', 'install', 'trimesh[easy]'])

#Sauce: https://github.com/comfyanonymous/ComfyUI/blob/8c6493578b3dda233e9b9a953feeaf1e6ca434ad/comfy_extras/nodes_mask.py#L309
def expand_mask(mask, expand, tapered_corners):
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                        [1, 1, 1],
                        [c, 1, c]])
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    return torch.stack(out, dim=0)

class Mesh_Graphormer_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            mask_bbox_padding=("INT", {"default": 30, "min": 0, "max": 100}),
            resolution=INPUT.RESOLUTION(),
            mask_type=INPUT.COMBO(["based_on_depth", "tight_bboxes", "original"]),
            mask_expand=INPUT.INT(default=5, min=-MAX_RESOLUTION, max=MAX_RESOLUTION),
            rand_seed=INPUT.INT(default=88, min=0, max=0xffffffffffffffff),
            detect_thr=INPUT.FLOAT(default=0.6, min=0.1),
            presence_thr=INPUT.FLOAT(default=0.6, min=0.1)
        )

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "INPAINTING_MASK")
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, mask_bbox_padding=30, mask_type="based_on_depth", mask_expand=5, resolution=512, rand_seed=88, detect_thr=0.6, presence_thr=0.6, **kwargs):
        install_deps()
        from custom_controlnet_aux.mesh_graphormer import MeshGraphormerDetector
        model = kwargs["model"] if "model" in kwargs \
            else MeshGraphormerDetector.from_pretrained(detect_thr=detect_thr, presence_thr=presence_thr).to(model_management.get_torch_device())
        
        depth_map_list = []
        mask_list = []
        for single_image in image:
            np_image = np.asarray(single_image.cpu() * 255., dtype=np.uint8)
            depth_map, mask, info = model(np_image, output_type="np", detect_resolution=resolution, mask_bbox_padding=mask_bbox_padding, seed=rand_seed)
            if mask_type == "based_on_depth":
                H, W = mask.shape[:2]
                mask = cv2.resize(depth_map.copy(), (W, H))
                mask[mask > 0] = 255

            elif mask_type == "tight_bboxes":
                mask = np.zeros_like(mask)
                hand_bboxes = (info or {}).get("abs_boxes") or []
                for hand_bbox in hand_bboxes: 
                    x_min, x_max, y_min, y_max = hand_bbox
                    mask[y_min:y_max+1, x_min:x_max+1, :] = 255 #HWC

            mask = mask[:, :, :1]
            depth_map_list.append(torch.from_numpy(depth_map.astype(np.float32) / 255.0))
            mask_list.append(torch.from_numpy(mask.astype(np.float32) / 255.0))
        depth_maps, masks = torch.stack(depth_map_list, dim=0), rearrange(torch.stack(mask_list, dim=0), "n h w 1 -> n 1 h w")
        return depth_maps, expand_mask(masks, mask_expand, tapered_corners=True)

def normalize_size_base_64(w, h):
    short_side = min(w, h)
    remainder = short_side % 64
    return short_side - remainder + (64 if remainder > 0 else 0)

class Mesh_Graphormer_With_ImpactDetector_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        types = define_preprocessor_inputs(
            # Impact pack
            bbox_threshold=INPUT.FLOAT(default=0.5, min=0.1),
            bbox_dilation=INPUT.INT(default=10, min=-512, max=512),
            bbox_crop_factor=INPUT.FLOAT(default=3.0, min=1.0, max=10.0),
            drop_size=INPUT.INT(default=10, min=1, max=MAX_RESOLUTION),
            # Mesh Graphormer
            mask_bbox_padding=INPUT.INT(default=30, min=0, max=100),
            mask_type=INPUT.COMBO(["based_on_depth", "tight_bboxes", "original"]),
            mask_expand=INPUT.INT(default=5, min=-MAX_RESOLUTION, max=MAX_RESOLUTION),
            rand_seed=INPUT.INT(default=88, min=0, max=0xffffffffffffffff),
            resolution=INPUT.RESOLUTION()
        )
        types["required"]["bbox_detector"] = ("BBOX_DETECTOR", )
        return types
     
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "INPAINTING_MASK")
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, bbox_detector, bbox_threshold=0.5, bbox_dilation=10, bbox_crop_factor=3.0, drop_size=10, resolution=512, **mesh_graphormer_kwargs):
        install_deps()
        from custom_controlnet_aux.mesh_graphormer import MeshGraphormerDetector
        mesh_graphormer_node = Mesh_Graphormer_Depth_Map_Preprocessor()
        model = MeshGraphormerDetector.from_pretrained(detect_thr=0.6, presence_thr=0.6).to(model_management.get_torch_device())
        mesh_graphormer_kwargs["model"] = model

        frames = image
        depth_maps, masks = [], []
        for idx in range(len(frames)):
            frame = frames[idx:idx+1,...] #Impact Pack's BBOX_DETECTOR only supports single batch image
            bbox_detector.setAux('face') # make default prompt as 'face' if empty prompt for CLIPSeg
            _, segs = bbox_detector.detect(frame, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)
            bbox_detector.setAux(None)

            n, h, w, _ = frame.shape
            depth_map, mask = torch.zeros_like(frame), torch.zeros(n, 1, h, w)
            for i, seg in enumerate(segs):
                x1, y1, x2, y2 = seg.crop_region
                cropped_image = frame[:, y1:y2, x1:x2, :]  # Never use seg.cropped_image to handle overlapping area
                mesh_graphormer_kwargs["resolution"] = 0 #Disable resizing
                sub_depth_map, sub_mask = mesh_graphormer_node.execute(cropped_image, **mesh_graphormer_kwargs)
                depth_map[:, y1:y2, x1:x2, :] = sub_depth_map
                mask[:, :, y1:y2, x1:x2] = sub_mask
            
            depth_maps.append(depth_map)
            masks.append(mask)
            
        return (torch.cat(depth_maps), torch.cat(masks))
    
NODE_CLASS_MAPPINGS = {
    "MeshGraphormer-DepthMapPreprocessor": Mesh_Graphormer_Depth_Map_Preprocessor,
    "MeshGraphormer+ImpactDetector-DepthMapPreprocessor": Mesh_Graphormer_With_ImpactDetector_Depth_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshGraphormer-DepthMapPreprocessor": "MeshGraphormer Hand Refiner",
    "MeshGraphormer+ImpactDetector-DepthMapPreprocessor": "MeshGraphormer Hand Refiner With External Detector"
}