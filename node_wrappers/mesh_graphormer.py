from ..utils import common_annotator_call, create_node_input_types, MAX_RESOLUTION
import comfy.model_management as model_management
import numpy as np
import torch
from einops import rearrange
import os, sys
import subprocess, threading
import scipy.ndimage

#Ref: https://github.com/ltdrdata/ComfyUI-Manager/blob/284e90dc8296a2e1e4f14b4b2d10fba2f52f0e53/__init__.py#L14
def handle_stream(stream, prefix):
    for line in stream:
        print(prefix, line, end="")


def run_script(cmd, cwd='.'):
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    stdout_thread = threading.Thread(target=handle_stream, args=(process.stdout, ""))
    stderr_thread = threading.Thread(target=handle_stream, args=(process.stderr, "[!]"))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()

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

#Based on https://github.com/comfyanonymous/ComfyUI/blob/8c6493578b3dda233e9b9a953feeaf1e6ca434ad/comfy_extras/nodes_mask.py#L309
def expand_mask(mask, expand, tapered_corners):
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                        [1, 1, 1],
                        [c, 1, c]])
    for _ in range(abs(expand)):
        if expand < 0:
            mask = scipy.ndimage.grey_erosion(mask, footprint=kernel)
        else:
            mask = scipy.ndimage.grey_dilation(mask, footprint=kernel)
    return mask

class Mesh_Graphormer_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        orig =  create_node_input_types(
            mask_bbox_padding=("INT", {"default": 30, "min": 0, "max": 100}),
        )
        orig = {
            **orig,
            **dict(
                mask_type=(["based_on_depth", "original"], {"default": "based_on_depth"}),
                mask_expand=("INT", {"default": 5, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                rand_seed=("INT", {"default": 88, "min": 0, "max": 0xffffffffffffffff})
            )
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "INPAINTING_MASK")
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, mask_bbox_padding=30, mask_type="based_on_depth", mask_expand=5, resolution=512, rand_seed=88, **kwargs):
        install_deps()
        from controlnet_aux.mesh_graphormer import MeshGraphormerDetector
        model = MeshGraphormerDetector.from_pretrained().to(model_management.get_torch_device())
        
        depth_map_list = []
        mask_list = []
        for single_image in image:
            np_image = np.asarray(single_image.cpu() * 255., dtype=np.uint8)
            depth_map, mask, info = model(np_image, output_type="np", detect_resolution=resolution, mask_bbox_padding=mask_bbox_padding, seed=rand_seed)
            
            if mask_type == "based_on_depth":
                mask = depth_map.copy()
                mask[mask > 0] == 1
            if mask_expand != 0:
                mask = expand_mask(mask, mask_expand, tapered_corners=True)

            depth_map_list.append(torch.from_numpy(depth_map.astype(np.float32) / 255.0))
            mask_list.append(torch.from_numpy(mask[:, :, :1].astype(np.float32) / 255.0))
        return torch.stack(depth_map_list, dim=0), rearrange(torch.stack(mask_list, dim=0), "n h w 1 -> n 1 h w")
    
NODE_CLASS_MAPPINGS = {
    "MeshGraphormer-DepthMapPreprocessor": Mesh_Graphormer_Depth_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshGraphormer-DepthMapPreprocessor": "MeshGraphormer Hand Refiner"
}