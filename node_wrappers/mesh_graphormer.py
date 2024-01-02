from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management
import numpy as np
import torch
from einops import rearrange
import os, sys
import subprocess, threading

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

class Mesh_Graphormer_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            mask_bbox_padding=("INT", {"default": 30, "min": 0, "max": 100})
        )

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "INPAINTING_MASK")
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, mask_bbox_padding=30, resolution=512, **kwargs):
        install_deps()
        from controlnet_aux.mesh_graphormer import MeshGraphormerDetector
        model = MeshGraphormerDetector.from_pretrained().to(model_management.get_torch_device())
        
        depth_map_list = []
        mask_list = []
        for single_image in image:
            np_image = np.asarray(single_image.cpu() * 255., dtype=np.uint8)
            depth_map, mask, info = model(np_image, output_type="np", detect_resolution=resolution, mask_bbox_padding=mask_bbox_padding)
            depth_map_list.append(torch.from_numpy(depth_map.astype(np.float32) / 255.0))
            mask_list.append(torch.from_numpy(mask[:, :, :1].astype(np.float32) / 255.0))
        return torch.stack(depth_map_list, dim=0), rearrange(torch.stack(mask_list, dim=0), "n h w 1 -> n 1 h w")
    
NODE_CLASS_MAPPINGS = {
    "MeshGraphormer-DepthMapPreprocessor": Mesh_Graphormer_Depth_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshGraphormer-DepthMapPreprocessor": "MeshGraphormer Hand Refiner"
}