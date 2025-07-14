from ..utils import INPUT
import comfy.model_management as model_management
import comfy.model_patcher
from custom_controlnet_aux.diffusion_edge import DiffusionEdgeDetector
import torch

class DiffusionEdgeModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "environment": (["indoor", "urban", "natrual"], {"default": "indoor"})
            }
        }

    RETURN_TYPES = ("DIFFUSION_EDGE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def load_model(self, environment="indoor"):
        model = DiffusionEdgeDetector.from_pretrained(filename=f"diffusion_edge_{environment}.pt")
        load_device = model_management.get_torch_device()
        offload_device = torch.device("cpu")
        patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        return (patcher,)

NODE_CLASS_MAPPINGS = {
    "DiffusionEdgeModelLoader": DiffusionEdgeModelLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionEdgeModelLoader": "Diffusion Edge Model Loader",
}
