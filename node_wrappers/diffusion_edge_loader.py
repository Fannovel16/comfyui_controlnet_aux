from ..utils import INPUT
import comfy.model_management as model_management
import comfy.model_patcher
from custom_controlnet_aux.diffusion_edge import DiffusionEdgeDetector
import torch
import torch.nn as nn

class DiffusionEdgeWrapper(nn.Module):
    """
    Thin nn.Module facade so ModelPatcher can treat DiffusionEdgeDetector
    like a regular torch model.
    """
    def __init__(self, detector):
        super().__init__()
        self.detector = detector    # the real pre‑processor network

    # ComfyUI samplers call this through patcher.model(...)
    def forward(self, *args, **kwargs):
        return self.detector(*args, **kwargs)

    # Needed only for size / VRAM bookkeeping
    def state_dict(self, *a, **kw):
        return self.detector.model.model.state_dict(*a, **kw)

    # Let Patcher / model_management move it between CPU↔GPU
    def to(self, device, *a, **kw):
        if hasattr(self.detector, "to"):
            self.detector.to(device)
        if hasattr(self.detector.model, "to"):
            self.detector.model.to(device)
        if hasattr(self.detector.model.model, "to"):
            self.detector.model.model.to(device)    
        return super().to(device, *a, **kw)
        

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
        wrapped  = DiffusionEdgeWrapper(model)
        load_device = model_management.get_torch_device()
        offload_device = torch.device("cpu")
        patcher = comfy.model_patcher.ModelPatcher(wrapped, load_device=load_device, offload_device=offload_device)
        return (patcher,)

NODE_CLASS_MAPPINGS = {
    "DiffusionEdgeModelLoader": DiffusionEdgeModelLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionEdgeModelLoader": "Diffusion Edge Model Loader",
}
