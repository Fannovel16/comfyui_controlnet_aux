from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME
from controlnet_aux.mlsd import MLSDdetector
import comfy.model_management as model_management
import numpy as np

class MLSD_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "score_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "dist_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "preprocessors/edge_line"

    def execute(self, image, score_threshold, dist_threshold, **kwargs):
        model = MLSDdetector.from_pretrained(HF_MODEL_NAME, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        return (common_annotator_call(model, image, thr_v=score_threshold, thr_d=dist_threshold), )

NODE_CLASS_MAPPINGS = {
    "M-LSDPreprocessor": MLSD_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "M-LSDPreprocessor": "M-LSD Lines"
}