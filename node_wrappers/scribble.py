from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME
from controlnet_aux.scribble import ScribbleDetector, ScribbleXDog_Detector
import comfy.model_management as model_management

class Scribble_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors"

    def execute(self, image, **kwargs):
        model = ScribbleDetector()
        return (common_annotator_call(model, image), )

class Scribble_XDoG_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("INT", {"default": 32, "min": 1, "max": 64, "step": 64})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors"

    def execute(self, image, **kwargs):
        model = ScribbleXDog_Detector()
        return (common_annotator_call(model, image), )

NODE_CLASS_MAPPINGS = {
    "ScribblePreprocessor": Scribble_Preprocessor,
    "Scribble_XDoG_Preprocessor": Scribble_XDoG_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ScribblePreprocessor": "Scribble Lines",
    "Scribble_XDoG_Preprocessor": "Scribble XDoG Lines"
}