from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME
import comfy.model_management as model_management

class Color_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 512, "min": 1, "max": 2048, "step": 1})
                }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/T2IAdapter-only"

    def execute(self, image, resolution, **kwargs):
        from controlnet_aux.color import ColorDetector

        return (common_annotator_call(ColorDetector(), image, detect_resolution=resolution), )



NODE_CLASS_MAPPINGS = {
    "ColorPreprocessor": Color_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorPreprocessor": "Color Pallete"
}