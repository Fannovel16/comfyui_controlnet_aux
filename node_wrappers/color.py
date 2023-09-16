from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME, create_node_input_types
import comfy.model_management as model_management

class Color_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types()

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/T2IAdapter-only"

    def execute(self, image, resolution, **kwargs):
        from controlnet_aux.color import ColorDetector

        return (common_annotator_call(ColorDetector(), image, resolution=resolution), )



NODE_CLASS_MAPPINGS = {
    "ColorPreprocessor": Color_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorPreprocessor": "Color Pallete"
}