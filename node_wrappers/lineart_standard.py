from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management

class Lineart_Standard_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            guassian_sigma=("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0}),
            intensity_threshold=("INT", {"default": 8, "min": 0, "max": 16})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, guassian_sigma, intensity_threshold, resolution=512, **kwargs):
        from controlnet_aux.lineart_standard import LineartStandardDetector
        return (common_annotator_call(LineartStandardDetector(), image, guassian_sigma=guassian_sigma, intensity_threshold=intensity_threshold, resolution=resolution), )

NODE_CLASS_MAPPINGS = {
    "LineartStandardPreprocessor": Lineart_Standard_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LineartStandardPreprocessor": "Standard Lineart"
}