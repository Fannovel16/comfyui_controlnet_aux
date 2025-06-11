from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management

class Lineart_Standard_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            guassian_sigma=INPUT.FLOAT(default=6.0, max=100.0),
            intensity_threshold=INPUT.INT(default=8, max=16),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, guassian_sigma=6, intensity_threshold=8, resolution=512, **kwargs):
        from custom_controlnet_aux.lineart_standard import LineartStandardDetector
        return (common_annotator_call(LineartStandardDetector(), image, guassian_sigma=guassian_sigma, intensity_threshold=intensity_threshold, resolution=resolution), )

NODE_CLASS_MAPPINGS = {
    "LineartStandardPreprocessor": Lineart_Standard_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LineartStandardPreprocessor": "Standard Lineart"
}