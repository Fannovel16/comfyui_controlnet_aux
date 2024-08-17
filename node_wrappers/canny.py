from ..utils import common_annotator_call, INPUT, define_preprocessor_inputs
import comfy.model_management as model_management

class Canny_Edge_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            low_threshold=INPUT.INT(default=100, max=255),
            high_threshold=INPUT.INT(default=200, max=255),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, low_threshold=100, high_threshold=200, resolution=512, **kwargs):
        from custom_controlnet_aux.canny import CannyDetector

        return (common_annotator_call(CannyDetector(), image, low_threshold=low_threshold, high_threshold=high_threshold, resolution=resolution), )



NODE_CLASS_MAPPINGS = {
    "CannyEdgePreprocessor": Canny_Edge_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CannyEdgePreprocessor": "Canny Edge"
}