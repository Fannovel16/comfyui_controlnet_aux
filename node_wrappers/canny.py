from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management

class Canny_Edge_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            low_threshold=("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
            high_threshold=("INT", {"default": 200, "min": 0, "max": 255, "step": 1})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, low_threshold, high_threshold, resolution=512, **kwargs):
        from controlnet_aux.canny import CannyDetector

        return (common_annotator_call(CannyDetector(), image, low_threshold=low_threshold, high_threshold=high_threshold, resolution=resolution), )



NODE_CLASS_MAPPINGS = {
    "CannyEdgePreprocessor": Canny_Edge_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CannyEdgePreprocessor": "Canny Edge"
}