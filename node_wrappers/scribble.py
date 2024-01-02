from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management

class Scribble_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types()

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, resolution=512, **kwargs):
        from controlnet_aux.scribble import ScribbleDetector

        model = ScribbleDetector()
        return (common_annotator_call(model, image, resolution=resolution), )

class Scribble_XDoG_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            threshold = ("INT", {"default": 32, "min": 1, "max": 64, "step": 64})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, resolution=512, **kwargs):
        from controlnet_aux.scribble import ScribbleXDog_Detector

        model = ScribbleXDog_Detector()
        return (common_annotator_call(model, image, resolution=resolution), )

NODE_CLASS_MAPPINGS = {
    "ScribblePreprocessor": Scribble_Preprocessor,
    "Scribble_XDoG_Preprocessor": Scribble_XDoG_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ScribblePreprocessor": "Scribble Lines",
    "Scribble_XDoG_Preprocessor": "Scribble XDoG Lines"
}