from ..utils import common_annotator_call, INPUT, define_preprocessor_inputs
import comfy.model_management as model_management

class PyraCanny_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            low_threshold=INPUT.INT(default=64, max=255),
            high_threshold=INPUT.INT(default=128, max=255),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, low_threshold=64, high_threshold=128, resolution=512, **kwargs):
        from custom_controlnet_aux.pyracanny import PyraCannyDetector

        return (common_annotator_call(PyraCannyDetector(), image, low_threshold=low_threshold, high_threshold=high_threshold, resolution=resolution), )



NODE_CLASS_MAPPINGS = {
    "PyraCannyPreprocessor": PyraCanny_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PyraCannyPreprocessor": "PyraCanny"
}