from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management

class Shuffle_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types()
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"

    CATEGORY = "ControlNet Preprocessors/T2IAdapter-only"

    def preprocess(self, image, resolution=512):
        from controlnet_aux.shuffle import ContentShuffleDetector

        return (common_annotator_call(ContentShuffleDetector(), image, resolution=resolution), )

NODE_CLASS_MAPPINGS = {
    "ShufflePreprocessor": Shuffle_Preprocessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShufflePreprocessor": "Content Shuffle"
}