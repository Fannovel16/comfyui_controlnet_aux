from ..utils import common_annotator_call, create_node_input_types, MAX_RESOLUTION
import comfy.model_management as model_management

class Shuffle_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": dict(
                image=("IMAGE",),
                resolution=("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                seed=("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            )
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"

    CATEGORY = "ControlNet Preprocessors/T2IAdapter-only"

    def preprocess(self, image, resolution=512, seed=None):
        from controlnet_aux.shuffle import ContentShuffleDetector

        return (common_annotator_call(ContentShuffleDetector(), image, resolution=resolution, seed=seed), )

NODE_CLASS_MAPPINGS = {
    "ShufflePreprocessor": Shuffle_Preprocessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShufflePreprocessor": "Content Shuffle"
}