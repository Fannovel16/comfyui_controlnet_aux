from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT, MAX_RESOLUTION
import comfy.model_management as model_management

class Shuffle_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            resolution=INPUT.RESOLUTION(),
            seed=INPUT.SEED()
        )
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"

    CATEGORY = "ControlNet Preprocessors/T2IAdapter-only"

    def preprocess(self, image, resolution=512, seed=0):
        from custom_controlnet_aux.shuffle import ContentShuffleDetector

        return (common_annotator_call(ContentShuffleDetector(), image, resolution=resolution, seed=seed), )

NODE_CLASS_MAPPINGS = {
    "ShufflePreprocessor": Shuffle_Preprocessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ShufflePreprocessor": "Content Shuffle"
}