from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management
import numpy as np

class MLSD_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            score_threshold=INPUT.FLOAT(default=0.1, min=0.01, max=2.0),
            dist_threshold=INPUT.FLOAT(default=0.1, min=0.01, max=20.0),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, score_threshold, dist_threshold, resolution=512, **kwargs):
        from custom_controlnet_aux.mlsd import MLSDdetector

        model = MLSDdetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, thr_v=score_threshold, thr_d=dist_threshold)
        return (out, )

NODE_CLASS_MAPPINGS = {
    "M-LSDPreprocessor": MLSD_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "M-LSDPreprocessor": "M-LSD Lines"
}