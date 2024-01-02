from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management
import numpy as np

class MIDAS_Normal_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            a =  ("FLOAT", {"default": np.pi * 2.0, "min": 0.0, "max": np.pi * 5.0, "step": 0.05}),
            bg_threshold = ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.05})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, a, bg_threshold, resolution=512, **kwargs):
        from controlnet_aux.midas import MidasDetector

        model = MidasDetector.from_pretrained().to(model_management.get_torch_device())
        #Dirty hack :))
        cb = lambda image, **kargs: model(image, **kargs)[1]
        out = common_annotator_call(cb, image, resolution=resolution, a=a, bg_th=bg_threshold, depth_and_normal=True)
        del model
        return (out, )

class MIDAS_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            a =  ("FLOAT", {"default": np.pi * 2.0, "min": 0.0, "max": np.pi * 5.0, "step": 0.05}),
            bg_threshold = ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.05})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, a, bg_threshold, resolution=512, **kwargs):
        from controlnet_aux.midas import MidasDetector

        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_depth2image.py
        model = MidasDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, a=a, bg_th=bg_threshold)
        del model
        return (out, )

NODE_CLASS_MAPPINGS = {
    "MiDaS-NormalMapPreprocessor": MIDAS_Normal_Map_Preprocessor,
    "MiDaS-DepthMapPreprocessor": MIDAS_Depth_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiDaS-NormalMapPreprocessor": "MiDaS Normal Map",
    "MiDaS-DepthMapPreprocessor": "MiDaS Depth Map"
}