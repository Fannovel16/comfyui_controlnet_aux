from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management

class BAE_Normal_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(resolution=INPUT.RESOLUTION())

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, resolution=512, **kwargs):
        from custom_controlnet_aux.normalbae import NormalBaeDetector

        model = NormalBaeDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out,)

NODE_CLASS_MAPPINGS = {
    "BAE-NormalMapPreprocessor": BAE_Normal_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAE-NormalMapPreprocessor": "BAE Normal Map"
}