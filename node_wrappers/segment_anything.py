from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management

class SAM_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(resolution=INPUT.RESOLUTION())

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/others"

    def execute(self, image, resolution=512, **kwargs):
        from custom_controlnet_aux.sam import SamDetector

        mobile_sam = SamDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(mobile_sam, image, resolution=resolution)
        del mobile_sam
        return (out, )

NODE_CLASS_MAPPINGS = {
    "SAMPreprocessor": SAM_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMPreprocessor": "SAM Segmentor"
}