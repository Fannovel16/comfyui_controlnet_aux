from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management

class PIDINET_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            safe=INPUT.COMBO(["enable", "disable"]),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, safe, resolution=512, **kwargs):
        from custom_controlnet_aux.pidi import PidiNetDetector

        model = PidiNetDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, safe = safe == "enable")
        del model
        return (out, )

NODE_CLASS_MAPPINGS = {
    "PiDiNetPreprocessor": PIDINET_Preprocessor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDiNetPreprocessor": "PiDiNet Soft-Edge Lines"
}