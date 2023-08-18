from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME
from controlnet_aux.lineart import LineartDetector
import comfy.model_management as model_management

class LineArt_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE",), },
            "optional": {
                "coarse": (["disable", "enable"], {"default": "disable"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, **kwargs):
        model = LineartDetector.from_pretrained(HF_MODEL_NAME, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        out = common_annotator_call(model, image, coarse = kwargs["coarse"] == "enable")
        del model
        return (out, )

NODE_CLASS_MAPPINGS = {
    "LineArtPreprocessor": LineArt_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LineArtPreprocessor": "Realistic Lineart"
}