from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME
from controlnet_aux.hed import HEDdetector
import comfy.model_management as model_management

class HED_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE", )},
            "optional": {
                "safe": (["enable", "disable"], {"default": "enable"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors"

    def execute(self, image, **kwargs):
        model = HEDdetector.from_pretrained(HF_MODEL_NAME, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        return (common_annotator_call(model, image, safe = kwargs["safe"] == "enable"), )

class Fake_Scribble_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE",)}, 
            "optional": {
                "safe": (["enable", "disable"], {"default": "enable"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors"

    def execute(self, image, **kwargs):
        model = HEDdetector.from_pretrained(HF_MODEL_NAME, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        return (common_annotator_call(model, image, scribble=True, safe=kwargs["safe"]=="enable"), )

NODE_CLASS_MAPPINGS = {
    "HEDPreprocessor": HED_Preprocessor,
    "FakeScribblePreprocessor": Fake_Scribble_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HEDPreprocessor": "HED Lines",
    "FakeScribblePreprocessor": "Fake Scribble Lines (aka scribble_hed)"
}