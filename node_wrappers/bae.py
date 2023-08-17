from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME
from controlnet_aux.normalbae import NormalBaeDetector
import comfy.model_management as model_management

class BAE_Normal_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "preprocessors/normal_depth_map"

    def execute(self, image, **kwargs):
        model = NormalBaeDetector.from_pretrained(HF_MODEL_NAME, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        return (common_annotator_call(model, image),)

NODE_CLASS_MAPPINGS = {
    "BAE-NormalMapPreprocessor": BAE_Normal_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BAE-NormalMapPreprocessor": "BAE - Normal Map"
}