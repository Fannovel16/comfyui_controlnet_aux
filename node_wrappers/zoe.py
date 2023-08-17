from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME
import controlnet_aux
import comfy.model_management as model_management

class Zoe_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "preprocessors/normal_depth_map"

    def execute(self, image, **kwargs):
        model = controlnet_aux.ZoeDetector.from_pretrained(HF_MODEL_NAME, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        return (common_annotator_call(model, image), )

NODE_CLASS_MAPPINGS = {
    "Zoe-DepthMapPreprocessor": Zoe_Depth_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Zoe-DepthMapPreprocessor": "Zoe - Depth Map"
}