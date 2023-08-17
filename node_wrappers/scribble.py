from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME
import controlnet_aux
import comfy.model_management as model_management

class Scribble_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "preprocessors/edge_line"

    def execute(self, image, **kwargs):
        model = controlnet_aux.PickyScribble()
        return (common_annotator_call(model, image), )

NODE_CLASS_MAPPINGS = {
    "ScribblePreprocessor": Scribble_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ScribblePreprocessor": "Scribble Lines"
}