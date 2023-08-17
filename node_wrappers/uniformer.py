from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME
from controlnet_aux.uniformer import UniformerSegmentor
import comfy.model_management as model_management

class Uniformer_SemSegPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "semantic_segmentate"

    CATEGORY = "ControlNet Preprocessors"

    def semantic_segmentate(self, image):
        model = UniformerSegmentor.from_pretrained(HF_MODEL_NAME, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        return (common_annotator_call(model, image), )

NODE_CLASS_MAPPINGS = {
    "UniFormer-SemSegPreprocessor": Uniformer_SemSegPreprocessor,
    "SemSegPreprocessor": Uniformer_SemSegPreprocessor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniFormer-SemSegPreprocessor": "UniFormer Segmentor",
    "SemSegPreprocessor": "Semantic Segmentor (legacy, aka UniFormer)",
}