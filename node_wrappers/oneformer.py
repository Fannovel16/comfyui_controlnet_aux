from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME, DWPOSE_MODEL_NAME
import controlnet_aux
import comfy.model_management as model_management

class OneFormer_COCO_SemSegPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "semantic_segmentate"

    CATEGORY = "preprocessors/semseg"

    def semantic_segmentate(self, image):
        model = controlnet_aux.OneformerSegmentor.from_pretrained(HF_MODEL_NAME, filename="150_16_swin_l_oneformer_coco_100ep.pth", cache_dir=annotator_ckpts_path)
        model = model.to(model_management.get_torch_device())
        return (common_annotator_call(model, image), )


class OneFormer_ADE20K_SemSegPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "semantic_segmentate"

    CATEGORY = "preprocessors/semseg"

    def semantic_segmentate(self, image):
        model = controlnet_aux.OneformerSegmentor.from_pretrained(HF_MODEL_NAME, filename="250_16_swin_l_oneformer_ade20k_160k.pth", cache_dir=annotator_ckpts_path)
        model = model.to(model_management.get_torch_device())
        return (common_annotator_call(model, image), )


NODE_CLASS_MAPPINGS = {
    "OneFormer-COCO-SemSegPreprocessor": OneFormer_COCO_SemSegPreprocessor,
    "OneFormer-ADE20K-SemSegPreprocessor": OneFormer_ADE20K_SemSegPreprocessor
}