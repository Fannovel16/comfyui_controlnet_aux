from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management

class OneFormer_COCO_SemSegPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types()

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "semantic_segmentate"

    CATEGORY = "ControlNet Preprocessors/Semantic Segmentation"

    def semantic_segmentate(self, image, resolution=512):
        from controlnet_aux.oneformer import OneformerSegmentor

        model = OneformerSegmentor.from_pretrained(filename="150_16_swin_l_oneformer_coco_100ep.pth")
        model = model.to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out,)

class OneFormer_ADE20K_SemSegPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types()

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "semantic_segmentate"

    CATEGORY = "ControlNet Preprocessors/Semantic Segmentation"

    def semantic_segmentate(self, image, resolution=512):
        from controlnet_aux.oneformer import OneformerSegmentor

        model = OneformerSegmentor.from_pretrained(filename="250_16_swin_l_oneformer_ade20k_160k.pth")
        model = model.to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out,)

NODE_CLASS_MAPPINGS = {
    "OneFormer-COCO-SemSegPreprocessor": OneFormer_COCO_SemSegPreprocessor,
    "OneFormer-ADE20K-SemSegPreprocessor": OneFormer_ADE20K_SemSegPreprocessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OneFormer-COCO-SemSegPreprocessor": "OneFormer COCO Segmentor",
    "OneFormer-ADE20K-SemSegPreprocessor": "OneFormer ADE20K Segmentor"
}