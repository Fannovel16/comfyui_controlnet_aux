from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management

class Uniformer_SemSegPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types()

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "semantic_segmentate"

    CATEGORY = "ControlNet Preprocessors/Semantic Segmentation"

    def semantic_segmentate(self, image, resolution=512):
        from controlnet_aux.uniformer import UniformerSegmentor

        model = UniformerSegmentor.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out, )

NODE_CLASS_MAPPINGS = {
    "UniFormer-SemSegPreprocessor": Uniformer_SemSegPreprocessor,
    "SemSegPreprocessor": Uniformer_SemSegPreprocessor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniFormer-SemSegPreprocessor": "UniFormer Segmentor",
    "SemSegPreprocessor": "Semantic Segmentor (legacy, alias for UniFormer)",
}