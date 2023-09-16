from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME, create_node_input_types
import comfy.model_management as model_management

class Zoe_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types()

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Map"

    def execute(self, image, resolution, **kwargs):
        from controlnet_aux.zoe import ZoeDetector

        model = ZoeDetector.from_pretrained(HF_MODEL_NAME, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out, )

NODE_CLASS_MAPPINGS = {
    "Zoe-DepthMapPreprocessor": Zoe_Depth_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Zoe-DepthMapPreprocessor": "Zoe - Depth Map"
}