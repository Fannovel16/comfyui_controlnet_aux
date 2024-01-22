from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management

class Depth_Anything_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            ckpt_name=(["depth_anything_vitl14.path"], {"default": "depth_anything_vitl14.pth"})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, resolution=512, **kwargs):
        from controlnet_aux.depth_anything import DepthAnythingDetector

        model = DepthAnythingDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution)
        del model
        return (out, )

NODE_CLASS_MAPPINGS = {
    "DepthAnythingPreprocessor": Depth_Anything_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnythingPreprocessor": "Depth Anything"
}