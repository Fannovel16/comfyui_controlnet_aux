from ..utils import common_annotator_call, INPUT, define_preprocessor_inputs
import comfy.model_management as model_management

class Depth_Anything_V2_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            ckpt_name=INPUT.COMBO(
                ["depth_anything_v2_vitg.pth", "depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth", "depth_anything_v2_vits.pth"],
                default="depth_anything_v2_vitl.pth"
            ),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, ckpt_name="depth_anything_v2_vitl.pth", resolution=512, **kwargs):
        from custom_controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector

        model = DepthAnythingV2Detector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, max_depth=1)
        del model
        return (out, )

""" class Depth_Anything_Metric_V2_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            environment=(["indoor", "outdoor"], {"default": "indoor"}),
            max_depth=("FLOAT", {"min": 0, "max": 100, "default": 20.0, "step": 0.01})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, environment, resolution=512, max_depth=20.0, **kwargs):
        from custom_controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector
        filename = dict(indoor="depth_anything_v2_metric_hypersim_vitl.pth", outdoor="depth_anything_v2_metric_vkitti_vitl.pth")[environment]
        model = DepthAnythingV2Detector.from_pretrained(filename=filename).to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, max_depth=max_depth)
        del model
        return (out, ) """

NODE_CLASS_MAPPINGS = {
    "DepthAnythingV2Preprocessor": Depth_Anything_V2_Preprocessor,
    #"Metric_DepthAnythingV2Preprocessor": Depth_Anything_Metric_V2_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthAnythingV2Preprocessor": "Depth Anything V2 - Relative",
    #"Metric_DepthAnythingV2Preprocessor": "Depth Anything V2 - Metric"
}