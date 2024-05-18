from ..utils import common_annotator_call, create_node_input_types, MAX_RESOLUTION
import comfy.model_management as model_management

class Metric3D_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            backbone=(["vit-small", "vit-large", "vit-giant2"], {"default": "vit-small"}),
            fx=("INT", {"default": 1000, 'min': 1, 'max': MAX_RESOLUTION}),
            fy=("INT", {"default": 1000, 'min': 1, 'max': MAX_RESOLUTION})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, backbone, fx, fy, resolution=512):
        from controlnet_aux.metric3d import Metric3DDetector
        model = Metric3DDetector.from_pretrained(filename=f"metric_depth_{backbone.replace('-', '_')}_800k.pth").to(model_management.get_torch_device())
        cb = lambda image, **kwargs: model(image, **kwargs)[0]
        out = common_annotator_call(cb, image, resolution=resolution, fx=fx, fy=fy, depth_and_normal=True)
        del model
        return (out, )

class Metric3D_Normal_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            backbone=(["vit-small", "vit-large", "vit-giant2"], {"default": "vit-small"}),
            fx=("INT", {"default": 1000, 'min': 1, 'max': MAX_RESOLUTION}),
            fy=("INT", {"default": 1000, 'min': 1, 'max': MAX_RESOLUTION})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, backbone, fx, fy, resolution=512):
        from controlnet_aux.metric3d import Metric3DDetector
        model = Metric3DDetector.from_pretrained(filename=f"metric_depth_{backbone.replace('-', '_')}_800k.pth").to(model_management.get_torch_device())
        cb = lambda image, **kwargs: model(image, **kwargs)[1]
        out = common_annotator_call(cb, image, resolution=resolution, fx=fx, fy=fy, depth_and_normal=True)
        del model
        return (out, )

NODE_CLASS_MAPPINGS = {
    "Metric3D-DepthMapPreprocessor": Metric3D_Depth_Map_Preprocessor,
    "Metric3D-NormalMapPreprocessor": Metric3D_Normal_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Metric3D-DepthMapPreprocessor": "Metric3D Depth Map",
    "Metric3D-NormalMapPreprocessor": "Metric3D Normal Map"
}
