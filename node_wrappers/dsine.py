from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management

class DSINE_Normal_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            fov=("FLOAT", {"min": 0.0, "max": 365.0, "step": 0.05, "default": 60.0}),
            iterations=("INT", {"min": 1, "max": 20, "step": 1, "default": 5})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, fov, iterations, resolution=512, **kwargs):
        from controlnet_aux.dsine import DsineDetector

        model = DsineDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, fov=fov, iterations=iterations, resolution=resolution)
        del model
        return (out,)

NODE_CLASS_MAPPINGS = {
    "DSINE-NormalMapPreprocessor": DSINE_Normal_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DSINE-NormalMapPreprocessor": "DSINE Normal Map"
}