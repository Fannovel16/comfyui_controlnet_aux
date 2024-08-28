from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management

class DSINE_Normal_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            fov=INPUT.FLOAT(max=365.0, default=60.0),
            iterations=INPUT.INT(min=1, max=20, default=5),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, fov=60.0, iterations=5, resolution=512, **kwargs):
        from custom_controlnet_aux.dsine import DsineDetector

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