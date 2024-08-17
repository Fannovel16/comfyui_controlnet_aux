from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management

class LERES_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            rm_nearest=INPUT.FLOAT(max=100.0),
            rm_background=INPUT.FLOAT(max=100.0),
            boost=INPUT.COMBO(["disable", "enable"]),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, rm_nearest=0, rm_background=0, resolution=512, boost="disable", **kwargs):
        from custom_controlnet_aux.leres import LeresDetector

        model = LeresDetector.from_pretrained().to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, thr_a=rm_nearest, thr_b=rm_background, boost=boost == "enable")
        del model
        return (out, )
    
NODE_CLASS_MAPPINGS = {
    "LeReS-DepthMapPreprocessor": LERES_Depth_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LeReS-DepthMapPreprocessor": "LeReS Depth Map (enable boost for leres++)"
}