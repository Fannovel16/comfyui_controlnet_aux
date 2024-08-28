from ..utils import common_annotator_call, INPUT, define_preprocessor_inputs
import comfy.model_management as model_management

class DensePose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            model=INPUT.COMBO(["densepose_r50_fpn_dl.torchscript", "densepose_r101_fpn_dl.torchscript"]),
            cmap=INPUT.COMBO(["Viridis (MagicAnimate)", "Parula (CivitAI)"]),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses Estimators"

    def execute(self, image, model="densepose_r50_fpn_dl.torchscript", cmap="Viridis (MagicAnimate)", resolution=512):
        from custom_controlnet_aux.densepose import DenseposeDetector
        model = DenseposeDetector \
                    .from_pretrained(filename=model) \
                    .to(model_management.get_torch_device())
        return (common_annotator_call(model, image, cmap="viridis" if "Viridis" in cmap else "parula", resolution=resolution), )


NODE_CLASS_MAPPINGS = {
    "DensePosePreprocessor": DensePose_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DensePosePreprocessor": "DensePose Estimator"
}