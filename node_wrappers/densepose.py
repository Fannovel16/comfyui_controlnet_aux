from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME, create_node_input_types
import comfy.model_management as model_management

class DensePose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            model=(["densepose_r50_fpn_dl.torchscript", "densepose_r101_fpn_dl.torchscript"], {"default": "densepose_r50_fpn_dl.torchscript"}),
            cmap=(["Viridis (MagicAnimate)", "Parula (CivitAI)"], {"default": "Viridis (MagicAnimate)"})
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Faces and Poses"

    def execute(self, image, model, cmap, resolution=512):
        from controlnet_aux.densepose import DenseposeDetector
        return (common_annotator_call(
                DenseposeDetector.from_pretrained("LayerNorm/DensePose-TorchScript-with-hint-image", model).to(model_management.get_torch_device()), 
                image, 
                cmap="viridis" if "Viridis" in cmap else "parula", 
                resolution=resolution), )


NODE_CLASS_MAPPINGS = {
    "DensePosePreprocessor": DensePose_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DensePosePreprocessor": "DensePose Estimation"
}