from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME
import comfy.model_management as model_management

class SAM_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "image": ("IMAGE",) }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/others"

    def execute(self, image, **kwargs):
        from controlnet_aux.segment_anything import SamDetector

        mobile_sam = SamDetector.from_pretrained("dhkim2810/MobileSAM", model_type="vit_t", filename="mobile_sam.pt").to(model_management.get_torch_device())
        out = common_annotator_call(mobile_sam, image)
        del mobile_sam
        return (out, )

NODE_CLASS_MAPPINGS = {
    "SAMPreprocessor": SAM_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMPreprocessor": "SAM Segmentor"
}