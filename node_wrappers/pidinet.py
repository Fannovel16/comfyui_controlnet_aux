from ..utils import common_annotator_call, annotator_ckpts_path, HF_MODEL_NAME
import comfy.model_management as model_management

class PIDINET_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "safe": (["enable", "disable"], {"default": "enable"})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, safe, **kwargs):
        from controlnet_aux.pidi import PidiNetDetector

        model = PidiNetDetector.from_pretrained(HF_MODEL_NAME, cache_dir=annotator_ckpts_path).to(model_management.get_torch_device())
        out = common_annotator_call(model, image, safe = safe == "enable")
        del model
        return (out, )

NODE_CLASS_MAPPINGS = {
    "PiDiNetPreprocessor": PIDINET_Preprocessor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDiNetPreprocessor": "PiDiNet Lines"
}