from ..utils import common_annotator_call, create_node_input_types, run_script
import comfy.model_management as model_management

class EDPF_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types()

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, resolution=512, **kwargs):
        from controlnet_aux.edpf import EDPF
        return (common_annotator_call(EDPF(), image, resolution=resolution), )



NODE_CLASS_MAPPINGS = {
    "EDPFPreprocessor": EDPF_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EDPFPreprocessor": "Edge-Drawing Parameter-Free (EDPF)"
}
