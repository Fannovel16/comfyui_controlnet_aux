from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT, run_script
import comfy.model_management as model_management
import comfy.model_patcher
import torch
import sys

def install_deps():
    try:
        import sklearn
    except:
        run_script([sys.executable, '-s', '-m', 'pip', 'install', 'scikit-learn'])

class DiffusionEdge_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(
            model=("DIFFUSION_EDGE_MODEL",),
            patch_batch_size=INPUT.INT(default=4, min=1, max=16),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, model, patch_batch_size=4, resolution=512, **kwargs):
        install_deps()
        model_management.load_model_gpu(model)
        out = common_annotator_call(model.model, image, resolution=resolution, patch_batch_size=patch_batch_size)
        return (out, )

NODE_CLASS_MAPPINGS = {
    "DiffusionEdge_Preprocessor": DiffusionEdge_Preprocessor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionEdge_Preprocessor": "Diffusion Edge (batch size ↑ => speed ↑, VRAM ↑)",
}
