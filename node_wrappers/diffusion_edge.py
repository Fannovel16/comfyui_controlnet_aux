from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT, run_script
import comfy.model_management as model_management
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
            environment=INPUT.COMBO(["indoor", "urban", "natrual"]),
            patch_batch_size=INPUT.INT(default=4, min=1, max=16),
            resolution=INPUT.RESOLUTION()
        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Line Extractors"

    def execute(self, image, environment="indoor", patch_batch_size=4, resolution=512, **kwargs):
        install_deps()
        from custom_controlnet_aux.diffusion_edge import DiffusionEdgeDetector

        model = DiffusionEdgeDetector \
            .from_pretrained(filename = f"diffusion_edge_{environment}.pt") \
            .to(model_management.get_torch_device())
        out = common_annotator_call(model, image, resolution=resolution, patch_batch_size=patch_batch_size)
        del model
        return (out, )

NODE_CLASS_MAPPINGS = {
    "DiffusionEdge_Preprocessor": DiffusionEdge_Preprocessor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionEdge_Preprocessor": "Diffusion Edge (batch size ↑ => speed ↑, VRAM ↑)",
}