from ..utils import common_annotator_call


class Tile_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE",), "pyrUp_iters": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/others"

    def execute(self, image, pyrUp_iters, **kwargs):
        from controlnet_aux.tile import TileDetector

        return (common_annotator_call(TileDetector(), image, pyrUp_iters=pyrUp_iters),)


NODE_CLASS_MAPPINGS = {
    "TilePreprocessor": Tile_Preprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TilePreprocessor": "Tile"
}
